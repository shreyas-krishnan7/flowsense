"""
Export a trained run for serving (no retraining):
  artifacts/encoder.onnx   encoder, dynamic batch
  artifacts/knn.index      FAISS index of the run's training embeddings
  artifacts/bundle.npz     labels, normalization stats, k, thresholds (validation-tuned)
  artifacts/manifest.json  which run/seed was exported, thresholds, parity check

The previous artifacts/ directory is moved to artifacts_backup_<timestamp>/ first.

    python export.py runs/transformer_ce_p30_s1
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse, json, shutil, time
import numpy as np, torch, faiss
import onnxruntime as ort

from features import transform, MAX_PACKETS, FEATURE_DIM
from model import build_encoder
import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--data", default="data/cesnet_top20.npz")
    ap.add_argument("--index", choices=["exact", "hnsw"], default="exact",
                    help="hnsw: approximate search, ~70x faster; accuracy change measured in results/index_comparison.json")
    ap.add_argument("--ef-search", type=int, default=128)
    a = ap.parse_args()

    ck = torch.load(os.path.join(a.run_dir, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    if args.get("packets", MAX_PACKETS) != MAX_PACKETS:
        raise SystemExit("serving expects full-length (30-packet) models")
    model = build_encoder(args["encoder"]); model.load_state_dict(ck["model"]); model.eval()
    e = np.load(os.path.join(a.run_dir, "embeddings.npz"), allow_pickle=True)
    names = e["label_names"].tolist(); k = args["k"]

    # thresholds from the validation split only, using the same index type that will be served
    if a.index == "hnsw":
        index = faiss.IndexHNSWFlat(e["emb_train"].shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(np.ascontiguousarray(e["emb_train"], np.float32))
        index.hnsw.efSearch = a.ef_search
    else:
        index = core.build_index(e["emb_train"])
    Dv, Iv = core.knn_search(index, e["emb_val"], k)
    _, _, gap_v = core.knn_vote(Dv, Iv, e["y_train"], len(names))
    novelty_thr = core.threshold_at_tpr(core.knn_novelty(Dv), 0.95)
    gap_thr = float(np.quantile(gap_v, 0.10))

    if os.path.exists(a.out):
        backup = f"{a.out}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(a.out, backup); print(f"Moved existing {a.out}/ -> {backup}/")
    os.makedirs(a.out)
    onnx_path = os.path.join(a.out, "encoder.onnx")
    torch.onnx.export(model, torch.randn(1, MAX_PACKETS, FEATURE_DIM), onnx_path,
                      input_names=["flow"], output_names=["embedding"],
                      dynamic_axes={"flow": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)
    faiss.write_index(index, os.path.join(a.out, "knn.index"))
    np.savez(os.path.join(a.out, "bundle.npz"), labels=e["y_train"], mean=ck["mean"], std=ck["std"],
             k=k, novelty_threshold=novelty_thr, gap_threshold=gap_thr,
             ef_search=a.ef_search if a.index == "hnsw" else 0,
             label_names=np.array(names, dtype=object))

    # parity on real test flows: ONNX vs PyTorch, and vs the run's stored embeddings
    X = np.load(a.data, allow_pickle=True)["X_test"][:2048]
    x = transform(X, ck["mean"], ck["std"]).astype(np.float32)
    with torch.no_grad():
        z_pt = model(torch.from_numpy(x)).numpy()
    z_onnx = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"flow": x})[0]
    parity = {"onnx_vs_pytorch_max_abs": float(np.abs(z_onnx - z_pt).max()),
              "onnx_vs_stored_max_abs": float(np.abs(z_onnx - e["emb_test"][:2048]).max()),
              "top1_label_agreement": float((core.knn_vote(*core.knn_search(index, z_onnx, k), e["y_train"], len(names))[0] ==
                                             core.knn_vote(*core.knn_search(index, e["emb_test"][:2048], k), e["y_train"], len(names))[0]).mean())}
    manifest = {"run": os.path.basename(os.path.normpath(a.run_dir)), "encoder": args["encoder"],
                "loss": args["loss"], "seed": args["seed"], "k": k, "n_index": int(index.ntotal),
                "index": a.index, "ef_search": a.ef_search if a.index == "hnsw" else None,
                "novelty_threshold": novelty_thr, "gap_threshold": gap_thr,
                "threshold_policy": "novelty: 95% known-flow retention on validation; review: 10% lowest validation gap",
                "parity": parity, "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
