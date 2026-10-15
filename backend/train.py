import os
# Must be set before numpy is imported -- limits how many threads OpenBLAS
# spawns, each of which grabs its own buffers. Prevents the
# "OpenBLAS ... Memory allocation still failed" crash on memory-constrained
# machines when loading a large real dataset.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, json, time
import numpy as np, torch, faiss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from features import make_synthetic_dataset, fit_stats, transform, MAX_PACKETS, FEATURE_DIM
from model import build_encoder, count_params, SupConLoss, CosineHead
import core


def parse_args():
    ap = argparse.ArgumentParser(description="Train a FlowSense encoder")
    ap.add_argument("--data", default="data/cesnet_top20.npz",
                    help="Cache from prepare_data.py, or 'synthetic' for a zero-download smoke test")
    ap.add_argument("--encoder", default="tcn", help="cnn | tcn | tcn_attn | gru | transformer")
    ap.add_argument("--loss", choices=["supcon", "ce"], default="supcon")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=None, help="Subsample the train set")
    ap.add_argument("--packets", type=int, default=MAX_PACKETS,
                    help="Use only the first N packets of each flow (early classification)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--export", action="store_true",
                    help="Also write ONNX + index + tuned thresholds to artifacts/ for serve.py")
    return ap.parse_args()


def load_data(args):
    if args.data == "synthetic":
        X, y = make_synthetic_dataset()
        idx = np.random.default_rng(0).permutation(len(y)); X, y = X[idx], y[idx]
        n1, n2 = int(0.7 * len(y)), int(0.8 * len(y))
        names = [str(c) for c in np.unique(y)]
        d = dict(X_train=X[:n1], y_train=y[:n1], X_val=X[n1:n2], y_val=y[n1:n2],
                 X_test=X[n2:], y_test=y[n2:], t_test=np.zeros(len(y) - n2, np.int64),
                 app_test=np.array([names[c] for c in y[n2:]]))
        return d, names
    b = np.load(args.data, allow_pickle=True)
    d = {k: b[k] for k in ["X_train", "y_train", "X_val", "y_val",
                           "X_test", "y_test", "t_test", "app_test"]}
    return d, b["label_names"].tolist()


def truncate(X, n):
    if n >= X.shape[1]:
        return X
    X = X.copy(); X[:, n:] = 0      # same as a flow that ended after n packets
    return X


@torch.no_grad()
def embed(model, X, dev, batch=4096):
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch]).to(dev)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    name = args.run_name or f"{args.encoder}_{args.loss}_p{args.packets}_s{args.seed}"
    run_dir = os.path.join("runs", name)
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        raise SystemExit(f"{run_dir} already exists -- refusing to overwrite (pass --run-name)")
    os.makedirs(run_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Data
    d, label_names = load_data(args)
    n_classes = len(label_names)
    if args.max_train and len(d["y_train"]) > args.max_train:
        keep = np.random.default_rng(args.seed).choice(len(d["y_train"]), args.max_train, replace=False)
        d["X_train"], d["y_train"] = d["X_train"][keep], d["y_train"][keep]
    for s in ["train", "val", "test"]:
        d[f"X_{s}"] = truncate(d[f"X_{s}"], args.packets)
    mean, std = fit_stats(d["X_train"])
    Xtr, Xva, Xte = (transform(d[f"X_{s}"], mean, std).astype(np.float32)
                     for s in ["train", "val", "test"])
    ytr, yva, yte = d["y_train"], d["y_val"], d["y_test"]
    print(f"[{name}] {len(ytr):,} train / {len(yva):,} val / {len(yte):,} test "
          f"({(yte == -1).sum():,} unknown) | {n_classes} classes | device={dev}")

    # 2. Model + objective
    model = build_encoder(args.encoder).to(dev)
    head = CosineHead(128, n_classes).to(dev) if args.loss == "ce" else None
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    print(f"encoder={args.encoder} params={count_params(model):,} loss={args.loss}")
    crit = SupConLoss() if args.loss == "supcon" else torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=args.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * len(loader))
    use_amp = dev == "cuda"

    # 3. Train
    t0 = time.time(); history = []
    for epoch in range(args.epochs):
        model.train(); tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                z = model(xb)
            z = z.float()
            loss = crit(z, yb) if head is None else crit(head(z), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        history.append(tot / len(loader))
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {history[-1]:.4f}  "
              f"({time.time() - t0:.0f}s)")
    train_time = time.time() - t0

    # 4. Embed everything, k-NN on known test flows
    emb_tr, emb_va, emb_te = (embed(model, X, dev) for X in (Xtr, Xva, Xte))
    index = core.build_index(emb_tr)
    known = yte >= 0
    D, I = core.knn_search(index, emb_te[known], args.k)
    pred, conf, gap = core.knn_vote(D, I, ytr, n_classes)
    acc = float((pred == yte[known]).mean())
    f1 = float(f1_score(yte[known], pred, average="macro"))
    per_class = f1_score(yte[known], pred, average=None, labels=range(n_classes))
    print(f"\nk-NN on known test flows: accuracy {acc * 100:.2f}%  macro-F1 {f1 * 100:.2f}%")
    preds = {"y_true": yte[known], "y_pred": pred, "label_names": np.array(label_names, dtype=object)}
    head_metrics = {}
    if head is not None:
        # secondary result: the CE model's own softmax head on the same test flows
        head.eval()
        emb_known = emb_te[known]
        with torch.no_grad():
            logits = torch.cat([head(torch.from_numpy(emb_known[s:s + 8192]).to(dev)).cpu()
                                for s in range(0, len(emb_known), 8192)])
        head_pred = logits.argmax(1).numpy()
        preds["y_pred_head"] = head_pred
        head_metrics = {"head_accuracy": float((head_pred == yte[known]).mean()),
                        "head_macro_f1": float(f1_score(yte[known], head_pred, average="macro"))}
        print(f"CE softmax head:          accuracy {head_metrics['head_accuracy'] * 100:.2f}%  "
              f"macro-F1 {head_metrics['head_macro_f1'] * 100:.2f}%")

    # 5. Save run (evaluate.py computes the full set of experiments from this)
    np.savez(os.path.join(run_dir, "embeddings.npz"),
             emb_train=emb_tr, y_train=ytr, emb_val=emb_va, y_val=yva,
             emb_test=emb_te, y_test=yte, t_test=d["t_test"], app_test=d["app_test"],
             label_names=np.array(label_names, dtype=object))
    np.savez(os.path.join(run_dir, "predictions.npz"), **preds)
    torch.save({"model": model.state_dict(), "head": head.state_dict() if head else None,
                "args": vars(args), "mean": mean, "std": std},
               os.path.join(run_dir, "model.pt"))
    summary = {"run": name, **vars(args), "params": count_params(model),
               "n_train": int(len(ytr)), "n_classes": n_classes,
               "train_time_s": round(train_time, 1), "loss_history": history,
               "knn_accuracy": acc, "knn_macro_f1": f1,
               "per_class_f1": {n: float(v) for n, v in zip(label_names, per_class)},
               **head_metrics}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {run_dir}/ (model.pt, embeddings.npz, predictions.npz, summary.json)")

    # 6. Optional export for serving, thresholds tuned on validation data only
    if args.export:
        Dv, Iv = core.knn_search(index, emb_va, args.k)
        _, _, gap_v = core.knn_vote(Dv, Iv, ytr, n_classes)
        novelty_thresh = core.threshold_at_tpr(core.knn_novelty(Dv), 0.95)
        gap_thresh = float(np.quantile(gap_v, 0.10))     # route ~10% least certain to review
        os.makedirs("artifacts", exist_ok=True)
        model_cpu = model.to("cpu").eval()
        dummy = torch.randn(1, MAX_PACKETS, FEATURE_DIM)
        torch.onnx.export(model_cpu, dummy, "artifacts/encoder.onnx",
                          input_names=["flow"], output_names=["embedding"],
                          dynamic_axes={"flow": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=17, dynamo=False)
        faiss.write_index(index, "artifacts/knn.index")
        np.savez("artifacts/bundle.npz", labels=ytr, mean=mean, std=std, k=args.k,
                 novelty_threshold=novelty_thresh, gap_threshold=gap_thresh,
                 label_names=np.array(label_names, dtype=object))
        print(f"Exported artifacts/ (novelty_threshold={novelty_thresh:.4f}, "
              f"gap_threshold={gap_thresh:.4f})")


if __name__ == "__main__":
    main()
