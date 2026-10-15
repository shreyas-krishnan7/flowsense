"""
Evaluation gaps for the selected model:
  calibration      ECE + reliability diagram (k-NN vote confidence, softmax head)
  open-set F1      known + unknown test flows together; unknown rejected at the
                   serving threshold (95% known-flow retention on validation)
  breakdowns       unknown-detection rate per provider / service category;
                   false-alarm rate per known service
  source-IP split  accuracy on test flows whose source IP never occurs in training

    python eval_extra.py runs/transformer_ce_p30_s1
Writes <run>/eval_extra.json and plots/reliability.png
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import core

SERVICEMAP = "C:/datasets/CESNET-QUIC22/M/servicemap.csv"
META = "data/cesnet_top20_meta.npz"


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(out), rows


def softmax_probs(run_dir, emb):
    import torch
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    if not ck.get("head"):
        return None
    w = ck["head"]["w"].numpy()
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    logits = 16.0 * emb @ w.T                       # CosineHead scale
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); a = ap.parse_args()
    e = np.load(os.path.join(a.run_dir, "embeddings.npz"), allow_pickle=True)
    s = json.load(open(os.path.join(a.run_dir, "summary.json")))
    names = e["label_names"].tolist(); C = len(names); k = s["k"]
    yte, app = e["y_test"], e["app_test"]; known = yte >= 0
    index = core.build_index(e["emb_train"])
    D, I = core.knn_search(index, e["emb_test"], k)
    pred, conf, gap = core.knn_vote(D, I, e["y_train"], C)
    res = {"run": s["run"]}

    # ---- calibration (known test flows) ----
    corr = (pred[known] == yte[known]).astype(float)
    res["calibration"] = {}
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--", lw=.7)
    e_knn, rows = ece(conf[known], corr)
    res["calibration"]["knn_vote_ece"] = e_knn
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label=f"k-NN vote (ECE {e_knn:.3f})")
    P = softmax_probs(a.run_dir, e["emb_test"][known])
    if P is not None:
        sp, sc = P.argmax(1), P.max(1)
        e_sm, rows = ece(sc, (sp == yte[known]).astype(float))
        res["calibration"]["softmax_ece"] = e_sm
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "s-", label=f"softmax (ECE {e_sm:.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Reliability (known test flows)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    os.makedirs(os.path.join(a.run_dir, "plots"), exist_ok=True)
    fig.savefig(os.path.join(a.run_dir, "plots", "reliability.png"), dpi=130); plt.close(fig)

    # ---- open-set F1 at the serving threshold ----
    Dv, _ = core.knn_search(index, e["emb_val"], k)
    thr = core.threshold_at_tpr(core.knn_novelty(Dv), 0.95)
    flagged = core.knn_novelty(D) > thr
    pred_os = np.where(flagged, -1, pred)
    labels = list(range(C)) + [-1]
    f1_all = f1_score(yte, pred_os, labels=labels, average=None)
    res["open_set"] = {
        "novelty_threshold": thr,
        "open_set_macro_f1_21cls": float(f1_all.mean()),
        "unknown_class_f1": float(f1_all[-1]),
        "known_macro_f1_with_rejection": float(f1_all[:-1].mean()),
        "closed_set_macro_f1_no_rejection": float(f1_score(yte[known], pred[known], average="macro")),
        "unknown_recall": float(flagged[~known].mean()),
        "known_false_alarm": float(flagged[known].mean()),
        "accuracy_on_accepted_known": float((pred[known & ~flagged] == yte[known & ~flagged]).mean()),
    }

    # ---- provider / category breakdowns ----
    sm = pd.read_csv(SERVICEMAP).set_index("Tag")
    prov = sm["Service Provider"].fillna("(independent)").to_dict()
    cat = sm["Service Category"].fillna("-").to_dict()
    unk = pd.DataFrame({"app": app[~known], "flag": flagged[~known]})
    unk["provider"] = unk.app.map(lambda x: prov.get(x, "(unmapped)"))
    unk["category"] = unk.app.map(lambda x: cat.get(x, "(unmapped)"))
    agg = lambda g: unk.groupby(g).flag.agg(["mean", "size"]).sort_values("size", ascending=False)
    res["unknown_detection_by_provider"] = {i: {"detected": round(float(r["mean"]), 4), "n": int(r["size"])}
                                            for i, r in agg("provider").iterrows()}
    res["unknown_detection_by_category"] = {i: {"detected": round(float(r["mean"]), 4), "n": int(r["size"])}
                                            for i, r in agg("category").iterrows()}
    res["unknown_detection_by_service_top15"] = {i: {"detected": round(float(r["mean"]), 4), "n": int(r["size"])}
                                                 for i, r in agg("app").head(15).iterrows()}
    res["known_false_alarm_by_service"] = {names[c]: round(float(flagged[yte == c].mean()), 4) for c in range(C)}

    # ---- source-IP separation ----
    if os.path.exists(META):
        m = np.load(META, allow_pickle=True)
        seen = set(m["src_ip_train"].tolist())
        src = m["src_ip_test"]
        assert len(src) == len(yte), "metadata not aligned with this run's test set"
        new_ip = np.array([x not in seen for x in src])
        kn_new, kn_seen = known & new_ip, known & ~new_ip
        res["source_ip_separation"] = {
            "unique_src_ips_train": len(seen),
            "test_known_with_unseen_src_ip": float(new_ip[known].mean()),
            "accuracy_unseen_src_ip": float((pred[kn_new] == yte[kn_new]).mean()) if kn_new.any() else None,
            "accuracy_seen_src_ip": float((pred[kn_seen] == yte[kn_seen]).mean()) if kn_seen.any() else None,
            "macro_f1_unseen_src_ip": float(f1_score(yte[kn_new], pred[kn_new], average="macro")) if kn_new.any() else None,
            "n_unseen": int(kn_new.sum()), "n_seen": int(kn_seen.sum()),
        }
    else:
        res["source_ip_separation"] = {"note": f"{META} missing -- run extract_meta.py"}

    with open(os.path.join(a.run_dir, "eval_extra.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k_: v for k_, v in res.items() if "by_service" not in k_}, indent=2))


if __name__ == "__main__":
    main()
