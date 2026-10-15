"""
Gradient-boosted-tree baseline on the same cached split as train.py.

Feature sets (--features):
  ppi          FAIR, default. Exactly the TCN's input: first 30 packets x
               [log size, log inter-packet time, direction] = 90 values.
  ppi_derived  FAIR. ppi + aggregates computed ONLY from those same 30 packets
               (counts, bytes per direction, duration, size/IAT statistics).
  full         UNFAIR, legacy. ppi + CESNET whole-flow statistics (BYTES,
               PACKETS, DURATION, end reason ...) that describe the entire
               flow, i.e. information not available after 30 packets. Kept
               only to reproduce the original runs/gbdt number.

Early stopping uses the official validation split (week 44), never test data.
Outputs go to runs/gbdt_<features>_s<seed>/ (summary.json, predictions.npz);
earlier runs are never overwritten.

    python baseline.py --features ppi --seed 0
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import argparse, json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from features import MAX_PACKETS


def ppi_features(X, packets):
    X = X[:, :packets]
    return np.concatenate([np.log1p(np.clip(X[..., 0], 0, None)),      # sizes
                           np.log1p(np.clip(X[..., 1], 0, None)),      # inter-packet times
                           X[..., 2]], axis=1)                         # directions


def derived_features(X, packets):
    """Aggregates over the same first `packets` packets (padding = size 0)."""
    X = X[:, :packets]
    size, iat, d = X[..., 0], np.clip(X[..., 1], 0, None), X[..., 2]
    valid = size > 0
    up, down = valid & (d > 0), valid & (d < 0)
    n = valid.sum(1)

    def masked_mean(v, m): return (v * m).sum(1) / np.maximum(m.sum(1), 1)
    def masked_max(v, m): return np.where(m, v, 0).max(1)

    feats = [n, up.sum(1), down.sum(1),
             (size * up).sum(1), (size * down).sum(1),
             (iat * valid).sum(1),                                    # duration of these packets
             masked_mean(size, up), masked_mean(size, down),
             masked_max(size, up), masked_max(size, down),
             masked_mean(iat, valid), masked_max(iat, valid),
             ((d[:, 1:] != d[:, :-1]) & valid[:, 1:]).sum(1)]         # direction changes (real packets only)
    F = np.stack(feats, axis=1).astype(np.float32)
    return np.log1p(np.clip(F, 0, None))


def make_features(b, split, kind, packets, mask=None):
    X = b[f"X_{split}"] if mask is None else b[f"X_{split}"][mask]
    parts = [ppi_features(X, packets)]
    if kind == "ppi_derived":
        parts.append(derived_features(X, packets))
    elif kind == "full":
        stats = b[f"stats_{split}"] if mask is None else b[f"stats_{split}"][mask]
        parts.append(np.log1p(np.clip(stats, 0, None)))
    return np.concatenate(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cesnet_top20.npz")
    ap.add_argument("--features", choices=["ppi", "ppi_derived", "full"], default="ppi")
    ap.add_argument("--packets", type=int, default=MAX_PACKETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    name = a.run_name or f"gbdt_{a.features}_p{a.packets}_s{a.seed}"
    run_dir = os.path.join("runs", name)
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        raise SystemExit(f"{run_dir} already exists -- refusing to overwrite (pass --run-name)")

    b = np.load(a.data, allow_pickle=True)
    names = b["label_names"].tolist()
    ytr, yva, yte = b["y_train"], b["y_val"], b["y_test"]
    known = yte >= 0
    Xtr = make_features(b, "train", a.features, a.packets)
    Xva = make_features(b, "val", a.features, a.packets)
    Xte = make_features(b, "test", a.features, a.packets, mask=known)
    yte = yte[known]
    if a.max_train and len(ytr) > a.max_train:
        keep = np.random.default_rng(a.seed).choice(len(ytr), a.max_train, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"[{name}] features={a.features} ({Xtr.shape[1]} dims) | {len(ytr):,} train / "
          f"{len(yva):,} val / {len(yte):,} known test")

    maj = np.bincount(ytr).argmax()
    maj_acc = float((yte == maj).mean())

    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=a.max_iter, learning_rate=0.1,
                                         max_leaf_nodes=63, early_stopping=True,
                                         n_iter_no_change=20, random_state=a.seed)
    clf.fit(Xtr, ytr, X_val=Xva, y_val=yva)
    train_time = time.time() - t0
    pred = clf.predict(Xte)
    acc = float((pred == yte).mean())
    f1 = float(f1_score(yte, pred, average="macro"))
    per_class = f1_score(yte, pred, average=None, labels=range(len(names)))
    print(f"GBDT: {clf.n_iter_} iters, {train_time:.0f}s | accuracy {acc * 100:.2f}%  "
          f"macro-F1 {f1 * 100:.2f}%  (majority-class accuracy {maj_acc * 100:.2f}%)")

    os.makedirs(run_dir, exist_ok=True)
    np.savez(os.path.join(run_dir, "predictions.npz"), y_true=yte, y_pred=pred,
             label_names=np.array(names, dtype=object))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"run": name, "model": "gbdt", "encoder": "gbdt", "loss": "-",
                   "features": a.features, "fair": a.features != "full",
                   "packets": a.packets, "seed": a.seed, "params": 0,
                   "n_features": int(Xtr.shape[1]), "n_train": int(len(ytr)),
                   "n_iter": int(clf.n_iter_), "train_time_s": round(train_time, 1),
                   "accuracy": acc, "macro_f1": f1,
                   "per_class_f1": {n: float(v) for n, v in zip(names, per_class)},
                   "majority_accuracy": maj_acc}, f, indent=2)
    print(f"Saved {run_dir}/")


if __name__ == "__main__":
    main()
