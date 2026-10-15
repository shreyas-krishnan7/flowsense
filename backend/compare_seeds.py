"""
Aggregate the seed study from saved predictions (nothing is re-trained or
typed in by hand: every number is recomputed from runs/<run>/predictions.npz).

    python compare_seeds.py
Writes results/seed_study.json and results/seed_study.md
"""
import os, glob, json
import numpy as np
from sklearn.metrics import f1_score

GROUPS = {
    "TCN + SupCon (k-NN)":           ("tcn_supcon_p30_s{}", "y_pred"),
    "TCN + cross-entropy (k-NN)":    ("tcn_ce_p30_s{}", "y_pred"),
    "TCN + cross-entropy (softmax)": ("tcn_ce_p30_s{}", "y_pred_head"),
    "Transformer + cross-entropy (k-NN)":    ("transformer_ce_p30_s{}", "y_pred"),
    "Transformer + cross-entropy (softmax)": ("transformer_ce_p30_s{}", "y_pred_head"),
    "GBDT, ppi (90 values, fair)":   ("gbdt_ppi_p30_s{}", "y_pred"),
    "GBDT, ppi_derived (fair)":      ("gbdt_ppi_derived_p30_s{}", "y_pred"),
}
SEEDS = [0, 1, 2]


def metrics(y, p, n):
    return {"accuracy": float((y == p).mean()),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "per_class_f1": f1_score(y, p, average=None, labels=range(n)).tolist()}


def mean_std(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def main():
    out, names, y_ref = {}, None, None
    for group, (pattern, key) in GROUPS.items():
        per_seed = {}
        for s in SEEDS:
            path = os.path.join("runs", pattern.format(s), "predictions.npz")
            if not os.path.exists(path):
                continue
            b = np.load(path, allow_pickle=True)
            if key not in b.files:
                continue
            names = names or b["label_names"].tolist()
            y = b["y_true"]
            if y_ref is None:
                y_ref = y
            assert np.array_equal(y, y_ref), f"{path}: test labels differ -- splits not identical"
            per_seed[s] = metrics(y, b[key], len(names))
        if not per_seed:
            continue
        acc = [m["accuracy"] for m in per_seed.values()]
        f1 = [m["macro_f1"] for m in per_seed.values()]
        pc = np.array([m["per_class_f1"] for m in per_seed.values()])
        out[group] = {
            "seeds": list(per_seed),
            "accuracy_mean_std": mean_std(acc), "macro_f1_mean_std": mean_std(f1),
            "per_seed": {str(s): {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                         for s, m in per_seed.items()},
            "per_class_f1_mean": dict(zip(names, pc.mean(0).tolist())),
            "per_class_f1_std": dict(zip(names, (pc.std(0, ddof=1) if len(pc) > 1
                                                 else np.zeros(pc.shape[1])).tolist())),
        }

    # paired SupCon - CE differences (same seed, same split)
    a, c = out.get("TCN + SupCon (k-NN)"), out.get("TCN + cross-entropy (k-NN)")
    paired = {}
    if a and c:
        common = sorted(set(a["seeds"]) & set(c["seeds"]))
        for metric in ("accuracy", "macro_f1"):
            d = [a["per_seed"][str(s)][metric] - c["per_seed"][str(s)][metric] for s in common]
            paired[metric] = {"per_seed": d, "mean_std": mean_std(d)}

    os.makedirs("results", exist_ok=True)
    with open("results/seed_study.json", "w") as f:
        json.dump({"groups": out, "supcon_minus_ce": paired, "label_names": names,
                   "n_test_known": int(len(y_ref)) if y_ref is not None else 0,
                   "std": "sample standard deviation (ddof=1) across seeds"}, f, indent=2)

    L = ["# Seed study: TCN SupCon vs cross-entropy, Transformer cross-entropy, and fair GBDT", "",
         f"Known test flows: {len(y_ref):,} (CESNET-QUIC22 weeks 45-47; trained on week 44). "
         "Mean ± sample std over seeds; all numbers recomputed from saved predictions.", "",
         "| Model | Seeds | Accuracy % | Macro-F1 % |", "|---|---|---|---|"]
    for g, r in out.items():
        (am, asd), (fm, fsd) = r["accuracy_mean_std"], r["macro_f1_mean_std"]
        L.append(f"| {g} | {len(r['seeds'])} | {am * 100:.2f} ± {asd * 100:.2f} | {fm * 100:.2f} ± {fsd * 100:.2f} |")
    if paired:
        L += ["", "**SupCon − cross-entropy (k-NN, paired by seed):** " + ", ".join(
            f"{m}: {v['mean_std'][0] * 100:+.2f} ± {v['mean_std'][1] * 100:.2f} pts "
            f"(per seed {', '.join(f'{x * 100:+.2f}' for x in v['per_seed'])})" for m, v in paired.items())]
    L += ["", "## Per-class F1 (mean ± std)", "",
          "| Service | " + " | ".join(out) + " |", "|---|" + "---|" * len(out)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(
            f"{r['per_class_f1_mean'][n] * 100:.1f} ± {r['per_class_f1_std'][n] * 100:.1f}" for r in out.values()) + " |")
    with open("results/seed_study.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
