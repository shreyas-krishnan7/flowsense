"""
Model selection on the VALIDATION split only (week 44 flows held out from
training). Test numbers are listed for reference but never used to choose.

For every deep run in runs/*/embeddings.npz, the k-NN decision layer is
rebuilt from the training embeddings and scored on the validation embeddings.
Configurations (encoder + loss) are ranked by mean validation macro-F1 over
the seeds available.

    python select_model.py
Writes results/model_selection.json and results/model_selection.md
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import glob, json, re
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

import core


def score_run(run_dir):
    with open(os.path.join(run_dir, "summary.json")) as f:
        s = json.load(f)
    e = np.load(os.path.join(run_dir, "embeddings.npz"), allow_pickle=True)
    n = len(e["label_names"])
    index = core.build_index(e["emb_train"])
    D, I = core.knn_search(index, e["emb_val"], s["k"])
    pred, _, _ = core.knn_vote(D, I, e["y_train"], n)
    yv = e["y_val"]
    return s, {"val_accuracy": float((pred == yv).mean()),
               "val_macro_f1": float(f1_score(yv, pred, average="macro")),
               "test_accuracy": s["knn_accuracy"], "test_macro_f1": s["knn_macro_f1"]}


def main():
    rows, by_cfg = [], defaultdict(list)
    for p in sorted(glob.glob("runs/*/embeddings.npz")):
        run_dir = os.path.dirname(p)
        s, m = score_run(run_dir)
        if s.get("packets", 30) != 30 or s.get("max_train") or not re.search(r"_s\d+$", s["run"]):
            continue      # only full-input, seed-tagged runs (legacy 'tcn_supcon_p30' duplicates seed 0)
        cfg = f"{s['encoder']} + {s['loss']}"
        rows.append({"run": s["run"], "config": cfg, "seed": s["seed"], "params": s["params"], **m})
        by_cfg[cfg].append(rows[-1])
        print(f"{s['run']:28s} val acc {m['val_accuracy'] * 100:.2f}  val F1 {m['val_macro_f1'] * 100:.2f}")

    summary = []
    for cfg, rs in by_cfg.items():
        def ms(k):
            v = np.array([r[k] for r in rs])
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        summary.append({"config": cfg, "n_seeds": len(rs), "params": rs[0]["params"],
                        **{k: ms(k) for k in ["val_accuracy", "val_macro_f1",
                                              "test_accuracy", "test_macro_f1"]}})
    summary.sort(key=lambda r: -r["val_macro_f1"][0])
    chosen = summary[0]

    os.makedirs("results", exist_ok=True)
    with open("results/model_selection.json", "w") as f:
        json.dump({"criterion": "mean validation macro-F1 (k-NN on embeddings), week-44 held-out flows",
                   "selected": chosen["config"], "configs": summary, "runs": rows}, f, indent=2)
    L = ["# Model selection (validation split only)", "",
         "Criterion: mean **validation** macro-F1 of the k-NN decision layer, over available seeds. "
         "Validation = 30,000 week-44 flows held out from training. Test columns are for reference only.", "",
         "| Rank | Config | Seeds | Params | Val acc % | Val macro-F1 % | Test acc % | Test macro-F1 % |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(summary, 1):
        f = lambda k: f"{r[k][0] * 100:.2f}" + (f" ± {r[k][1] * 100:.2f}" if r["n_seeds"] > 1 else "")
        L.append(f"| {i} | {r['config']} | {r['n_seeds']} | {r['params']:,} | {f('val_accuracy')} | "
                 f"{f('val_macro_f1')} | {f('test_accuracy')} | {f('test_macro_f1')} |")
    L += ["", f"**Selected: {chosen['config']}**"]
    with open("results/model_selection.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
