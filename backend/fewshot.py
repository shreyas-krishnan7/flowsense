"""
E7 (revised): few-shot registration of unseen services, prototype-based.

Each known service is represented by the L2-normalized mean of its training
embeddings; each new service by the mean of K labeled examples. A flow goes to
the nearest prototype (cosine). The previous k-NN method is run on the same
services and draws for comparison.

New services are taken from the test-set *unknown* services with at most ONE
service per provider (CESNET servicemap), so the task is not "tell ten Google
services apart".

Success criterion (fixed before running): at K=10, new-service accuracy >= 70%
AND known-service accuracy drops by <= 2 points vs. the 0-shot prototype model.

    python fewshot.py runs/transformer_ce_p30_s0
Writes <run>/fewshot.json and <run>/plots/few_shot_prototype.png
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core

SERVICEMAP = "C:/datasets/CESNET-QUIC22/M/servicemap.csv"
SHOTS, TRIALS, N_NEW, N_KNOWN_QUERY = (1, 5, 10, 20), 5, 10, 20000
CRITERION = {"K": 10, "min_new_acc": 0.70, "max_known_drop": 0.02}


def pick_new_services(apps, counts, servicemap):
    """Most frequent unknown services, at most one per provider."""
    provider = servicemap.set_index("Tag")["Service Provider"].to_dict()
    chosen, used = [], set()
    for a in apps[np.argsort(-counts)]:
        p = provider.get(a)
        p = a if (p is None or (isinstance(p, float) and np.isnan(p))) else p
        if p in used:
            continue
        chosen.append(a); used.add(p)
        if len(chosen) == N_NEW:
            break
    return chosen, {a: (provider.get(a) if isinstance(provider.get(a), str) else "-") for a in chosen}


def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    e = np.load(os.path.join(a.run_dir, "embeddings.npz"), allow_pickle=True)
    k = json.load(open(os.path.join(a.run_dir, "summary.json")))["k"]
    names = e["label_names"].tolist(); C = len(names)
    emb_tr, y_tr, emb_te, y_te, app_te = e["emb_train"], e["y_train"], e["emb_test"], e["y_test"], e["app_test"]

    unk = y_te == -1
    apps, counts = np.unique(app_te[unk], return_counts=True)
    keep = counts >= max(SHOTS) + 30
    new_apps, prov = pick_new_services(apps[keep], counts[keep], pd.read_csv(SERVICEMAP))
    print("New services:", {s: prov[s] for s in new_apps})

    known_protos = core.class_prototypes(emb_tr, y_tr, C)
    rng = np.random.default_rng(0)
    kq = rng.choice(np.where(~unk)[0], min(N_KNOWN_QUERY, int((~unk).sum())), replace=False)
    base_known = float(((emb_te[kq] @ known_protos.T).argmax(1) == y_te[kq]).mean())
    knn_index = core.build_index(emb_tr)
    res = {"new_services": new_apps, "providers": prov,
           "known_acc_0shot_prototype": base_known, "criterion": CRITERION, "by_K": {}}

    for K in SHOTS:
        acc = {"proto_new": [], "proto_known": [], "knn_new": [], "knn_known": []}
        for t in range(TRIALS):
            r = np.random.default_rng(100 + t)
            sup, qi, qy = [], [], []
            for j, s in enumerate(new_apps):
                idx = r.permutation(np.where(unk & (app_te == s))[0])
                sup.append(emb_te[idx[:K]]); qi.append(idx[K:]); qy += [C + j] * (len(idx) - K)
            qi, qy = np.concatenate(qi), np.array(qy)
            # prototype classifier
            P = np.concatenate([known_protos, norm(np.stack([s.mean(0) for s in sup]))])
            acc["proto_new"].append(float(((emb_te[qi] @ P.T).argmax(1) == qy).mean()))
            acc["proto_known"].append(float(((emb_te[kq] @ P.T).argmax(1) == y_te[kq]).mean()))
            # previous k-NN method, same draws
            idx_k = core.build_index(np.concatenate([emb_tr] + sup))
            lab = np.concatenate([y_tr, np.repeat(np.arange(C, C + len(new_apps)), K)])
            kk = min(k, K)
            D, I = core.knn_search(idx_k, emb_te[qi], kk)
            acc["knn_new"].append(float((core.knn_vote(D, I, lab, C + len(new_apps))[0] == qy).mean()))
            D, I = core.knn_search(idx_k, emb_te[kq], k)
            acc["knn_known"].append(float((core.knn_vote(D, I, lab, C + len(new_apps))[0] == y_te[kq]).mean()))
        res["by_K"][K] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for m, v in acc.items()}
        print(f"K={K:2d}  prototype: new {np.mean(acc['proto_new']) * 100:5.1f}%  known {np.mean(acc['proto_known']) * 100:5.1f}%"
              f"  |  k-NN: new {np.mean(acc['knn_new']) * 100:5.1f}%  known {np.mean(acc['knn_known']) * 100:5.1f}%")

    r10 = res["by_K"][CRITERION["K"]]
    passed = (r10["proto_new"]["mean"] >= CRITERION["min_new_acc"] and
              base_known - r10["proto_known"]["mean"] <= CRITERION["max_known_drop"])
    res["criterion_passed"] = bool(passed)
    print(f"\nCriterion (K=10: new >= 70%, known drop <= 2 pts): {'PASSED' if passed else 'FAILED'} "
          f"(new {r10['proto_new']['mean'] * 100:.1f}%, known {r10['proto_known']['mean'] * 100:.1f}% "
          f"vs 0-shot {base_known * 100:.1f}%)")

    os.makedirs(os.path.join(a.run_dir, "plots"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, st in [("proto_new", "o-"), ("knn_new", "s--"), ("proto_known", "o:")]:
        ax.plot(SHOTS, [res["by_K"][K][m]["mean"] * 100 for K in SHOTS], st, label=m.replace("_", " "))
    ax.set_xlabel("Labeled examples per new service (K)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Few-shot registration, {len(new_apps)} services from different providers")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(a.run_dir, "plots", "few_shot_prototype.png"), dpi=130); plt.close(fig)
    with open(os.path.join(a.run_dir, "fewshot.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)


if __name__ == "__main__":
    main()
