"""
Full evaluation of a trained run (reads runs/<name>/embeddings.npz, no GPU needed).

    python evaluate.py runs/tcn_supcon_p30            # E1, E5, E6, E7, E8 + plots
    python evaluate.py runs/tcn_supcon_p30 --tsne     # also E10 t-SNE plot
    python evaluate.py --table                        # comparison table of all runs

E1  closed-set classification   accuracy, macro-F1, per-class F1, confusion matrix
E5  selective classification    accuracy vs coverage when routing low-gap flows to review
E6  open-set detection          unknown-service detection: AUROC, FPR@95%TPR (3 detectors)
E7  few-shot registration       accuracy on unseen services from K labeled examples
E8  temporal drift              accuracy per test day / week
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix

import core


def load(run_dir):
    b = np.load(os.path.join(run_dir, "embeddings.npz"), allow_pickle=True)
    with open(os.path.join(run_dir, "summary.json")) as f:
        s = json.load(f)
    return {k: b[k] for k in b.files}, s


# ---------- E1 ----------
def e1_closed_set(e, k, n_classes, names, plot_dir):
    known = e["y_test"] >= 0
    D, I = core.knn_search(e["index"], e["emb_test"][known], k)
    pred, conf, gap = core.knn_vote(D, I, e["y_train"], n_classes)
    y = e["y_test"][known]
    per_class = f1_score(y, pred, average=None, labels=range(n_classes))
    cm = confusion_matrix(y, pred, labels=range(n_classes), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (row-normalized)")
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "confusion_matrix.png"), dpi=130); plt.close(fig)
    off = cm.copy(); np.fill_diagonal(off, 0)
    worst = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:5], off.shape))[0]
    return {"accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class_f1": {names[i]: round(float(v), 4) for i, v in enumerate(per_class)},
            "top_confusions": [{"true": names[a], "pred": names[b], "rate": round(float(off[a, b]), 4)}
                               for a, b in worst]}, (pred, gap, y)


# ---------- E5 ----------
def e5_selective(pred, gap, y, plot_dir):
    order = np.argsort(-gap)                    # most confident first
    correct = (pred[order] == y[order]).astype(np.float32)
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    cov = np.arange(1, len(correct) + 1) / len(correct)
    at = {f"{int(c * 100)}%": round(float(cum_acc[int(c * len(cum_acc)) - 1]), 4)
          for c in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cov * 100, cum_acc * 100)
    ax.set_xlabel("Coverage (% of flows auto-classified)"); ax.set_ylabel("Accuracy on accepted flows (%)")
    ax.set_title("Selective classification (confidence-gap routing)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "selective.png"), dpi=130); plt.close(fig)
    return {"accuracy_at_coverage": at, "aurc": float(1 - cum_acc.mean())}


# ---------- E6 ----------
def e6_open_set(e, k, n_classes, plot_dir):
    is_unk = e["y_test"] == -1
    if is_unk.sum() == 0:
        return {"note": "no unknown flows in test set"}
    protos = core.class_prototypes(e["emb_train"], e["y_train"], n_classes)
    dcore = core.dbscan_core(e["emb_train"])
    Dv, _ = core.knn_search(e["index"], e["emb_val"], k)
    Dt, _ = core.knn_search(e["index"], e["emb_test"], k)
    detectors = {
        "knn_distance": (core.knn_novelty(Dv), core.knn_novelty(Dt)),
        "prototype": (core.proto_novelty(e["emb_val"], protos), core.proto_novelty(e["emb_test"], protos)),
        "dbscan_core": (core.dbscan_novelty(e["emb_val"], dcore), core.dbscan_novelty(e["emb_test"], dcore)),
    }
    out = {}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (sv, st) in detectors.items():
        thr = core.threshold_at_tpr(sv, 0.95)    # chosen on validation (known only)
        auroc = roc_auc_score(is_unk, st)
        out[name] = {"auroc": round(float(auroc), 4),
                     "known_kept_at_val_thr": round(float((st[~is_unk] <= thr).mean()), 4),
                     "unknown_detected_at_val_thr": round(float((st[is_unk] > thr).mean()), 4),
                     "fpr_at_95tpr": round(float(_fpr_at_tpr(st, is_unk)), 4)}
        fpr, tpr, _ = roc_curve(is_unk, st)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.7)
    ax.set_xlabel("False positive rate (known flagged)"); ax.set_ylabel("True positive rate (unknown flagged)")
    ax.set_title("Unknown-service detection"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "open_set_roc.png"), dpi=130); plt.close(fig)
    return out


def _fpr_at_tpr(scores, is_pos, tpr_target=0.95):
    """FPR on known flows when 95% of unknown flows are detected."""
    thr = np.quantile(scores[is_pos], 1 - tpr_target)
    return (scores[~is_pos] >= thr).mean()


# ---------- E7 ----------
def e7_few_shot(e, k, n_classes, shots=(1, 5, 10, 20), n_new=10, trials=5, plot_dir=None):
    unk = e["y_test"] == -1
    if unk.sum() == 0:
        return {"note": "no unknown flows in test set"}
    apps, counts = np.unique(e["app_test"][unk], return_counts=True)
    eligible = apps[counts >= max(shots) + 30]
    new_apps = eligible[np.argsort(-counts[np.isin(apps, eligible)])][:n_new]
    rng = np.random.default_rng(0)
    known_idx = np.where(e["y_test"] >= 0)[0]
    known_idx = rng.choice(known_idx, min(20000, len(known_idx)), replace=False)
    res = {}
    for K in shots:
        new_acc, old_acc = [], []
        for t in range(trials):
            r = np.random.default_rng(100 + t)
            sup_emb, sup_y, qry_idx, qry_y = [], [], [], []
            for j, a in enumerate(new_apps):
                idx = r.permutation(np.where(unk & (e["app_test"] == a))[0])
                sup_emb.append(e["emb_test"][idx[:K]]); sup_y += [n_classes + j] * K
                qry_idx.append(idx[K:]); qry_y += [n_classes + j] * (len(idx) - K)
            labels = np.concatenate([e["y_train"], np.array(sup_y)])
            index = core.build_index(np.concatenate([e["emb_train"]] + sup_emb))
            qi = np.concatenate(qry_idx)
            # with few support points per new class, fewer neighbours is fairer
            kk = min(k, K) if K < k else k
            D, I = core.knn_search(index, e["emb_test"][qi], kk)
            p, _, _ = core.knn_vote(D, I, labels, n_classes + len(new_apps))
            new_acc.append((p == np.array(qry_y)).mean())
            D, I = core.knn_search(index, e["emb_test"][known_idx], k)
            p, _, _ = core.knn_vote(D, I, labels, n_classes + len(new_apps))
            old_acc.append((p == e["y_test"][known_idx]).mean())
        res[f"{K}-shot"] = {"new_services_acc": round(float(np.mean(new_acc)), 4),
                            "new_services_acc_std": round(float(np.std(new_acc)), 4),
                            "known_services_acc": round(float(np.mean(old_acc)), 4)}
    if plot_dir:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(shots, [res[f"{K}-shot"]["new_services_acc"] * 100 for K in shots], "o-", label="new services")
        ax.plot(shots, [res[f"{K}-shot"]["known_services_acc"] * 100 for K in shots], "s--", label="known services")
        ax.set_xlabel("Labeled examples per new service (K)"); ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Few-shot registration of {len(new_apps)} unseen services"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "few_shot.png"), dpi=130); plt.close(fig)
    return {"new_services": new_apps.tolist(), **res}


# ---------- E8 ----------
def e8_drift(e, pred, y, plot_dir):
    t = e["t_test"][e["y_test"] >= 0]
    if (t == 0).all():
        return {"note": "no timestamps"}
    day = t.astype("datetime64[s]").astype("datetime64[D]")
    days = np.unique(day)
    per_day = {str(dd): round(float((pred[day == dd] == y[day == dd]).mean()), 4) for dd in days}
    week = day.astype("datetime64[W]")
    per_week = {str(w): round(float((pred[week == w] == y[week == w]).mean()), 4) for w in np.unique(week)}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(len(days)), [per_day[str(dd)] * 100 for dd in days], "o-")
    ax.set_xticks(range(len(days))); ax.set_xticklabels([str(dd)[5:] for dd in days], rotation=60, fontsize=7)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy per test day (trained on week 44)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "drift.png"), dpi=130); plt.close(fig)
    return {"per_week": per_week, "per_day": per_day}


# ---------- E10 ----------
def e10_tsne(e, names, plot_dir, n=6000):
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(0)
    idx = rng.choice(len(e["y_test"]), min(n, len(e["y_test"])), replace=False)
    Z = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(e["emb_test"][idx])
    y = e["y_test"][idx]
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab20")
    for c in range(len(names)):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=3, color=cmap(c % 20), label=names[c])
    m = y == -1
    ax.scatter(Z[m, 0], Z[m, 1], s=3, color="k", alpha=.35, label="unknown services")
    ax.legend(markerscale=4, fontsize=6, ncol=2, loc="best"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE of test-flow embeddings (real CESNET-QUIC22 data)")
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "tsne.png"), dpi=130); plt.close(fig)


def evaluate(run_dir, tsne=False, few_shot=True):
    e, s = load(run_dir)
    names = e["label_names"].tolist(); n_classes = len(names); k = s["k"]
    plot_dir = os.path.join(run_dir, "plots"); os.makedirs(plot_dir, exist_ok=True)
    e["index"] = core.build_index(e["emb_train"])
    r = {"run": s["run"]}
    r["E1_closed_set"], (pred, gap, y) = e1_closed_set(e, k, n_classes, names, plot_dir)
    r["E5_selective"] = e5_selective(pred, gap, y, plot_dir)
    r["E6_open_set"] = e6_open_set(e, k, n_classes, plot_dir)
    if few_shot:
        r["E7_few_shot"] = e7_few_shot(e, k, n_classes, plot_dir=plot_dir)
    r["E8_drift"] = e8_drift(e, pred, y, plot_dir)
    if tsne:
        e10_tsne(e, names, plot_dir)
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(r, f, indent=2)
    print(json.dumps({k_: v for k_, v in r.items() if k_ != "E8_drift"}, indent=2)[:4000])
    print(f"\nSaved {run_dir}/eval.json and {plot_dir}/")


def table():
    rows = []
    for sp in sorted(glob.glob("runs/*/summary.json")):
        with open(sp) as f: s = json.load(f)
        ep = sp.replace("summary.json", "eval.json")
        ev = json.load(open(ep)) if os.path.exists(ep) else {}
        os_ = ev.get("E6_open_set", {}).get("knn_distance", {})
        fs = ev.get("E7_few_shot", {}).get("10-shot", {})
        rows.append(f"| {s['run']} | {s.get('encoder', '-')} | {s.get('loss', '-')} | {s.get('packets', '-')} "
                    f"| {s.get('params', 0):,} | {s.get('knn_accuracy', s.get('accuracy')) * 100:.2f} "
                    f"| {s.get('knn_macro_f1', s.get('macro_f1')) * 100:.2f} "
                    f"| {os_.get('auroc', float('nan')):.3f} | {fs.get('new_services_acc', float('nan')) * 100:.1f} "
                    f"| {s.get('train_time_s', 0):.0f} |")
    hdr = ("| Run | Encoder | Loss | Packets | Params | Acc % | Macro-F1 % | Unknown AUROC | 10-shot new % | Train s |\n"
           "|---|---|---|---|---|---|---|---|---|---|")
    text = hdr + "\n" + "\n".join(rows)
    note = ("\n\nNotes: `gbdt` (no seed suffix) uses whole-flow statistics and is NOT a fair baseline; "
            "fair GBDT runs are `gbdt_ppi*`. `tcn_supcon_p30` is the original run, reproduced exactly by "
            "`tcn_supcon_p30_s0`. Unknown AUROC / few-shot columns are filled only for runs evaluated with evaluate.py.")
    os.makedirs("results", exist_ok=True)
    with open("results/all_runs.md", "w", encoding="utf-8") as f: f.write(text + note + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--tsne", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--skip-fewshot", action="store_true", help="Skip E7 (slow: rebuilds the index per trial)")
    a = ap.parse_args()
    if a.table:
        table()
    else:
        evaluate(a.run_dir, a.tsne, few_shot=not a.skip_fewshot)
