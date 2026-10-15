"""
E4: accuracy vs number of packets, recomputed from saved predictions.
Selected model (Transformer + CE, seed 0) vs fair GBDT on the same packet prefix.

    python e4_packets.py
Writes results/e4_packets.json, results/e4_packets.md, results/e4_packets.png
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

PACKETS = [3, 5, 10, 15, 20, 30]
MODELS = {"Transformer + CE (k-NN)": ("transformer_ce_p{}_s0", "y_pred"),
          "GBDT, fair (packet sequence only)": ("gbdt_ppi_p{}_s0", "y_pred")}


def main():
    res = {}
    for m, (pat, key) in MODELS.items():
        res[m] = {}
        for p in PACKETS:
            path = os.path.join("runs", pat.format(p), "predictions.npz")
            if not os.path.exists(path):
                continue
            b = np.load(path, allow_pickle=True)
            y, yp = b["y_true"], b[key]
            res[m][p] = {"accuracy": float((y == yp).mean()), "macro_f1": float(f1_score(y, yp, average="macro"))}

    # share of test flows that actually have at least N packets (context for small N)
    b = np.load("data/cesnet_top20.npz", allow_pickle=True)
    X = b["X_test"][b["y_test"] >= 0]
    n_pkts = (X[..., 0] > 0).sum(1)
    have = {p: float((n_pkts >= p).mean()) for p in PACKETS}

    os.makedirs("results", exist_ok=True)
    with open("results/e4_packets.json", "w") as f:
        json.dump({"results": res, "share_of_test_flows_with_at_least_N_packets": have}, f, indent=2)
    L = ["# E4: accuracy vs number of packets", "",
         "Known test flows, seed 0. Truncation to N packets is applied to train, validation and test.", "",
         "| Packets | Flows with ≥ N pkts | " + " | ".join(f"{m} acc / macro-F1" for m in res) + " |",
         "|---|---|" + "---|" * len(res)]
    for p in PACKETS:
        cells = [f"{res[m][p]['accuracy'] * 100:.2f} / {res[m][p]['macro_f1'] * 100:.2f}" if p in res[m] else "—"
                 for m in res]
        L.append(f"| {p} | {have[p] * 100:.0f}% | " + " | ".join(cells) + " |")
    with open("results/e4_packets.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    fig, ax = plt.subplots(figsize=(6, 4))
    for (m, r), st in zip(res.items(), ["o-", "s--"]):
        ps = sorted(r)
        ax.plot(ps, [r[p]["accuracy"] * 100 for p in ps], st, label=f"{m} accuracy")
        ax.plot(ps, [r[p]["macro_f1"] * 100 for p in ps], st, alpha=.45, label=f"{m} macro-F1")
    ax.set_xlabel("Packets observed (first N of the flow)"); ax.set_ylabel("%")
    ax.set_title("How early can the service be identified?"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/e4_packets.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
