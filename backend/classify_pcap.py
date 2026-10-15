"""
Offline PCAP inference: extract flows from a capture file and classify each one.
(Offline only -- live capture / streaming flow assembly is not implemented.)

    python classify_pcap.py capture.pcap [--min-packets 3] [--out flows.csv]
"""
import argparse
import pandas as pd
from features import extract_flows_from_pcap
from inference import FlowClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("--min-packets", type=int, default=3, help="skip flows with fewer payload packets")
    ap.add_argument("--udp-443-only", action="store_true", help="keep only QUIC-like flows (UDP/443)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    flows = extract_flows_from_pcap(a.pcap)
    keys, seqs = [], []
    for k, s in flows.items():
        (cip, cport), (sip, sport), proto, idx = k
        if a.udp_443_only and not (proto == 17 and sport == 443):
            continue
        if (s[:, 0] > 0).sum() < a.min_packets:
            continue
        keys.append({"client": f"{cip}:{cport}", "server": f"{sip}:{sport}", "proto": proto,
                     "flow_idx": idx, "packets": int((s[:, 0] > 0).sum())})
        seqs.append(s)
    if not seqs:
        raise SystemExit("no flows with enough packets")
    df = pd.DataFrame([{**k, **r} for k, r in zip(keys, FlowClassifier().classify(seqs))])
    print(df.to_string(index=False))
    print("\n" + df["decision"].value_counts().to_string())
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
