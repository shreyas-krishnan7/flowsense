"""Round trip on REAL CESNET flows: packet sequences -> pcap -> extract_flows_from_pcap
must reproduce the original sequences, and (if a model is exported) the same predictions."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pytest
from scapy.all import Ether, IP, UDP, Raw, wrpcap

from features import extract_flows_from_pcap

DATA = os.path.join(ROOT, "data", "cesnet_top20.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason="cached CESNET data not present")
N_FLOWS = 60


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    b = np.load(DATA, allow_pickle=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(b["X_test"]), N_FLOWS, replace=False)
    X = b["X_test"][idx]
    pkts, t0 = [], 1_000_000.0
    for i, f in enumerate(X):
        client, server, cport = f"10.1.{i // 250}.{i % 250 + 1}", "203.0.113.5", 40000 + i
        t = t0 + i * 1000.0                      # flows far apart -> no timeout interaction
        for size, iat, d in f[f[:, 0] > 0]:
            t += float(iat) / 1000.0
            src, dst, sp, dp = (client, server, cport, 443) if d > 0 else (server, client, 443, cport)
            p = Ether() / IP(src=src, dst=dst) / UDP(sport=sp, dport=dp) / Raw(b"\x00" * int(size))
            p.time = t
            pkts.append(p)
    path = str(tmp_path_factory.mktemp("rt") / "cesnet_roundtrip.pcap")
    wrpcap(path, pkts)
    flows = extract_flows_from_pcap(path)
    got = np.stack([flows[((f"10.1.{i // 250}.{i % 250 + 1}", 40000 + i), ("203.0.113.5", 443), 17, 1)]
                    for i in range(N_FLOWS)])
    return X, got


def test_sequences_match(roundtrip):
    X, got = roundtrip
    np.testing.assert_array_equal(got[..., 0], X[..., 0])               # payload sizes exact
    np.testing.assert_array_equal(got[..., 2], X[..., 2])               # directions exact
    np.testing.assert_allclose(got[..., 1], X[..., 1], atol=2e-3)        # ms gaps (pcap has µs resolution)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "artifacts", "manifest.json")),
                    reason="no exported model")
def test_predictions_match(roundtrip):
    from inference import FlowClassifier
    X, got = roundtrip
    clf = FlowClassifier()
    a, b = clf.classify(list(X)), clf.classify(list(got))
    assert [r["label"] for r in a] == [r["label"] for r in b]
    assert [r["decision"] for r in a] == [r["decision"] for r in b]
