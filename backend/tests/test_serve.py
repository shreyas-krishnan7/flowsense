"""End-to-end checks of the exported model; skipped when artifacts/ has not been exported."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pytest

ART = os.path.join(ROOT, "artifacts")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ART, "manifest.json")),
                                reason="no exported model (run export.py)")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import serve
    return TestClient(serve.app)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["classes"] == 20


def test_classify_real_flow(client):
    X = np.load(os.path.join(ROOT, "data", "cesnet_top20.npz"), allow_pickle=True)["X_test"][:20]
    for f in X:
        n = int((f[:, 0] > 0).sum())
        r = client.post("/classify", json={"packets": f[:n].tolist()}).json()
        assert r["decision"] in {"classified", "needs_review", "novel_traffic"}
        assert 0 <= r["confidence"] <= 1 and 0 <= r["gap"] <= 1


def test_short_and_long_inputs(client):
    assert client.post("/classify", json={"packets": [[1200, 0, 1]]}).status_code == 200
    assert client.post("/classify", json={"packets": [[100, 1, -1]] * 50}).status_code == 200   # truncated to 30


def test_manifest_parity():
    m = json.load(open(os.path.join(ART, "manifest.json")))
    assert m["parity"]["onnx_vs_pytorch_max_abs"] < 1e-4
    # exact search must agree perfectly; HNSW may flip a near-tie on ~1e-7 embedding differences
    min_agree = 1.0 if m.get("index", "exact") == "exact" else 0.999
    assert m["parity"]["top1_label_agreement"] >= min_agree
