import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import core


def test_knn_vote_weighted():
    labels = np.array([0, 0, 1, 2])
    D = np.array([[0.9, 0.8, 0.5]], np.float32)          # neighbours 0,1 (class 0) and 2 (class 1)
    I = np.array([[0, 1, 2]])
    pred, conf, gap = core.knn_vote(D, I, labels, 3)
    assert pred[0] == 0
    np.testing.assert_allclose(conf[0], 1.7 / 2.2, rtol=1e-5)
    np.testing.assert_allclose(gap[0], (1.7 - 0.5) / 2.2, rtol=1e-5)


def test_knn_vote_ignores_negative_similarity():
    labels = np.array([0, 1])
    pred, conf, _ = core.knn_vote(np.array([[0.5, -0.9]], np.float32), np.array([[0, 1]]), labels, 2)
    assert pred[0] == 0 and conf[0] > 0.99


def test_index_search_matches_bruteforce():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 16)).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[:7]
    D, I = core.knn_search(core.build_index(emb), q, 5)
    brute = np.argsort(-(q @ emb.T), axis=1)[:, :5]
    assert (I == brute).all()
    np.testing.assert_allclose(D[:, 0], 1.0, atol=1e-5)    # each query finds itself


def test_threshold_at_tpr():
    s = np.arange(1000) / 1000
    thr = core.threshold_at_tpr(s, 0.95)
    assert abs((s <= thr).mean() - 0.95) < 0.002


def test_prototypes_unit_norm():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(100, 8)); y = rng.integers(0, 4, 100)
    P = core.class_prototypes(emb, y, 4)
    np.testing.assert_allclose(np.linalg.norm(P, axis=1), 1, atol=1e-6)
