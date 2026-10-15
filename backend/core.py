"""Shared decision layer on top of embeddings: k-NN classification with
confidence-gap routing, and novelty scores. Used by train.py, evaluate.py
and serve.py so all three make identical decisions."""
import numpy as np
import faiss


def build_index(emb):
    index = faiss.IndexFlatIP(emb.shape[1])     # inner product == cosine on L2-normalized vectors
    index.add(np.ascontiguousarray(emb, np.float32))
    return index


def knn_search(index, q, k, batch=8192):
    D, I = [], []
    q = np.ascontiguousarray(q, np.float32)
    for s in range(0, len(q), batch):
        d, i = index.search(q[s:s + batch], k)
        D.append(d); I.append(i)
    return np.concatenate(D), np.concatenate(I)


def knn_vote(D, I, labels, n_classes):
    """Similarity-weighted vote over the k neighbours (vectorized).
    Returns predicted label, confidence (top share) and gap (top1 - top2 share)."""
    votes = np.zeros((len(D), n_classes), np.float32)
    rows = np.repeat(np.arange(len(D)), D.shape[1])
    np.add.at(votes, (rows, labels[I].ravel()), np.clip(D, 0, None).ravel())
    total = votes.sum(1) + 1e-9
    top2 = np.sort(votes, axis=1)[:, -2:]
    pred = votes.argmax(1)
    conf = top2[:, 1] / total
    gap = (top2[:, 1] - top2[:, 0]) / total
    return pred, conf, gap


# ---------- novelty scores (higher = more novel) ----------
def knn_novelty(D):
    """1 - mean cosine similarity to the k nearest training flows."""
    return 1 - D.mean(1)


def class_prototypes(emb, labels, n_classes):
    P = np.stack([emb[labels == c].mean(0) for c in range(n_classes)])
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)


def proto_novelty(q, protos):
    """1 - cosine similarity to the closest class prototype (class mean)."""
    return 1 - (q @ protos.T).max(1)


def dbscan_core(emb, eps=0.15, min_samples=5, max_points=20000, seed=0):
    """Original FlowSense detector: DBSCAN core points of the training
    embeddings. DBSCAN is O(n^2), so it runs on a random subsample."""
    from sklearn.cluster import DBSCAN
    rng = np.random.default_rng(seed)
    sub = emb[rng.choice(len(emb), min(len(emb), max_points), replace=False)]
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(sub)
    core = sub[db.core_sample_indices_]
    return core if len(core) else sub


def dbscan_novelty(q, core, batch=4096):
    out = []
    for s in range(0, len(q), batch):
        out.append(1 - (q[s:s + batch] @ core.T).max(1))
    return np.concatenate(out)


def threshold_at_tpr(known_scores, tpr=0.95):
    """Novelty threshold that keeps `tpr` of known (in-distribution) flows."""
    return float(np.quantile(known_scores, tpr))
