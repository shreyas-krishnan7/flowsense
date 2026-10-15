"""Shared inference over exported artifacts (used by serve.py and classify_pcap.py).

Threading: each request is small, so every library is limited to one thread per
call (FLOWSENSE_THREADS, default 1) and parallelism comes from concurrent requests.
Without this, every concurrent request spawned a full OpenMP/OpenBLAS thread pool,
which oversubscribed the CPU and exhausted memory under load."""
import os
_T = os.environ.get("FLOWSENSE_THREADS", "1")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _T)
import numpy as np
import onnxruntime as ort
import faiss

from features import transform, MAX_PACKETS, FEATURE_DIM
import core

DEFAULT_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def pad(seq):
    out = np.zeros((MAX_PACKETS, FEATURE_DIM), np.float32)
    seq = np.asarray(seq, np.float32).reshape(-1, FEATURE_DIM)[:MAX_PACKETS]
    out[:len(seq)] = seq
    return out


class FlowClassifier:
    def __init__(self, art_dir=DEFAULT_ART, threads=int(_T)):
        faiss.omp_set_num_threads(threads)
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.sess = ort.InferenceSession(os.path.join(art_dir, "encoder.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.index = faiss.read_index(os.path.join(art_dir, "knn.index"))
        b = np.load(os.path.join(art_dir, "bundle.npz"), allow_pickle=True)
        self.labels, self.mean, self.std, self.k = b["labels"], b["mean"], b["std"], int(b["k"])
        self.novelty_thr, self.gap_thr = float(b["novelty_threshold"]), float(b["gap_threshold"])
        self.names = b["label_names"].tolist()
        if "ef_search" in b.files and int(b["ef_search"]) > 0:      # HNSW index
            self.index.hnsw.efSearch = int(b["ef_search"])

    def classify(self, flows):
        """flows: (N, <=30, 3) packet sequences [size_bytes, iat_ms, direction]."""
        X = np.stack([pad(f) for f in flows])
        emb = self.sess.run(None, {"flow": transform(X, self.mean, self.std).astype(np.float32)})[0]
        D, I = self.index.search(emb, self.k)
        pred, conf, gap = core.knn_vote(D, I, self.labels, len(self.names))
        novelty = core.knn_novelty(D)
        out = []
        for p, c, g, n in zip(pred, conf, gap, novelty):
            decision = ("novel_traffic" if n > self.novelty_thr else
                        "needs_review" if g < self.gap_thr else "classified")
            out.append({"label": int(p), "label_name": self.names[int(p)],
                        "confidence": round(float(c), 4), "gap": round(float(g), 4),
                        "novelty_score": round(float(n), 4), "decision": decision})
        return out
