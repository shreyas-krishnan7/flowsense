"""
HTTP benchmark of serve.py with REAL CESNET test flows (not a constant payload).

    uvicorn serve:app --port 8000          # in another terminal
    python loadtest.py --requests 2000 --concurrency 1 16

Reports end-to-end client-side latency (p50/p95/p99) and throughput per
concurrency level; writes results/serving_benchmark.json.
"""
import argparse, json, os, platform, statistics, time, concurrent.futures
import numpy as np
import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/classify")
    ap.add_argument("--data", default="data/cesnet_top20.npz")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--warmup", type=int, default=100)
    a = ap.parse_args()

    X = np.load(a.data, allow_pickle=True)["X_test"]
    rng = np.random.default_rng(0)
    flows = X[rng.choice(len(X), a.requests, replace=False)]
    payloads = []
    for f in flows:
        n = int((f[:, 0] > 0).sum()) or 1            # send only the real packets, like a client would
        payloads.append({"packets": f[:n].tolist()})

    client = httpx.Client(timeout=30, limits=httpx.Limits(max_connections=max(a.concurrency)))
    for p in payloads[:a.warmup]:
        client.post(a.url, json=p).raise_for_status()

    def one(p):
        t = time.perf_counter()
        r = client.post(a.url, json=p)
        r.raise_for_status()
        return time.perf_counter() - t, r.json()["decision"]

    results = {}
    for c in a.concurrency:
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            start = time.perf_counter()
            out = list(ex.map(one, payloads))
            wall = time.perf_counter() - start
        lat = sorted(o[0] * 1000 for o in out)
        decisions = {d: sum(1 for o in out if o[1] == d) / len(out) for d in {o[1] for o in out}}
        results[f"concurrency_{c}"] = {
            "requests": len(lat), "throughput_rps": round(len(lat) / wall, 1),
            "p50_ms": round(statistics.median(lat), 2),
            "p95_ms": round(lat[int(0.95 * len(lat)) - 1], 2),
            "p99_ms": round(lat[int(0.99 * len(lat)) - 1], 2),
            "decision_mix": {k: round(v, 4) for k, v in decisions.items()}}
        print(f"concurrency {c:3d}: {results[f'concurrency_{c}']}")

    manifest = {}
    if os.path.exists("artifacts/manifest.json"):
        manifest = json.load(open("artifacts/manifest.json"))
    os.makedirs("results", exist_ok=True)
    with open("results/serving_benchmark.json", "w") as f:
        json.dump({"model": manifest.get("run"), "server": "uvicorn serve:app (1 worker), same machine",
                   "machine": f"{platform.processor()} / {os.cpu_count()} logical CPUs",
                   "payload": "real CESNET test flows (only real packets sent)", **results}, f, indent=2)


if __name__ == "__main__":
    main()
