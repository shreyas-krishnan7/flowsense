"""
Load CESNET-QUIC22 once through cesnet-datazoo and cache it as plain numpy
arrays, so every later experiment loads in seconds instead of re-reading the
13 GB database.

Split (time-based, the realistic setup):
    train/val : week W-2022-44
    test      : weeks W-2022-45..47 (each flow keeps its timestamp -> drift eval)

Open-set setup: the top-X services are *known*; every other service is
*unknown* and appears only in the test set (label -1). Those unknown flows are
used for novelty detection (E6) and few-shot registration (E7).

Usage:
    python prepare_data.py --data-root C:/datasets/CESNET-QUIC22 --size M
Output:
    data/cesnet_top{X}.npz
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, time
import numpy as np
import pandas as pd

from features import MAX_PACKETS, FEATURE_DIM

FLOWSTAT_COLS = ["BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "DURATION",
                 "PPI_LEN", "PPI_ROUNDTRIPS", "PPI_DURATION",
                 "FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_OTHER"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="C:/datasets/CESNET-QUIC22")
    ap.add_argument("--size", choices=["XS", "S", "M", "L"], default="M")
    ap.add_argument("--top-x", type=int, default=20, help="Number of known services")
    ap.add_argument("--train-size", type=int, default=300_000)
    ap.add_argument("--val-size", type=int, default=30_000)
    ap.add_argument("--test-known-size", type=int, default=150_000)
    ap.add_argument("--test-unknown-size", type=int, default=50_000)
    ap.add_argument("--train-period", default="W-2022-44")
    ap.add_argument("--test-period", default="W45-47")
    ap.add_argument("--out-dir", default="data")
    return ap.parse_args()


def df_to_arrays(df, app_to_id):
    """CESNET PPI is (channels=[IPT, DIR, SIZE], 30); our convention is
    (30, [size, iat, direction])."""
    ppi = np.stack([np.asarray(p, np.float32) for p in df["PPI"].tolist()])  # (N, 3, T)
    T = min(ppi.shape[2], MAX_PACKETS)
    X = np.zeros((len(df), MAX_PACKETS, FEATURE_DIM), np.float32)
    X[:, :T, 0] = ppi[:, 2, :T]   # size
    X[:, :T, 1] = ppi[:, 0, :T]   # inter-packet time (ms)
    X[:, :T, 2] = ppi[:, 1, :T]   # direction
    y = df["APP"].map(lambda a: app_to_id.get(a, -1)).to_numpy(np.int64)
    app = df["APP"].astype(str).to_numpy()
    stats = df[FLOWSTAT_COLS].to_numpy(np.float32)
    t = pd.to_datetime(df["TIME_FIRST"]).to_numpy("datetime64[s]").astype(np.int64)
    return X, y, app, stats, t


def main():
    args = parse_args()
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection

    t0 = time.time()
    dataset = CESNET_QUIC22(args.data_root, size=args.size, silent=False)
    config = DatasetConfig(
        dataset=dataset,
        train_period_name=args.train_period,
        test_period_name=args.test_period,
        apps_selection=AppSelection.TOPX_KNOWN,
        apps_selection_topx=args.top_x,
        train_size=args.train_size,
        val_known_size=args.val_size,
        test_known_size=args.test_known_size,
        test_unknown_size=args.test_unknown_size,
        return_other_fields=True,
        disable_label_encoding=True,     # keep app names as strings; we encode ourselves
        train_workers=0, val_workers=0, test_workers=0,
        batch_size=4096, test_batch_size=4096,
    )
    dataset.set_dataset_config_and_initialize(config)
    known = sorted(dataset.get_known_apps())
    unknown = sorted(dataset.get_unknown_apps())
    app_to_id = {a: i for i, a in enumerate(known)}
    print(f"Known services ({len(known)}): {known}")
    print(f"Unknown services: {len(unknown)}")

    out = {}
    for split, getter in [("train", dataset.get_train_df),
                          ("val", dataset.get_val_df),
                          ("test", dataset.get_test_df)]:
        ts = time.time()
        df = getter(flatten_ppi=False)
        X, y, app, stats, t = df_to_arrays(df, app_to_id)
        del df
        out.update({f"X_{split}": X, f"y_{split}": y, f"app_{split}": app,
                    f"stats_{split}": stats, f"t_{split}": t})
        print(f"{split:5s}: {len(y):,} flows ({(y == -1).sum():,} unknown) "
              f"in {time.time() - ts:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"cesnet_top{args.top_x}.npz")
    np.savez(path, label_names=np.array(known, dtype=object),
             unknown_names=np.array(unknown, dtype=object),
             stat_names=np.array(FLOWSTAT_COLS, dtype=object), **out)
    print(f"Saved {path} ({os.path.getsize(path) / 1e6:.0f} MB) "
          f"in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
