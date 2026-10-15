"""
Re-load CESNET-QUIC22 with the exact configuration used by prepare_data.py and
save per-flow metadata (flow ID, source IP, destination IP) that the cache does
not contain. Used only for evaluation checks (source-IP separation), never as
model input. The existing data/cesnet_top20.npz is not modified.

Alignment with the cache is verified (labels and packet sequences must be
identical row by row); the script aborts otherwise.

    python extract_meta.py
Output: data/cesnet_top20_meta.npz
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time
import numpy as np

sys.argv = [sys.argv[0]]                    # use prepare_data.py defaults
import prepare_data as P


def main():
    args = P.parse_args()
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    t0 = time.time()
    dataset = CESNET_QUIC22(args.data_root, size=args.size, silent=True)
    config = DatasetConfig(
        dataset=dataset, train_period_name=args.train_period, test_period_name=args.test_period,
        apps_selection=AppSelection.TOPX_KNOWN, apps_selection_topx=args.top_x,
        train_size=args.train_size, val_known_size=args.val_size,
        test_known_size=args.test_known_size, test_unknown_size=args.test_unknown_size,
        return_other_fields=True, disable_label_encoding=True,
        train_workers=0, val_workers=0, test_workers=0, batch_size=4096, test_batch_size=4096)
    dataset.set_dataset_config_and_initialize(config)
    known = sorted(dataset.get_known_apps())
    app_to_id = {a: i for i, a in enumerate(known)}

    cache = np.load(os.path.join(args.out_dir, f"cesnet_top{args.top_x}.npz"), allow_pickle=True)
    out = {}
    for split, getter in [("train", dataset.get_train_df), ("val", dataset.get_val_df),
                          ("test", dataset.get_test_df)]:
        df = getter(flatten_ppi=False)
        X, y, _, _, _ = P.df_to_arrays(df, app_to_id)
        if not (np.array_equal(y, cache[f"y_{split}"]) and np.array_equal(X, cache[f"X_{split}"])):
            raise SystemExit(f"{split}: rows do not match the cache -- aborting (no file written)")
        for col in ["ID", "SRC_IP", "DST_IP"]:
            out[f"{col.lower()}_{split}"] = df[col].astype(str).to_numpy()
        print(f"{split}: {len(df):,} rows, aligned with cache")
    path = os.path.join(args.out_dir, f"cesnet_top{args.top_x}_meta.npz")
    np.savez(path, **out)
    print(f"Saved {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
