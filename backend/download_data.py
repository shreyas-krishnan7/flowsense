"""
Run this ONCE to download the CESNET-QUIC22 dataset via the official
cesnet-datazoo package (much smaller than the raw 21 GB Zenodo zip --
this pulls a pre-packaged size bucket instead).

Usage:
    pip install cesnet-datazoo
    python download_data.py --data-root /datasets/CESNET-QUIC22 --size M

Sizes: XS < S < M < L. M is a good default -- enough volume/class balance
for real training without eating your whole disk.
"""
import argparse
from cesnet_datazoo.datasets import CESNET_QUIC22


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/datasets/CESNET-QUIC22",
                     help="Where the dataset gets downloaded/cached on disk")
    ap.add_argument("--size", choices=["XS", "S", "M", "L"], default="M")
    args = ap.parse_args()

    # Instantiating the class triggers the download (cached after first run --
    # safe to re-run, it won't re-download what's already there).
    dataset = CESNET_QUIC22(args.data_root, size=args.size)
    print(f"Dataset ready at {args.data_root} (size={args.size})")
    print(f"Available time periods: {list(dataset.time_periods.keys())}")
    print(f"Default train period: {dataset.default_train_period_name}")
    print(f"Default test period:  {dataset.default_test_period_name}")


if __name__ == "__main__":
    main()
