"""
Real-traffic loaders for FlowSense.

CESNET-QUIC22 (labeled -> used for training/classification)
-------------------------------------------------------------
Distributed as CSV shards (one per capture day/week) by CESNET.
Docs: https://github.com/CESNET/cesnet-datazoo
      https://zenodo.org/records/10728760
Each row has a service label column ("APP") and a PPI column containing the
first ~30 packets of the flow as a stringified nested list:
    PPI = "[[iat_1, iat_2, ...], [dir_1, dir_2, ...], [size_1, size_2, ...]]"
(inter-arrival time in ms, direction in {-1, 1}, size in bytes)

You don't need the full 153M-flow release to get something real and
resume-worthy — CESNET publishes weekly shards; even one day's CSV
(a few hundred MB) is plenty for a solid multi-class training set.
Download a shard, point --csv at it, and this module does the rest.

MAWI (unlabeled -> used ONLY to stress-test the novelty detector)
-------------------------------------------------------------
MAWI (https://mawi.wide.ad.jp/mawi/) is raw backbone pcap traffic with no
application labels, so it can't teach the classifier new classes. What it's
good for: real, genuinely out-of-distribution traffic to check that the
DBSCAN novelty detector correctly flags "never seen this before" instead of
silently misclassifying it. This is the honest, defensible way to use MAWI
in this project — don't claim it as labeled training data anywhere.
"""
import ast
import numpy as np
import pandas as pd

from features import MAX_PACKETS, FEATURE_DIM, extract_flows_from_pcap


# ---------- CESNET-QUIC22 (labeled) ----------
def _parse_ppi(raw):
    """CESNET's PPI order is [[iat...], [dir...], [size...]]; features.py's
    convention is [size, iat, direction] per timestep, so reorder here."""
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    iat, direction, size = raw
    T = min(len(size), MAX_PACKETS)
    out = np.zeros((MAX_PACKETS, FEATURE_DIM), np.float32)
    for i in range(T):
        out[i] = [size[i], iat[i], direction[i]]
    return out


def load_cesnet_quic22(csv_paths, label_col="APP", ppi_col="PPI",
                        top_k=10, max_rows_per_class=None, seed=0):
    """Load one or more CESNET-QUIC22 CSV shards, keep the top_k most
    frequent service labels (102 raw labels is overkill for a resume
    project and badly class-imbalanced), and return (X, y, label_names).
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    if label_col not in df.columns or ppi_col not in df.columns:
        raise ValueError(
            f"Expected columns '{label_col}' and '{ppi_col}' in the CSV. "
            f"Found: {list(df.columns)}. If CESNET renamed fields in your "
            f"shard, pass label_col=/ppi_col= explicitly."
        )

    top_labels = df[label_col].value_counts().nlargest(top_k).index.tolist()
    df = df[df[label_col].isin(top_labels)].reset_index(drop=True)

    if max_rows_per_class:
        rng = np.random.default_rng(seed)
        df = (df.groupby(label_col, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_rows_per_class),
                                           random_state=seed)))

    label_names = sorted(df[label_col].unique().tolist())
    label_to_id = {l: i for i, l in enumerate(label_names)}

    X = np.stack([_parse_ppi(p) for p in df[ppi_col].tolist()])
    y = df[label_col].map(label_to_id).to_numpy()
    return X, y, label_names


# ---------- CESNET-QUIC22 via the official cesnet-datazoo package ----------
def load_cesnet_quic22_datazoo(data_root, size="M", top_k=10,
                                train_period_name=None, test_period_name=None):
    """Load CESNET-QUIC22 through the cesnet-datazoo package (run
    download_data.py once first). This returns train/test splits already
    separated by time period, which is the realistic evaluation setup the
    dataset was designed for -- no manual train/test split needed here.

    Returns: (Xtr, ytr, Xte, yte, label_names)
    """
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection

    dataset = CESNET_QUIC22(data_root, size=size)
    config = DatasetConfig(
        dataset=dataset,
        train_period_name=train_period_name or dataset.default_train_period_name,
        test_period_name=test_period_name or dataset.default_test_period_name,
        apps_selection=AppSelection.TOPX_KNOWN,
        apps_selection_topx=top_k,
        # Default train_workers=4 spawns multiple threads each doing BLAS ops
        # concurrently, which can exhaust memory on a laptop -- force
        # single-worker, smaller-batch loading instead.
        train_workers=0,
        test_workers=0,
        batch_size=64,
        test_batch_size=256,
    )
    dataset.set_dataset_config_and_initialize(config)

    train_df = dataset.get_train_df(flatten_ppi=False)
    test_df = dataset.get_test_df(flatten_ppi=False)

    # cesnet-datazoo already integer-encodes the APP column internally (via
    # sklearn LabelEncoder fit on sorted known-app names), so df["APP"] is
    # already the class id -- get the real string names from the dataset
    # object itself, not by re-deriving them from already-encoded integers.
    label_names = dataset.get_known_apps()

    def _to_xy(df):
        # PPI here is (channels=[IPT, DIR, SIZE], seq_len<=30) per row
        X = np.zeros((len(df), MAX_PACKETS, FEATURE_DIM), np.float32)
        for i, ppi in enumerate(df["PPI"].tolist()):
            ppi = np.asarray(ppi)
            iat, direction, size = ppi[0], ppi[1], ppi[2]
            T = min(len(size), MAX_PACKETS)
            X[i, :T, 0] = size[:T]
            X[i, :T, 1] = iat[:T]
            X[i, :T, 2] = direction[:T]
        y = df["APP"].to_numpy()   # already integer class ids
        return X, y

    Xtr, ytr = _to_xy(train_df)
    Xte, yte = _to_xy(test_df)
    return Xtr, ytr, Xte, yte, label_names


# ---------- MAWI (unlabeled, novelty-eval only) ----------
def load_mawi_unlabeled(pcap_paths, max_flows=5000):
    """Extract flows from real MAWI pcap(s) with NO labels attached. Use
    only to evaluate whether the novelty detector correctly flags this as
    unknown traffic -- never mix into the labeled train/test split."""
    if isinstance(pcap_paths, str):
        pcap_paths = [pcap_paths]
    seqs = []
    for p in pcap_paths:
        flows = extract_flows_from_pcap(p)
        seqs.extend(flows.values())
        if len(seqs) >= max_flows:
            break
    if not seqs:
        raise ValueError("No flows extracted from the given MAWI pcap(s).")
    return np.stack(seqs[:max_flows])
