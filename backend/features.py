"""Flow-level feature extraction (no payload inspection)."""
from collections import defaultdict
import numpy as np

MAX_PACKETS = 30   # first N packets per flow (CESNET PPI length)
FEATURE_DIM = 3    # [size, inter-arrival-time, direction]

# ---------- Real data: extract flows from a pcap ----------
# Conventions match CESNET-QUIC22 PPI sequences (what the model is trained on):
#   size      = transport-layer payload bytes (UDP/TCP payload, no headers)
#   iat       = milliseconds since the previous packet of the same flow (0 for the first)
#   direction = +1 client->server, -1 server->client; client = sender of the flow's first packet
# Flows are bidirectional 5-tuples split by inactive / active timeouts (ipfixprobe-style).
INACTIVE_TIMEOUT_S = 65.0
ACTIVE_TIMEOUT_S = 300.0

def _packet_fields(pkt):
    """(src, dst, sport, dport, proto, payload_len) or None for non TCP/UDP packets."""
    from scapy.all import IP, IPv6, TCP, UDP
    if IP in pkt:
        ip = pkt[IP]; src, dst, proto = ip.src, ip.dst, ip.proto
    elif IPv6 in pkt:
        ip = pkt[IPv6]; src, dst, proto = ip.src, ip.dst, ip.nh
    else:
        return None
    if UDP in pkt:
        l4 = pkt[UDP]
        payload = (l4.len - 8) if l4.len else len(l4.payload)
    elif TCP in pkt:
        l4 = pkt[TCP]; payload = len(l4.payload)
        # strip Ethernet trailer padding that scapy may attach to short TCP segments
        if IP in pkt and pkt[IP].len:
            payload = min(payload, pkt[IP].len - 4 * pkt[IP].ihl - 4 * l4.dataofs)
    else:
        return None
    return src, dst, l4.sport, l4.dport, proto, max(int(payload), 0)


def extract_flows_from_pcap(path, max_packets=MAX_PACKETS, skip_empty=True,
                            inactive_timeout=INACTIVE_TIMEOUT_S, active_timeout=ACTIVE_TIMEOUT_S):
    """Returns {(client, server, proto, flow_index): (max_packets, 3) array}.
    Payload bytes are never read -- only their length."""
    from scapy.all import PcapReader
    open_flows, done, counter = {}, [], defaultdict(int)
    for pkt in PcapReader(path):
        f = _packet_fields(pkt)
        if f is None:
            continue
        src, dst, sport, dport, proto, size = f
        if skip_empty and size == 0:          # pure TCP ACKs / control packets carry no payload
            continue
        t = float(pkt.time)
        a, b = (src, sport), (dst, dport)
        key = (min(a, b), max(a, b), proto)
        fl = open_flows.get(key)
        if fl is not None and (t - fl["last"] > inactive_timeout or t - fl["start"] > active_timeout):
            done.append(fl); fl = None
        if fl is None:
            counter[key] += 1
            fl = open_flows[key] = {"client": a, "server": b, "proto": proto, "start": t,
                                    "last": t, "idx": counter[key], "pkts": []}
        fl["last"] = t
        fl["pkts"].append((t, size, 1 if a == fl["client"] else -1))
    done.extend(open_flows.values())
    return {(f["client"], f["server"], f["proto"], f["idx"]): flow_to_sequence(f["pkts"], max_packets)
            for f in done}

def flow_to_sequence(packets, max_packets=MAX_PACKETS):
    """packets: list of (timestamp_seconds, payload_size, direction)."""
    packets = sorted(packets, key=lambda p: p[0])[:max_packets]
    out, prev_t = np.zeros((max_packets, FEATURE_DIM), np.float32), packets[0][0]
    for i, (t, size, direction) in enumerate(packets):
        out[i] = [size, (t - prev_t) * 1000.0, direction]
        prev_t = t
    return out

# ---------- Synthetic data: run the whole pipeline with zero downloads ----------
def make_synthetic_dataset(n_per_class=600, n_classes=5, seed=0):
    """5 separable 'app' classes with distinct size/timing signatures."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        base_size, base_iat = 100 + c * 250, 0.001 * (c + 1)
        for _ in range(n_per_class):
            seq = np.zeros((MAX_PACKETS, FEATURE_DIM), np.float32)
            for i in range(MAX_PACKETS):
                seq[i] = [rng.normal(base_size, 40),
                          abs(rng.normal(base_iat, base_iat * 0.5)),
                          rng.choice([1, -1])]
            X.append(seq); y.append(c)
    return np.stack(X), np.array(y)

# ---------- Normalization (size & IAT are heavy-tailed → log first) ----------
def fit_stats(X):
    flat = X.reshape(-1, X.shape[-1]).copy()
    # Real captures can have small negative inter-packet-time values (packet
    # reordering in transit) which log1p can't handle -- clip the same way
    # transform() does, or fit stats silently go to -inf/NaN on real data.
    flat[:, 0], flat[:, 1] = np.log1p(np.clip(flat[:, 0], 0, None)), \
                              np.log1p(np.clip(flat[:, 1], 0, None))
    return flat.mean(0), flat.std(0) + 1e-6

def transform(x, mean, std):
    x = x.copy()
    x[..., 0], x[..., 1] = np.log1p(np.clip(x[..., 0], 0, None)), \
                           np.log1p(np.clip(x[..., 1], 0, None))
    return (x - mean) / std