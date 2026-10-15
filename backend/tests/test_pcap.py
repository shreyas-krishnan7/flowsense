"""PCAP extraction must reproduce CESNET-QUIC22 PPI conventions exactly:
size = transport payload bytes, iat = milliseconds, direction +1 = client->server."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from scapy.all import Ether, IP, IPv6, UDP, TCP, Raw, wrpcap

from features import extract_flows_from_pcap, MAX_PACKETS

C, S = "10.0.0.1", "142.250.1.1"


def udp(src, dst, sport, dport, n, t):
    p = Ether() / IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / Raw(b"\x00" * n)
    p.time = t
    return p


@pytest.fixture
def pcap(tmp_path):
    pkts = [
        # QUIC-like flow: client Initial padded to 1200, server replies, MTU-sized packet
        udp(C, S, 50000, 443, 1200, 100.000),
        udp(S, C, 443, 50000, 1472, 100.012),   # 12 ms later, max UDP payload at 1500 MTU
        udp(S, C, 443, 50000, 1000, 100.013),
        udp(C, S, 50000, 443, 40, 100.050),
        # same 5-tuple after a 100 s idle gap -> must be a NEW flow (inactive timeout 65 s)
        udp(C, S, 50000, 443, 1250, 200.050),
        udp(S, C, 443, 50000, 300, 200.060),
    ]
    # TCP flow: SYN / ACKs without payload must be skipped
    syn = Ether() / IP(src=C, dst=S) / TCP(sport=40000, dport=443, flags="S"); syn.time = 300.0
    data = Ether() / IP(src=C, dst=S) / TCP(sport=40000, dport=443, flags="PA") / Raw(b"\x01" * 517); data.time = 300.010
    ack = Ether() / IP(src=S, dst=C) / TCP(sport=443, dport=40000, flags="A"); ack.time = 300.020
    resp = Ether() / IP(src=S, dst=C) / TCP(sport=443, dport=40000, flags="PA") / Raw(b"\x02" * 1400); resp.time = 300.030
    # IPv6 UDP flow
    v6 = Ether() / IPv6(src="2001:db8::1", dst="2001:db8::2") / UDP(sport=50001, dport=443) / Raw(b"\x00" * 1200); v6.time = 400.0
    path = str(tmp_path / "test.pcap")
    wrpcap(path, pkts + [syn, data, ack, resp, v6])
    return path


def test_flow_count_and_timeout_split(pcap):
    flows = extract_flows_from_pcap(pcap)
    udp_flows = [k for k in flows if k[2] == 17 and k[0][0] == C]
    assert len(udp_flows) == 2, "idle gap > inactive timeout must start a new flow"
    assert len(flows) == 4                               # 2 UDP + 1 TCP + 1 IPv6


def test_quic_flow_sequence(pcap):
    flows = extract_flows_from_pcap(pcap)
    seq = flows[((C, 50000), (S, 443), 17, 1)]
    expected = np.array([[1200, 0, 1], [1472, 12, -1], [1000, 1, -1], [40, 37, 1]], np.float32)
    np.testing.assert_allclose(seq[:4], expected, atol=1e-3)
    assert seq.shape == (MAX_PACKETS, 3)
    assert not seq[4:].any(), "padding must be zeros"


def test_second_flow_after_timeout(pcap):
    seq = extract_flows_from_pcap(pcap)[((C, 50000), (S, 443), 17, 2)]
    np.testing.assert_allclose(seq[:2], [[1250, 0, 1], [300, 10, -1]], atol=1e-3)


def test_tcp_payload_only_and_direction(pcap):
    seq = extract_flows_from_pcap(pcap)[((C, 40000), (S, 443), 6, 1)]
    # SYN and bare ACK skipped; sizes are TCP payload bytes, not frame length
    np.testing.assert_allclose(seq[:2], [[517, 0, 1], [1400, 20, -1]], atol=1e-3)
    assert not seq[2:].any()


def test_tcp_keep_empty(pcap):
    seq = extract_flows_from_pcap(pcap, skip_empty=False)[((C, 40000), (S, 443), 6, 1)]
    np.testing.assert_allclose(seq[:4, 0], [0, 517, 0, 1400])
    np.testing.assert_allclose(seq[:4, 2], [1, 1, -1, -1])


def test_ipv6(pcap):
    flows = extract_flows_from_pcap(pcap)
    seq = flows[(("2001:db8::1", 50001), ("2001:db8::2", 443), 17, 1)]
    assert seq[0].tolist() == [1200, 0, 1]


def test_direction_follows_first_sender_not_address_order(tmp_path):
    # server address sorts BEFORE the client address; client must still be +1
    a = udp("192.168.1.10", "8.8.8.8", 51000, 443, 1200, 1.0)
    b = udp("8.8.8.8", "192.168.1.10", 443, 51000, 900, 1.005)
    path = str(tmp_path / "dir.pcap"); wrpcap(path, [a, b])
    (seq,) = extract_flows_from_pcap(path).values()
    assert seq[0, 2] == 1 and seq[1, 2] == -1
