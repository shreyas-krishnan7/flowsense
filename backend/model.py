import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

# All encoders map (B, seq_len, feat) -> (B, embed_dim) L2-normalized embeddings,
# so they are interchangeable for SupCon training, k-NN and novelty detection.


# ---------- pooling ----------
class AttentionPool(nn.Module):
    """Learned query attends over timesteps -> weighted average.
    The weights also show which packets drove the decision."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
    def forward(self, h):                       # h: (B, T, C)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)   # (B, T)
        return (h * w.unsqueeze(-1)).sum(1), w


def _pool(h, pool):                             # h: (B, T, C)
    return pool(h)[0] if pool is not None else h.mean(dim=1)


# ---------- TCN ----------
class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return x[:, :, :-self.c].contiguous() if self.c else x

class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, k, dilation, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_c, out_c, k, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_c, out_c, k, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout))
        self.down = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.relu(self.net(x) + res)

class TCNEncoder(nn.Module):
    def __init__(self, in_feat=3, channels=(64, 64, 128), k=3, embed_dim=128,
                 dropout=0.1, attn_pool=False):
        super().__init__()
        layers, prev = [], in_feat
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(prev, ch, k, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = AttentionPool(prev) if attn_pool else None
        self.proj = nn.Linear(prev, embed_dim)
    def forward(self, x):                       # x: (B, seq_len, feat)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, C)
        return F.normalize(self.proj(_pool(h, self.pool)), dim=-1)


# ---------- plain 1-D CNN (simple baseline encoder) ----------
class CNNEncoder(nn.Module):
    def __init__(self, in_feat=3, channels=(64, 128, 128), k=5, embed_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_feat
        for ch in channels:
            layers += [nn.Conv1d(prev, ch, k, padding=k // 2), nn.BatchNorm1d(ch),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(prev, embed_dim)
    def forward(self, x):
        h = self.cnn(x.transpose(1, 2)).mean(dim=-1)
        return F.normalize(self.proj(h), dim=-1)


# ---------- GRU ----------
class GRUEncoder(nn.Module):
    def __init__(self, in_feat=3, hidden=128, layers=2, embed_dim=128, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_feat, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(2 * hidden, embed_dim)
    def forward(self, x):
        h, _ = self.gru(x)                      # (B, T, 2H)
        return F.normalize(self.proj(h.mean(dim=1)), dim=-1)


# ---------- small Transformer ----------
class TransformerFlowEncoder(nn.Module):
    def __init__(self, in_feat=3, d_model=128, heads=4, layers=2, ffn=256,
                 max_len=64, embed_dim=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ffn, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.pool = AttentionPool(d_model)
        self.proj = nn.Linear(d_model, embed_dim)
    def forward(self, x):
        h = self.inp(x) + self.pos[:, :x.size(1)]
        h = self.enc(h)
        return F.normalize(self.proj(_pool(h, self.pool)), dim=-1)


ENCODERS = {
    "cnn": lambda: CNNEncoder(),
    "tcn": lambda: TCNEncoder(),
    "tcn_attn": lambda: TCNEncoder(attn_pool=True),
    "gru": lambda: GRUEncoder(),
    "transformer": lambda: TransformerFlowEncoder(),
}

def build_encoder(name):
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}', choose from {list(ENCODERS)}")
    return ENCODERS[name]()

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------- losses ----------
class CosineHead(nn.Module):
    """Classification head for cross-entropy training on L2-normalized
    embeddings (normalized softmax): logits = s * cos(z, w_c)."""
    def __init__(self, embed_dim, n_classes, scale=16.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, embed_dim) / math.sqrt(embed_dim))
        self.s = scale
    def forward(self, z):
        return self.s * z @ F.normalize(self.w, dim=-1).T


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.07): super().__init__(); self.t = temperature
    def forward(self, feats, labels):
        B = feats.size(0); dev = feats.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(dev)
        sim = feats @ feats.T / self.t
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()   # stability
        self_mask = 1 - torch.eye(B, device=dev)
        mask = mask * self_mask
        exp = torch.exp(sim) * self_mask
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
        pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -pos.mean()
