import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from model import ENCODERS, build_encoder, SupConLoss, CosineHead
from features import MAX_PACKETS, FEATURE_DIM


@pytest.mark.parametrize("name", list(ENCODERS))
def test_encoder_shape_and_norm(name):
    torch.manual_seed(0)
    m = build_encoder(name).eval()
    z = m(torch.randn(8, MAX_PACKETS, FEATURE_DIM))
    assert z.shape == (8, 128)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(8), atol=1e-5, rtol=0)


@pytest.mark.parametrize("name", list(ENCODERS))
def test_encoder_independent_of_batch(name):
    # eval-mode embedding of a flow must not depend on the other flows in the batch
    torch.manual_seed(0)
    m = build_encoder(name).eval()
    x = torch.randn(4, MAX_PACKETS, FEATURE_DIM)
    with torch.no_grad():
        torch.testing.assert_close(m(x)[:1], m(x[:1]), atol=1e-5, rtol=1e-4)


def test_supcon_finite_and_lower_for_clustered():
    loss = SupConLoss()
    y = torch.tensor([0, 0, 1, 1])
    good = torch.nn.functional.normalize(torch.tensor([[1., 0], [1, .01], [0, 1], [.01, 1]]), dim=1)
    bad = torch.nn.functional.normalize(torch.tensor([[1., 0], [0, 1], [1, .01], [.01, 1]]), dim=1)
    assert torch.isfinite(loss(good, y)) and loss(good, y) < loss(bad, y)


def test_cosine_head_scale():
    h = CosineHead(128, 5)
    z = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)
    assert h(z).abs().max() <= 16.0 + 1e-4
