import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from features import fit_stats, transform, MAX_PACKETS, FEATURE_DIM
from baseline import derived_features, ppi_features
from train import truncate


def _flows(n=50, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, MAX_PACKETS, FEATURE_DIM), np.float32)
    X[..., 0] = rng.integers(40, 1472, (n, MAX_PACKETS))
    X[..., 1] = rng.exponential(10, (n, MAX_PACKETS))
    X[..., 2] = rng.choice([-1, 1], (n, MAX_PACKETS))
    return X


def test_transform_standardizes_train():
    X = _flows(200)
    m, s = fit_stats(X)
    Z = transform(X, m, s).reshape(-1, FEATURE_DIM)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(Z.std(0), 1, atol=1e-3)


def test_transform_handles_negative_iat():
    X = _flows(10); X[0, 3, 1] = -0.5                    # reordering can give tiny negative IAT
    m, s = fit_stats(X)
    assert np.isfinite(m).all() and np.isfinite(transform(X, m, s)).all()


def test_truncate_zeroes_tail_only():
    X = _flows(5)
    T = truncate(X, 10)
    assert (T[:, :10] == X[:, :10]).all() and not T[:, 10:].any()
    assert truncate(X, MAX_PACKETS) is X


def test_derived_features_known_values():
    X = np.zeros((1, MAX_PACKETS, 3), np.float32)
    X[0, :4] = [[1250, 0, 1], [1200, 5, -1], [100, 10, 1], [90, 1, -1]]
    F = np.expm1(derived_features(X, MAX_PACKETS))[0]
    expected = [4, 2, 2, 1350, 1290, 16, 675, 645, 1250, 1200, 4, 10, 3]
    np.testing.assert_allclose(F, expected, rtol=1e-4)


def test_ppi_features_use_only_first_packets():
    X = _flows(3)
    a = ppi_features(X, 10)
    X2 = X.copy(); X2[:, 10:] = 999                      # changing later packets must not matter
    assert a.shape == (3, 30) and np.array_equal(a, ppi_features(X2, 10))
