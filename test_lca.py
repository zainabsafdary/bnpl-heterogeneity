"""Tests for the hand-written EM latent class model.

Run with:  pytest -q
(or without pytest:  python tests/test_lca.py)

The key idea: simulate data where we KNOW the true classes, then check that
the estimator recovers them. This is how you show an estimator you wrote
yourself is correct.
"""
from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shed_bnpl.lca import LatentClassModel  # noqa: E402

TRUE_PI = np.array([0.5, 0.3, 0.2])
TRUE_THETA = [  # 4 indicators; rows = classes, cols = categories
    np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]),
    np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    np.array([[0.85, 0.15], [0.85, 0.15], [0.1, 0.9]]),
    np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]),
]


def simulate(n: int, seed: int = 0, missing_rate: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.choice(len(TRUE_PI), size=n, p=TRUE_PI)
    data = {}
    for j, th in enumerate(TRUE_THETA):
        draws = np.array([rng.choice(th.shape[1], p=th[k]) for k in z])
        col = np.array([f"c{v}" for v in draws], dtype=object)
        if missing_rate:
            col[rng.random(n) < missing_rate] = np.nan
        data[f"x{j}"] = col
    return pd.DataFrame(data)


def best_match_error(model: LatentClassModel) -> tuple[float, float]:
    """Max abs error in pi and theta after matching estimated to true classes."""
    best = (np.inf, np.inf)
    for perm in permutations(range(len(TRUE_PI))):
        pi_err = np.abs(model.pi_[list(perm)] - TRUE_PI).max()
        th_err = max(np.abs(est[list(perm)] - true).max()
                     for est, true in zip(model.theta_, TRUE_THETA))
        if pi_err + th_err < sum(best):
            best = (pi_err, th_err)
    return best


def test_recovers_true_parameters():
    X = simulate(6000, seed=1)
    m = LatentClassModel(3, n_init=15, random_state=0).fit(X)
    pi_err, th_err = best_match_error(m)
    assert m.converged_
    assert pi_err < 0.06, pi_err
    assert th_err < 0.08, th_err


def test_handles_missing_answers():
    X = simulate(6000, seed=2, missing_rate=0.15)
    m = LatentClassModel(3, n_init=15, random_state=0).fit(X)
    pi_err, th_err = best_match_error(m)
    assert pi_err < 0.08 and th_err < 0.10, (pi_err, th_err)


def test_objective_never_decreases():
    X = simulate(1500, seed=3)
    m = LatentClassModel(3, n_init=1, random_state=4).fit(X)
    diffs = np.diff(m.history_)
    assert np.all(diffs > -1e-6 * np.abs(m.history_[1:])), diffs.min()


def test_weight_equals_duplicating_rows():
    """A weight of 2 must give the same estimates as listing the row twice.

    This is the core property of frequency/survey weights, so it is the
    cleanest check that weights enter the likelihood correctly.
    """
    X = simulate(800, seed=5)
    rng = np.random.default_rng(6)
    w = rng.choice([1.0, 2.0], size=len(X))
    X_dup = pd.concat([X, X[w == 2.0]], ignore_index=True)

    kw = dict(n_init=1, random_state=7, smoothing=0.0, tol=1e-12)
    with np.errstate(divide="ignore"):
        m_w = LatentClassModel(3, **kw).fit(X, w)
        m_d = LatentClassModel(3, **kw).fit(X_dup)
    assert np.allclose(m_w.pi_, m_d.pi_, atol=1e-6)
    for a, b in zip(m_w.theta_, m_d.theta_):
        assert np.allclose(a, b, atol=1e-6)


def test_bic_prefers_true_number_of_classes():
    X = simulate(5000, seed=8)
    bics = {k: LatentClassModel(k, n_init=10, random_state=0).fit(X).bic() for k in (2, 3, 4)}
    assert min(bics, key=bics.get) == 3, bics


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
