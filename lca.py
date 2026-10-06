"""Survey-weighted latent class analysis (LCA), estimated by EM from scratch.

Model
-----
Each person i belongs to one of K unobserved classes. Class k has share pi_k.
Given the class, the J categorical indicators are independent ("local
independence"), and indicator j takes category c with probability
theta[j][k, c]. The likelihood for one person is

    L_i = sum_k  pi_k * prod_j theta[j][k, x_ij]

and the survey-weighted log-likelihood is  sum_i w_i * log L_i.

Estimation (EM = maximum likelihood with missing class labels)
-------------------------------------------------------------
E-step: posterior class probabilities  r_ik = P(class k | x_i).
M-step: pi_k      = sum_i w_i r_ik / sum_i w_i
        theta_jkc = sum_i w_i r_ik 1[x_ij = c] / sum_i w_i r_ik 1[x_ij observed]

Details that matter in real survey data
---------------------------------------
* Missing answers (skip patterns, NaN) are dropped from that person's product,
  which is valid under missing-at-random. No one is thrown out.
* A tiny Dirichlet pseudo-count (`smoothing`) keeps probabilities off exactly
  0, which avoids log(0) and degenerate "boundary" solutions. EM then
  maximizes the penalized (MAP) objective, which is what we monitor.
* EM only finds a local maximum, so we run many random starts and keep the
  best. We also report how many starts reached it: a sign of a stable answer.
* Weights are rescaled to sum to n, so the log-likelihood stays on the scale
  of the sample and BIC uses the actual number of respondents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp


class LatentClassModel:
    def __init__(
        self,
        n_classes: int,
        n_init: int = 20,
        max_iter: int = 2000,
        tol: float = 1e-8,
        smoothing: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing
        self.random_state = random_state

    # ------------------------------------------------------------------ data
    def _encode_fit(self, X: pd.DataFrame) -> np.ndarray:
        self.columns_ = list(X.columns)
        self.categories_ = []
        codes = np.empty(X.shape, dtype=np.int64)
        for j, col in enumerate(self.columns_):
            cat = pd.Categorical(X[col])
            self.categories_.append(list(cat.categories))
            codes[:, j] = cat.codes  # NaN -> -1
        return codes

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        codes = np.empty((len(X), len(self.columns_)), dtype=np.int64)
        for j, col in enumerate(self.columns_):
            codes[:, j] = pd.Categorical(X[col], categories=self.categories_[j]).codes
        return codes

    def _one_hots(self, codes: np.ndarray) -> list[np.ndarray]:
        """For each indicator, an n x C_j matrix; rows are all-zero if missing."""
        mats = []
        for j, cats in enumerate(self.categories_):
            m = np.zeros((codes.shape[0], len(cats)))
            obs = codes[:, j] >= 0
            m[np.where(obs)[0], codes[obs, j]] = 1.0
            mats.append(m)
        return mats

    # ------------------------------------------------------------ EM pieces
    @staticmethod
    def _log_cond(one_hots, log_theta) -> np.ndarray:
        """n x K matrix of log P(x_i | class k), skipping missing answers."""
        return sum(oh @ lt.T for oh, lt in zip(one_hots, log_theta))

    def _e_step(self, one_hots, log_pi, log_theta, w):
        joint = self._log_cond(one_hots, log_theta) + log_pi
        ll_i = logsumexp(joint, axis=1)
        resp = np.exp(joint - ll_i[:, None])
        return resp, float(w @ ll_i)

    def _m_step(self, one_hots, resp, w):
        wr = resp * w[:, None]
        pi = wr.sum(axis=0) / w.sum()
        theta = []
        for oh in one_hots:
            counts = wr.T @ oh + self.smoothing          # K x C_j
            theta.append(counts / counts.sum(axis=1, keepdims=True))
        return pi, theta

    def _objective(self, ll, theta) -> float:
        """Penalized log-likelihood that MAP-EM increases at every step."""
        return ll + self.smoothing * sum(np.log(t).sum() for t in theta)

    def _single_run(self, one_hots, w, rng):
        K = self.n_classes
        log_pi = np.log(np.full(K, 1.0 / K))
        theta = [rng.dirichlet(np.ones(oh.shape[1]), size=K) for oh in one_hots]
        history, converged, prev = [], False, -np.inf
        for _ in range(self.max_iter):
            resp, ll = self._e_step(one_hots, log_pi, [np.log(t) for t in theta], w)
            obj = self._objective(ll, theta)
            history.append(obj)
            if obj - prev < self.tol * abs(obj):
                converged = True
                break
            prev = obj
            pi, theta = self._m_step(one_hots, resp, w)
            log_pi = np.log(np.clip(pi, 1e-300, None))
        return dict(pi=np.exp(log_pi), theta=theta, loglik=ll, objective=obj,
                    history=history, converged=converged)

    # --------------------------------------------------------------- public
    def fit(self, X: pd.DataFrame, weights=None) -> "LatentClassModel":
        codes = self._encode_fit(X)
        n = codes.shape[0]
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        if np.any(~np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        w = w * n / w.sum()
        one_hots = self._one_hots(codes)

        rng = np.random.default_rng(self.random_state)
        runs = [self._single_run(one_hots, w, rng) for _ in range(self.n_init)]
        best = max(runs, key=lambda r: r["objective"])

        # Order classes from largest to smallest so output is stable across runs.
        order = np.argsort(-best["pi"])
        self.pi_ = best["pi"][order]
        self.theta_ = [t[order] for t in best["theta"]]
        self.loglik_ = best["loglik"]
        self.history_ = best["history"]
        self.converged_ = best["converged"]
        self.n_obs_ = n
        objs = np.array([r["objective"] for r in runs])
        # Starts within 0.01 log-lik units reached the same optimum; tiny gaps
        # come only from where each run stopped under the convergence tolerance.
        self.n_starts_at_best_ = int(np.sum(objs.max() - objs < 0.01))

        self._train_resp = self.predict_proba(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        one_hots = self._one_hots(self._encode(X))
        joint = self._log_cond(one_hots, [np.log(t) for t in self.theta_]) + np.log(self.pi_)
        return np.exp(joint - logsumexp(joint, axis=1)[:, None])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------- model fit stats
    @property
    def n_params(self) -> int:
        K = self.n_classes
        return (K - 1) + K * sum(len(c) - 1 for c in self.categories_)

    def bic(self) -> float:
        return -2 * self.loglik_ + self.n_params * np.log(self.n_obs_)

    def aic(self) -> float:
        return -2 * self.loglik_ + 2 * self.n_params

    def entropy(self) -> float:
        """Relative entropy in [0, 1]; near 1 means classes are well separated."""
        if self.n_classes == 1:
            return float("nan")
        r = np.clip(self._train_resp, 1e-300, 1)
        return float(1 - (-(r * np.log(r)).sum()) / (self.n_obs_ * np.log(self.n_classes)))

    def summary(self) -> dict:
        return dict(
            K=self.n_classes, loglik=self.loglik_, n_params=self.n_params,
            AIC=self.aic(), BIC=self.bic(), entropy=self.entropy(),
            smallest_class=float(self.pi_.min()), converged=self.converged_,
            starts_at_best=f"{self.n_starts_at_best_}/{self.n_init}",
        )

    def profiles(self) -> pd.DataFrame:
        """Tidy table: P(category | class) for every indicator."""
        rows = []
        for col, cats, th in zip(self.columns_, self.categories_, self.theta_):
            for k in range(self.n_classes):
                for c, cat in enumerate(cats):
                    rows.append(dict(cls=k + 1, class_share=self.pi_[k],
                                     variable=col, category=cat, prob=th[k, c]))
        return pd.DataFrame(rows)


def compare_k(X: pd.DataFrame, weights, k_values, **kwargs):
    """Fit one model per K and return (fit-statistics table, dict of models)."""
    models, stats = {}, []
    for k in k_values:
        m = LatentClassModel(n_classes=k, **kwargs).fit(X, weights)
        models[k] = m
        stats.append(m.summary())
    return pd.DataFrame(stats), models


def distal_outcome_by_class(resp: np.ndarray, y, weights) -> pd.DataFrame:
    """Weighted outcome mean per class using proportional (soft) assignment.

    Each person counts toward every class in proportion to their posterior
    probability. This is a simple first pass; the bias-adjusted three-step
    (BCH) method is the rigorous upgrade and a good next step.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    obs = np.isfinite(y)
    rows = []
    for k in range(resp.shape[1]):
        wk = w[obs] * resp[obs, k]
        mean = float(wk @ y[obs] / wk.sum())
        eff_n = float(wk.sum() ** 2 / (wk ** 2).sum())   # Kish effective sample size
        se = float(np.sqrt(mean * (1 - mean) / eff_n))
        rows.append(dict(cls=k + 1, mean=mean, se_approx=se,
                         ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se,
                         effective_n=eff_n))
    return pd.DataFrame(rows)
