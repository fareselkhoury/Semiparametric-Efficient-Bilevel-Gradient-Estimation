#!/usr/bin/env python3
"""
Figure 2 experiment: inference diagnostics for the cross-fitted OBiGrad gradient.

This script is a standalone companion to ``IV_figure1_obigrad_gradient_experiment.py``.
It uses the same easy sine-IV DGP, but focuses on inference rather than RMSE:

    - empirical coverage of nominal 95% Wald intervals,
    - average CI length,
    - studentized / standardized errors,
    - QQ plot and histogram diagnostics for one coordinate.

Target
------
For fixed omega, estimate

    Psi(omega) = grad_omega 0.5 E[(Y - h^*_omega(X))^2],

where

    h^*_omega(X) = E[g_omega(T) | X],
    g_omega(T) = omega^T phi(T),
    phi_l(T) = sin(T + l), l=1,...,d.

Easy DGP
--------

    X ~ N(0, I_p)
    T = 2 * sum_j X_j + eta,       eta ~ N(0, sigma_t^2)
    Y = omega_star^T phi(T) + eps, eps ~ N(0, sigma_y^2)

The conditional nuisances are analytic:

    j^*(X) = E[phi(T) | X]
           = exp(-sigma_t^2 / 2) * sin(2 sum_j X_j + ell),

    h^*_omega(X) = j^*(X)^T omega,
    m^*(X)       = E[Y | X] = j^*(X)^T omega_star.

Therefore the true gradient is known:

    Psi(omega) = A (omega - omega_star),
    A = E[j^*(X) j^*(X)^T].

OBiGrad / orthogonal score
--------------------------
For each coordinate k,

    varphi_k(O; h, j, m)
      = (g_omega(T) - Y) j_k(X)
        + (phi_k(T) - j_k(X)) (h(X) - m(X)).

The estimator is the cross-fitted average of this score. The standard error is
computed from the empirical covariance of the cross-fitted pseudo-outcomes.

Outputs
-------
Inside --outdir, the script writes:

    figure2_coverage_vs_n.png
    figure2_ci_length_vs_n.png
    figure2_studentized_qq_coord{coord}_n{n}.png
    figure2_studentized_hist_coord{coord}_n{n}.png
    table_iv_figure2_wald.tex
    table_iv_figure2_studentized.tex

Recommended run
---------------

    python IV_figure2_obigrad_inference_experiment.py \
        --n-grid 200,400,800,1600,3200 \
        --reps 500 \
        --outdir results/IV/figure2

Fast smoke test
---------------

    python IV_figure2_obigrad_inference_experiment.py \
        --n-grid 100,200 \
        --reps 10 \
        --outdir results/IV/figure2_smoke
"""


import argparse
import math
import os
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_reporting import format_number, format_pm, write_latex_table


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DGPConfig:
    """Configuration for the easy sine-IV DGP."""

    p: int = 3
    d: int = 4
    sigma_t: float = math.sqrt(0.10)
    sigma_y: float = 0.25
    omega_star_scale: float = 1.0

    @property
    def omega_star(self):
        base = np.arange(1, self.d + 1, dtype=float)
        return self.omega_star_scale * base / np.linalg.norm(base)


@dataclass(frozen=True)
class LearnerConfig:
    """Nuisance learner configuration.

    Default ``sum_fourier`` matches the paper experiment: it uses observable
    Fourier features of sum(X), which is stable for the sine-IV benchmark.
    Set ``--learner rff`` for a more generic random-feature stress test.
    """

    kind: str = "sum_fourier"
    rff_dim: int = 512
    gamma: float = 2.0
    ridge_alpha: float = 1e-6
    fit_intercept: bool = True
    fourier_max_freq: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    omega_eval_shift: float = 0.35
    n_folds: int = 2
    seed: int = 12345
    reps: int = 500
    n_grid: Tuple[int, ...] = (200, 400, 800, 1600, 3200)
    alpha: float = 0.05
    coord: int = 0
    qq_n: Optional[int] = None
    outdir: str = "results/IV/figure2"


# -----------------------------------------------------------------------------
# DGP and analytic truth
# -----------------------------------------------------------------------------


def parse_n_grid(text):
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("--n-grid must contain at least one integer")
    if any(n < 20 for n in values):
        raise ValueError("All n values should be >= 20")
    return values


def phi_features(t, d):
    """Return phi_l(t)=sin(t+l), l=1,...,d, as an array of shape (n,d)."""
    ell = np.arange(1, d + 1, dtype=float)
    return np.sin(t[:, None] + ell[None, :])


def simulate_data(n, cfg, rng):
    """Generate one independent sample from the sine-IV DGP."""
    x = rng.normal(size=(n, cfg.p))
    eta = rng.normal(scale=cfg.sigma_t, size=n)
    t = 2.0 * x.sum(axis=1) + eta
    phi_t = phi_features(t, cfg.d)
    eps_y = rng.normal(scale=cfg.sigma_y, size=n)
    y = phi_t @ cfg.omega_star + eps_y
    return {"X": x, "T": t, "Phi": phi_t, "Y": y}


def true_conditional_phi(x, cfg):
    """Analytic j^*(X)=E[phi(T)|X]."""
    ell = np.arange(1, cfg.d + 1, dtype=float)
    attenuation = math.exp(-0.5 * cfg.sigma_t**2)
    return attenuation * np.sin(2.0 * x.sum(axis=1)[:, None] + ell[None, :])


def true_A_matrix(cfg):
    """Analytic A = E[j^*(X) j^*(X)^T].

    If S=sum_j X_j ~ N(0,p), then for j_l(X)=c sin(2S+l),

        E[j_l j_k] = c^2/2 * {cos(l-k) - exp(-8p) cos(l+k)}.
    """
    ell = np.arange(1, cfg.d + 1, dtype=float)
    l_minus_k = ell[:, None] - ell[None, :]
    l_plus_k = ell[:, None] + ell[None, :]
    c2 = math.exp(-cfg.sigma_t**2)
    return 0.5 * c2 * (np.cos(l_minus_k) - math.exp(-8.0 * cfg.p) * np.cos(l_plus_k))


def true_gradient(omega, cfg):
    return true_A_matrix(cfg) @ (omega - cfg.omega_star)


def make_omega_eval(cfg, omega_eval_shift):
    """A fixed target point separated from omega_star in a deterministic direction."""
    direction = np.linspace(1.0, -1.0, cfg.d)
    direction = direction / np.linalg.norm(direction)
    return cfg.omega_star + omega_eval_shift * direction


# -----------------------------------------------------------------------------
# Nuisance learners
# -----------------------------------------------------------------------------


class RFFRidgeRegressor:
    """Dependency-free RBF random Fourier feature ridge regressor.

    Multi-output. Used for h, j, and m nuisance regressions.
    """

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = int(seed)
        self.x_mean_ = None
        self.x_scale_ = None
        self.W_ = None
        self.b_ = None
        self.coef_ = None

    def _standardize_fit(self, x):
        self.x_mean_ = x.mean(axis=0, keepdims=True)
        self.x_scale_ = x.std(axis=0, keepdims=True)
        self.x_scale_ = np.where(self.x_scale_ < 1e-12, 1.0, self.x_scale_)
        return (x - self.x_mean_) / self.x_scale_

    def _standardize_transform(self, x):
        if self.x_mean_ is None or self.x_scale_ is None:
            raise RuntimeError("Model is not fitted")
        return (x - self.x_mean_) / self.x_scale_

    def _features_from_standardized(self, xs):
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted")
        z = math.sqrt(2.0 / self.cfg.rff_dim) * np.cos(xs @ self.W_ + self.b_)
        if self.cfg.fit_intercept:
            z = np.column_stack([np.ones(xs.shape[0]), z])
        return z

    def fit(self, x, y):
        if y.ndim == 1:
            y = y[:, None]
        rng = np.random.default_rng(self.seed)
        xs = self._standardize_fit(np.asarray(x, dtype=float))
        n_features = xs.shape[1]
        self.W_ = rng.normal(
            scale=math.sqrt(2.0 * self.cfg.gamma), size=(n_features, self.cfg.rff_dim)
        )
        self.b_ = rng.uniform(low=0.0, high=2.0 * math.pi, size=self.cfg.rff_dim)
        z = self._features_from_standardized(xs)
        gram = z.T @ z
        penalty = self.cfg.ridge_alpha * np.eye(gram.shape[0])
        if self.cfg.fit_intercept:
            penalty[0, 0] = 0.0
        rhs = z.T @ y
        try:
            self.coef_ = np.linalg.solve(gram + penalty, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")
        xs = self._standardize_transform(np.asarray(x, dtype=float))
        z = self._features_from_standardized(xs)
        pred = z @ self.coef_
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred


class SumFourierRidgeRegressor:
    """Fourier-sieve ridge regressor on S=sum_j X_j.

    This is not an oracle estimator: coefficients are learned on the training fold.
    It is deliberately aligned with the easy DGP to isolate inference behavior.
    """

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = int(seed)
        self.s_mean_ = None
        self.coef_ = None

    def _features(self, x, fit):
        s = np.asarray(x, dtype=float).sum(axis=1)
        if fit:
            self.s_mean_ = float(s.mean())
        if self.s_mean_ is None:
            raise RuntimeError("Model is not fitted")
        # Center only; the true frequency scale remains in the raw sum(X).
        s_centered = s - self.s_mean_
        cols = []
        if self.cfg.fit_intercept:
            cols.append(np.ones_like(s_centered))
        for k in range(1, self.cfg.fourier_max_freq + 1):
            cols.append(np.sin(k * s_centered))
            cols.append(np.cos(k * s_centered))
        return np.column_stack(cols)

    def fit(self, x, y):
        if y.ndim == 1:
            y = y[:, None]
        z = self._features(x, fit=True)
        gram = z.T @ z
        penalty = self.cfg.ridge_alpha * np.eye(gram.shape[0])
        if self.cfg.fit_intercept:
            penalty[0, 0] = 0.0
        rhs = z.T @ y
        try:
            self.coef_ = np.linalg.solve(gram + penalty, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")
        z = self._features(x, fit=False)
        pred = z @ self.coef_
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred


def make_regressor(cfg, seed):
    if cfg.kind == "sum_fourier":
        return SumFourierRidgeRegressor(cfg, seed=seed)
    if cfg.kind == "rff":
        return RFFRidgeRegressor(cfg, seed=seed)
    raise ValueError(f"Unknown learner kind: {cfg.kind!r}")


# -----------------------------------------------------------------------------
# Cross-fitted OBiGrad estimator and inference quantities
# -----------------------------------------------------------------------------


def make_folds(n, n_folds, rng):
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for cross-fitting")
    indices = rng.permutation(n)
    return [fold.astype(int) for fold in np.array_split(indices, n_folds)]


def covariance_se(phi):
    """Return empirical covariance of pseudo-outcomes and SE of their average."""
    n, d = phi.shape
    if n <= 1:
        raise ValueError("Need at least two observations for covariance")
    cov = np.cov(phi, rowvar=False, ddof=1)
    if d == 1:
        cov = np.array([[float(cov)]])
    se = np.sqrt(np.maximum(np.diag(cov), 0.0) / n)
    return cov, se


def estimate_one_replication(
    data,
    omega_eval,
    dgp_cfg,
    learner_cfg,
    n_folds,
    rng,
    seed_offset,
):
    """Estimate gradients and standard errors for one Monte Carlo sample.

    Returns arrays for three methods:
        plugin: cross-fitted plug-in score, included as a negative/diagnostic control.
        dr: cross-fitted OBiGrad orthogonal score.
        oracle: same score evaluated at analytic nuisances.
    """
    x, phi_t, y = data["X"], data["Phi"], data["Y"]
    n, d = phi_t.shape

    g_eval = phi_t @ omega_eval
    true_j = true_conditional_phi(x, dgp_cfg)
    true_h = true_j @ omega_eval
    true_m = true_j @ dgp_cfg.omega_star

    phi_plugin = np.zeros((n, d))
    phi_dr = np.zeros((n, d))
    hhat_oof = np.zeros(n)
    mhat_oof = np.zeros(n)
    jhat_oof = np.zeros((n, d))

    folds = make_folds(n, n_folds, rng)
    all_idx = np.arange(n)

    for fold_id, test_idx in enumerate(folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]

        targets = np.column_stack(
            [
                g_eval[train_idx],  # h target
                phi_t[train_idx, :],  # j target, all d coordinates
                y[train_idx],  # m target
            ]
        )

        learner = make_regressor(learner_cfg, seed=seed_offset + 1009 * fold_id)
        learner.fit(x[train_idx], targets)
        pred = learner.predict(x[test_idx])

        hhat = pred[:, 0]
        jhat = pred[:, 1 : 1 + d]
        mhat = pred[:, 1 + d]

        hhat_oof[test_idx] = hhat
        jhat_oof[test_idx, :] = jhat
        mhat_oof[test_idx] = mhat

        # Cross-fitted naive plug-in score.
        phi_plugin[test_idx, :] = jhat * (hhat - y[test_idx])[:, None]

        # Cross-fitted orthogonal OBiGrad score.
        phi_dr[test_idx, :] = (
            jhat * (g_eval[test_idx] - y[test_idx])[:, None]
            + (phi_t[test_idx, :] - jhat) * (hhat - mhat)[:, None]
        )

    # Oracle score using analytic nuisances.
    phi_oracle = (
        true_j * (g_eval - y)[:, None] + (phi_t - true_j) * (true_h - true_m)[:, None]
    )

    outputs = {
        "plugin_est": phi_plugin.mean(axis=0),
        "dr_est": phi_dr.mean(axis=0),
        "oracle_est": phi_oracle.mean(axis=0),
    }

    for method, phi in [
        ("plugin", phi_plugin),
        ("dr", phi_dr),
        ("oracle", phi_oracle),
    ]:
        cov, se = covariance_se(phi)
        outputs[f"{method}_cov"] = cov
        outputs[f"{method}_se"] = se

    outputs["err_h"] = math.sqrt(float(np.mean((hhat_oof - true_h) ** 2)))
    outputs["err_j"] = math.sqrt(float(np.mean((jhat_oof - true_j) ** 2)))
    outputs["err_m"] = math.sqrt(float(np.mean((mhat_oof - true_m) ** 2)))
    outputs["product_bias_proxy"] = float(outputs["err_j"]) * (
        float(outputs["err_h"]) + float(outputs["err_m"])
    )
    return outputs


def normal_quantile_975(alpha):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def run_experiment(
    dgp_cfg,
    learner_cfg,
    exp_cfg,
):
    os.makedirs(exp_cfg.outdir, exist_ok=True)
    if not (0 <= exp_cfg.coord < dgp_cfg.d):
        raise ValueError(f"--coord must be between 0 and d-1; got {exp_cfg.coord}")

    omega_eval = make_omega_eval(dgp_cfg, exp_cfg.omega_eval_shift)
    psi_true = true_gradient(omega_eval, dgp_cfg)
    zcrit = normal_quantile_975(exp_cfg.alpha)
    qq_n = exp_cfg.qq_n if exp_cfg.qq_n is not None else max(exp_cfg.n_grid)
    if qq_n not in exp_cfg.n_grid:
        raise ValueError("--qq-n must be one of the values in --n-grid")

    master_rng = np.random.default_rng(exp_cfg.seed)
    rep_rows = []
    long_rows = []

    methods = ["plugin", "dr", "oracle"]

    for n in exp_cfg.n_grid:
        print(f"n={n} reps={exp_cfg.reps}", flush=True)
        for rep in range(exp_cfg.reps):
            seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
            rng = np.random.default_rng(seed)
            data = simulate_data(n, dgp_cfg, rng)
            est = estimate_one_replication(
                data=data,
                omega_eval=omega_eval,
                dgp_cfg=dgp_cfg,
                learner_cfg=learner_cfg,
                n_folds=exp_cfg.n_folds,
                rng=rng,
                seed_offset=seed + 17,
            )

            wide = {"n": float(n), "rep": float(rep), "seed": float(seed)}
            for method in methods:
                estimate = np.asarray(est[f"{method}_est"], dtype=float)
                se = np.asarray(est[f"{method}_se"], dtype=float)
                error = estimate - psi_true
                z = np.divide(
                    error, se, out=np.full_like(error, np.nan), where=se > 0.0
                )
                cover = np.abs(error) <= zcrit * se
                ci_len = 2.0 * zcrit * se

                wide[f"{method}_mean_coverage"] = float(np.mean(cover))
                wide[f"{method}_mean_ci_length"] = float(np.mean(ci_len))
                wide[f"{method}_l2_error"] = float(np.linalg.norm(error))

                for k in range(dgp_cfg.d):
                    wide[f"{method}_est_{k}"] = float(estimate[k])
                    wide[f"{method}_se_{k}"] = float(se[k])
                    wide[f"{method}_err_{k}"] = float(error[k])
                    wide[f"{method}_z_{k}"] = float(z[k])
                    wide[f"{method}_cover_{k}"] = float(cover[k])
                    wide[f"{method}_ci_len_{k}"] = float(ci_len[k])

                    long_rows.append(
                        {
                            "n": float(n),
                            "rep": float(rep),
                            "seed": float(seed),
                            "method": method,
                            "coord": float(k),
                            "estimate": float(estimate[k]),
                            "truth": float(psi_true[k]),
                            "error": float(error[k]),
                            "se": float(se[k]),
                            "z": float(z[k]),
                            "covered": float(cover[k]),
                            "ci_length": float(ci_len[k]),
                        }
                    )

            for name in ["err_h", "err_j", "err_m", "product_bias_proxy"]:
                wide[name] = float(est[name])
            rep_rows.append(wide)

        partial_summary = summarize_by_n_and_method(
            long_rows, methods, dgp_cfg.d, n_filter=n
        )
        for row in partial_summary:
            if row["coord"] == -1.0:
                print(
                    f"  {row['method']:<7s} coverage={row['coverage']:.3f} "
                    f"ci_len={row['ci_length_mean']:.4g} z_std={row['z_std']:.3f}",
                    flush=True,
                )

    summary_rows = summarize_by_n_and_method(
        long_rows, methods, dgp_cfg.d, n_filter=None
    )
    add_nuisance_summary(summary_rows, rep_rows)

    make_coverage_plot(summary_rows, exp_cfg.outdir)
    make_ci_length_plot(summary_rows, exp_cfg.outdir)
    make_studentized_qq_plot(
        long_rows=long_rows,
        outdir=exp_cfg.outdir,
        coord=exp_cfg.coord,
        n_value=qq_n,
        methods=("dr", "oracle"),
    )
    make_studentized_hist_plot(
        long_rows=long_rows,
        outdir=exp_cfg.outdir,
        coord=exp_cfg.coord,
        n_value=qq_n,
        methods=("dr", "oracle"),
    )
    write_latex_outputs(summary_rows, exp_cfg.outdir)
    return rep_rows, long_rows, summary_rows


# -----------------------------------------------------------------------------
# Summaries and output utilities
# -----------------------------------------------------------------------------


def finite_array(values):
    arr = np.array(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def quantile_or_nan(values, q):
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def summarize_by_n_and_method(
    long_rows,
    methods,
    d,
    n_filter,
):
    """Create per-n, per-method summaries, both pooled and coordinate-specific."""
    n_values = sorted({int(row["n"]) for row in long_rows})
    if n_filter is not None:
        n_values = [int(n_filter)]
    out = []

    for n in n_values:
        for method in methods:
            # coord=-1 means pooled across all coordinates.
            for coord in [-1] + list(range(d)):
                subset = [
                    row
                    for row in long_rows
                    if int(row["n"]) == n
                    and row["method"] == method
                    and (coord == -1 or int(row["coord"]) == coord)
                ]
                if not subset:
                    continue
                errors = finite_array(row["error"] for row in subset)
                zs = finite_array(row["z"] for row in subset)
                ci_lengths = finite_array(row["ci_length"] for row in subset)
                ses = finite_array(row["se"] for row in subset)
                covers = finite_array(row["covered"] for row in subset)

                out.append(
                    {
                        "n": float(n),
                        "method": method,
                        "coord": float(coord),
                        "coverage": (
                            float(np.mean(covers)) if covers.size else float("nan")
                        ),
                        "coverage_se": (
                            math.sqrt(float(np.mean(covers)) * (1.0 - float(np.mean(covers))) / covers.size)
                            if covers.size
                            else float("nan")
                        ),
                        "ci_length_mean": (
                            float(np.mean(ci_lengths))
                            if ci_lengths.size
                            else float("nan")
                        ),
                        "ci_length_se": (
                            float(np.std(ci_lengths, ddof=1) / math.sqrt(ci_lengths.size))
                            if ci_lengths.size > 1
                            else 0.0
                        ),
                        "ci_length_median": (
                            float(np.median(ci_lengths))
                            if ci_lengths.size
                            else float("nan")
                        ),
                        "se_mean": float(np.mean(ses)) if ses.size else float("nan"),
                        "bias": float(np.mean(errors)) if errors.size else float("nan"),
                        "rmse": (
                            math.sqrt(float(np.mean(errors**2)))
                            if errors.size
                            else float("nan")
                        ),
                        "rmse_se": (
                            float(
                                np.std(errors**2, ddof=1)
                                / math.sqrt(errors.size)
                                / (2.0 * math.sqrt(float(np.mean(errors**2))))
                            )
                            if errors.size > 1
                            and math.sqrt(float(np.mean(errors**2))) > 0
                            else 0.0
                        ),
                        "mae": (
                            float(np.mean(np.abs(errors)))
                            if errors.size
                            else float("nan")
                        ),
                        "z_mean": float(np.mean(zs)) if zs.size else float("nan"),
                        "z_std": (
                            float(np.std(zs, ddof=1)) if zs.size > 1 else float("nan")
                        ),
                        "z_q025": quantile_or_nan(zs, 0.025),
                        "z_q500": quantile_or_nan(zs, 0.500),
                        "z_q975": quantile_or_nan(zs, 0.975),
                        "num_values": float(len(subset)),
                    }
                )
    return out


def add_nuisance_summary(summary_rows, rep_rows):
    """Attach nuisance diagnostics to pooled rows, leaving coord-specific rows blank."""
    by_n = {}
    for row in rep_rows:
        by_n.setdefault(int(row["n"]), []).append(row)

    for row in summary_rows:
        if row["coord"] != -1.0:
            row["err_h_mean"] = float("nan")
            row["err_j_mean"] = float("nan")
            row["err_m_mean"] = float("nan")
            row["product_bias_proxy_mean"] = float("nan")
            continue
        rows_n = by_n[int(row["n"])]
        for name in ["err_h", "err_j", "err_m", "product_bias_proxy"]:
            row[f"{name}_mean"] = float(np.mean([r[name] for r in rows_n]))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def pooled_summary(summary_rows):
    return [row for row in summary_rows if row["coord"] == -1.0]


def ordered_methods(rows):
    desired = ["dr", "oracle", "plugin"]
    present = {row["method"] for row in rows}
    return [m for m in desired if m in present]


def display_label(method):
    return {
        "dr": "OBiGrad / orthogonal DR",
        "oracle": "Oracle DR",
        "plugin": "Plug-in normal interval",
    }.get(method, method)


def make_coverage_plot(summary_rows, outdir):
    rows = pooled_summary(summary_rows)
    methods = ordered_methods(rows)

    plt.figure(figsize=(7.2, 4.8))
    for method, marker in zip(methods, ["s", "^", "o"]):
        sub = sorted(
            [row for row in rows if row["method"] == method], key=lambda r: r["n"]
        )
        ns = np.array([row["n"] for row in sub], dtype=float)
        coverage = np.array([row["coverage"] for row in sub], dtype=float)
        se = np.array([row.get("coverage_se", 0.0) for row in sub], dtype=float)
        plt.errorbar(
            ns,
            coverage,
            yerr=1.96 * se,
            marker=marker,
            linewidth=2,
            capsize=3,
            label=display_label(method),
        )

    plt.axhline(0.95, linestyle="--", linewidth=1.5, label="nominal 95%")
    plt.xscale("log")
    plt.ylim(0.75, 1.02)
    plt.xlabel("sample size n")
    plt.ylabel("empirical coordinate-wise coverage")
    plt.title("Figure 2: Wald interval coverage")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure2_coverage_vs_n.png"), dpi=240)
    plt.close()


def make_ci_length_plot(summary_rows, outdir):
    rows = pooled_summary(summary_rows)
    methods = ordered_methods(rows)

    plt.figure(figsize=(7.2, 4.8))
    for method, marker in zip(methods, ["s", "^", "o"]):
        sub = sorted(
            [row for row in rows if row["method"] == method], key=lambda r: r["n"]
        )
        ns = np.array([row["n"] for row in sub], dtype=float)
        lengths = np.array([row["ci_length_mean"] for row in sub], dtype=float)
        se = np.array([row.get("ci_length_se", 0.0) for row in sub], dtype=float)
        plt.errorbar(
            ns,
            lengths,
            yerr=1.96 * se,
            marker=marker,
            linewidth=2,
            capsize=3,
            label=display_label(method),
        )

    # Reference n^{-1/2} anchored to OBiGrad at the largest n when available.
    dr_sub = sorted(
        [row for row in rows if row["method"] == "dr"], key=lambda r: r["n"]
    )
    if dr_sub:
        ns_ref = np.array([row["n"] for row in dr_sub], dtype=float)
        len_ref = np.array([row["ci_length_mean"] for row in dr_sub], dtype=float)
        anchor = len_ref[-1] * math.sqrt(ns_ref[-1])
        plt.loglog(
            ns_ref,
            anchor / np.sqrt(ns_ref),
            linestyle="--",
            linewidth=1.5,
            label=r"reference $n^{-1/2}$",
        )

    plt.xlabel("sample size n")
    plt.ylabel("mean 95% CI length")
    plt.title("Figure 2: Wald interval length")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure2_ci_length_vs_n.png"), dpi=240)
    plt.close()


def write_latex_outputs(summary_rows, outdir):
    rows = pooled_summary(summary_rows)
    table_rows = []
    z_rows = []
    for n in sorted({int(row["n"]) for row in rows}):
        dr = next(row for row in rows if int(row["n"]) == n and row["method"] == "dr")
        plugin = next(
            (row for row in rows if int(row["n"]) == n and row["method"] == "plugin"),
            None,
        )
        oracle = next(
            (row for row in rows if int(row["n"]) == n and row["method"] == "oracle"),
            None,
        )
        table_rows.append(
            [
                str(n),
                format_pm(dr["coverage"], dr.get("coverage_se"), digits=3),
                format_pm(dr["ci_length_mean"], dr.get("ci_length_se"), digits=4),
                format_pm(dr["rmse"], dr.get("rmse_se"), digits=4),
                format_number(dr.get("product_bias_proxy_mean"), digits=3),
                format_pm(oracle["coverage"], oracle.get("coverage_se"), digits=3)
                if oracle
                else "--",
            ]
        )
        z_rows.append(
            [
                str(n),
                format_number(dr["z_mean"], digits=3),
                format_number(dr["z_std"], digits=3),
                format_number(dr["z_q025"], digits=3),
                format_number(dr["z_q500"], digits=3),
                format_number(dr["z_q975"], digits=3),
                format_number(plugin["coverage"], digits=3) if plugin else "--",
            ]
        )

    write_latex_table(
        os.path.join(outdir, "table_iv_figure2_wald.tex"),
        "OBiGrad Wald calibration on the vector sine-IV design.",
        "tab:generated-iv-wald",
        [r"$n$", "DR coverage", "DR length", "DR RMSE", "Product bias", "Oracle coverage"],
        table_rows,
    )
    write_latex_table(
        os.path.join(outdir, "table_iv_figure2_studentized.tex"),
        "Studentized-error diagnostics for the vector sine-IV inference experiment.",
        "tab:generated-iv-studentized",
        [r"$n$", "Mean", "SD", r"2.5\%", "Median", r"97.5\%", "Plug-in coverage"],
        z_rows,
    )


def get_z_values(
    long_rows,
    method,
    coord,
    n_value,
):
    z = finite_array(
        row["z"]
        for row in long_rows
        if row["method"] == method
        and int(row["coord"]) == coord
        and int(row["n"]) == n_value
    )
    return z


def make_studentized_qq_plot(
    long_rows,
    outdir,
    coord,
    n_value,
    methods,
):
    normal = NormalDist()
    plt.figure(figsize=(6.2, 6.2))
    min_seen = -3.0
    max_seen = 3.0

    for method, marker in zip(methods, ["o", "^"]):
        z = np.sort(get_z_values(long_rows, method, coord, n_value))
        if z.size == 0:
            continue
        probs = (np.arange(1, z.size + 1) - 0.5) / z.size
        theoretical = np.array([normal.inv_cdf(float(p)) for p in probs])
        plt.scatter(
            theoretical, z, marker=marker, s=18, alpha=0.75, label=display_label(method)
        )
        min_seen = min(min_seen, float(np.min(theoretical)), float(np.min(z)))
        max_seen = max(max_seen, float(np.max(theoretical)), float(np.max(z)))

    lim = max(abs(min_seen), abs(max_seen), 3.0)
    plt.plot(
        [-lim, lim], [-lim, lim], linestyle="--", linewidth=1.5, label="standard normal"
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("theoretical standard-normal quantile")
    plt.ylabel("empirical studentized error quantile")
    plt.title(f"Figure 2: QQ diagnostic, coord {coord}, n={n_value}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"figure2_studentized_qq_coord{coord}_n{n_value}.png"),
        dpi=240,
    )
    plt.close()


def make_studentized_hist_plot(
    long_rows,
    outdir,
    coord,
    n_value,
    methods,
):
    plt.figure(figsize=(7.2, 4.8))
    all_z = []
    for method in methods:
        z = get_z_values(long_rows, method, coord, n_value)
        all_z.extend(z.tolist())
    if all_z:
        lo = float(np.quantile(all_z, 0.005))
        hi = float(np.quantile(all_z, 0.995))
        bound = max(abs(lo), abs(hi), 3.0)
    else:
        bound = 3.0

    bins = np.linspace(-bound, bound, 31)
    for method, histtype in zip(methods, ["step", "stepfilled"]):
        z = get_z_values(long_rows, method, coord, n_value)
        if z.size == 0:
            continue
        plt.hist(
            z,
            bins=bins,
            density=True,
            alpha=0.35,
            histtype=histtype,
            label=display_label(method),
        )

    grid = np.linspace(-bound, bound, 300)
    density = np.exp(-0.5 * grid**2) / math.sqrt(2.0 * math.pi)
    plt.plot(
        grid, density, linestyle="--", linewidth=1.5, label="standard normal density"
    )
    plt.xlabel("studentized error")
    plt.ylabel("density")
    plt.title(f"Figure 2: studentized-error histogram, coord {coord}, n={n_value}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"figure2_studentized_hist_coord{coord}_n{n_value}.png"),
        dpi=240,
    )
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Figure 2 OBiGrad inference experiment."
    )
    parser.add_argument("--n-grid", type=str, default="200,400,800,1600,3200")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=str, default="results/IV/figure2")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--sigma-t", type=float, default=math.sqrt(0.10))
    parser.add_argument("--sigma-y", type=float, default=0.25)
    parser.add_argument("--omega-eval-shift", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--coord", type=int, default=0)
    parser.add_argument(
        "--qq-n",
        type=int,
        default=None,
        help="n value for QQ/hist plots; default=max(n-grid)",
    )
    parser.add_argument(
        "--learner", type=str, default="sum_fourier", choices=["sum_fourier", "rff"]
    )
    parser.add_argument("--rff-dim", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--fourier-max-freq", type=int, default=8)
    return parser


def main():
    args = build_arg_parser().parse_args()
    dgp_cfg = DGPConfig(
        p=args.p,
        d=args.d,
        sigma_t=args.sigma_t,
        sigma_y=args.sigma_y,
    )
    learner_cfg = LearnerConfig(
        kind=args.learner,
        rff_dim=args.rff_dim,
        gamma=args.gamma,
        ridge_alpha=args.ridge_alpha,
        fourier_max_freq=args.fourier_max_freq,
    )
    exp_cfg = ExperimentConfig(
        omega_eval_shift=args.omega_eval_shift,
        n_folds=args.folds,
        seed=args.seed,
        reps=args.reps,
        n_grid=parse_n_grid(args.n_grid),
        alpha=args.alpha,
        coord=args.coord,
        qq_n=args.qq_n,
        outdir=args.outdir,
    )
    run_experiment(dgp_cfg, learner_cfg, exp_cfg)


if __name__ == "__main__":
    main()
