#!/usr/bin/env python3
"""
Figure 1 experiment: fixed-omega gradient estimation for a simple semiparametric
bilevel/IV problem.

Goal
----
Compare the naive plug-in gradient, cross-fitted OBiGrad/DR gradient, and oracle
DR gradient for the unregularized population target

    Psi(omega) = grad_omega 0.5 E[(Y - h^*_omega(X))^2]

where

    h^*_omega(X) = E[g_omega(T) | X],
    g_omega(T) = omega^T phi(T),
    phi_l(T) = sin(T + l), l=1,...,d.

DGP
---
    X ~ N(0, I_p)
    T = 2 * sum_j X_j + eta,       eta ~ N(0, sigma_t^2)
    Y = omega_star^T phi(T) + eps, eps ~ N(0, sigma_y^2)

Then the nuisance functions are analytic:

    j^*_omega(X) = E[partial_omega g_omega(T) | X]
                  = E[phi(T) | X]
                  = exp(-sigma_t^2 / 2) * sin(2 sum_j X_j + ell)

    h^*_omega(X) = j^*(X)^T omega
    m^*(X)       = E[Y | X] = j^*(X)^T omega_star

and the true gradient is

    Psi(omega) = A (omega - omega_star),
    A = E[j^*(X) j^*(X)^T].

This script estimates h, j, m by cross-fitted sieve/ridge regression and evaluates the orthogonal OBiGrad score

    phi_k(O; h, j, m)
      = (g_omega(T) - Y) j_k(X)
        + (partial_omega_k g_omega(T) - j_k(X)) (h(X) - m(X)).

Outputs
-------
    figure1_gradient_rmse.png
    figure1_dr_coverage.png
    table_iv_figure1_gradient.tex
    table_iv_figure1_nuisance.tex

Example
-------
    python IV_figure1_obigrad_gradient_experiment.py \
        --n-grid 200,400,800,1600,3200 \
        --reps 300 \
        --rff-dim 512 \
        --outdir results/IV/figure1

A fast smoke test:
    python IV_figure1_obigrad_gradient_experiment.py --n-grid 80,160 --reps 5 --rff-dim 128
"""


import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_reporting import format_number, format_pm, write_latex_table


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
    reps: int = 200
    n_grid: Tuple[int, ...] = (200, 400, 800, 1600, 3200)
    outdir: str = "results/IV/figure1"


def parse_n_grid(text):
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("--n-grid must contain at least one integer")
    if any(n < 20 for n in values):
        raise ValueError("All n values should be >= 20")
    return values


def phi_features(t, d):
    """phi_l(t) = sin(t + l), l=1,...,d. Returns shape (n, d)."""
    ell = np.arange(1, d + 1, dtype=float)
    return np.sin(t[:, None] + ell[None, :])


def simulate_data(n, cfg, rng):
    """Generate one sample from the sine-IV DGP."""
    x = rng.normal(size=(n, cfg.p))
    eta = rng.normal(scale=cfg.sigma_t, size=n)
    t = 2.0 * x.sum(axis=1) + eta
    phi_t = phi_features(t, cfg.d)
    eps_y = rng.normal(scale=cfg.sigma_y, size=n)
    y = phi_t @ cfg.omega_star + eps_y
    return {"X": x, "T": t, "Phi": phi_t, "Y": y}


def true_conditional_phi(x, cfg):
    """j^*(X) = E[phi(T) | X] for the Gaussian-noise sine DGP."""
    ell = np.arange(1, cfg.d + 1, dtype=float)
    attenuation = math.exp(-0.5 * cfg.sigma_t**2)
    return attenuation * np.sin(2.0 * x.sum(axis=1)[:, None] + ell[None, :])


def true_A_matrix(cfg):
    """Analytic A = E[j^*(X) j^*(X)^T].

    With X_j iid N(0,1), S=sum_j X_j ~ N(0,p). For j_l(X)=c sin(2S+l),

        E[j_l j_k] = c^2/2 * {cos(l-k) - exp(-8p) cos(l+k)}.
    """
    ell = np.arange(1, cfg.d + 1, dtype=float)
    l_minus_k = ell[:, None] - ell[None, :]
    l_plus_k = ell[:, None] + ell[None, :]
    c2 = math.exp(-cfg.sigma_t**2)
    return 0.5 * c2 * (np.cos(l_minus_k) - math.exp(-8.0 * cfg.p) * np.cos(l_plus_k))


def true_gradient(omega, cfg):
    return true_A_matrix(cfg) @ (omega - cfg.omega_star)


class RFFRidgeRegressor:
    """Dependency-free RBF random Fourier feature ridge regressor.

    The model is multi-output. It is used as the nuisance learner for h, j, and m.

    Feature map:
        z(x) = sqrt(2 / rff_dim) cos(x_standardized W + b),
        W_ij ~ N(0, 2 gamma), b_j ~ Unif(0, 2pi).
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

    This is not an oracle estimator of the nuisances: the coefficients are
    learned from data. The feature class is still hand-aligned with the toy DGP,
    so it is best viewed as an optional benchmark rather than the generic
    default.
    """

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = int(seed)
        self.s_mean_ = None
        self.s_scale_ = None
        self.coef_ = None

    def _features(self, x, fit):
        s = np.asarray(x, dtype=float).sum(axis=1)
        if fit:
            self.s_mean_ = float(s.mean())
            self.s_scale_ = float(s.std())
            if self.s_scale_ < 1e-12:
                self.s_scale_ = 1.0
        if self.s_mean_ is None or self.s_scale_ is None:
            raise RuntimeError("Model is not fitted")
        # Do not standardize the frequency scale away entirely: the true signal is
        # sin(2 * sum(X) + ell). We center only for numerical conditioning.
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


def make_folds(n, n_folds, rng):
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for cross-fitting")
    indices = rng.permutation(n)
    return [fold.astype(int) for fold in np.array_split(indices, n_folds)]


def estimate_gradients_one_sample(
    data,
    omega_eval,
    dgp_cfg,
    learner_cfg,
    n_folds,
    rng,
    seed_offset,
):
    """Compute plug-in, cross-fitted DR, and oracle DR gradients for one sample."""
    x, phi_t, y = data["X"], data["Phi"], data["Y"]
    n, d = phi_t.shape

    g_eval = phi_t @ omega_eval
    true_j = true_conditional_phi(x, dgp_cfg)
    true_h = true_j @ omega_eval
    true_m = true_j @ dgp_cfg.omega_star

    phi_dr_oof = np.zeros((n, d))
    phi_pi_oof = np.zeros((n, d))
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
                g_eval[train_idx],  # h target, scalar
                phi_t[train_idx, :],  # j target, d outputs
                y[train_idx],  # m target, scalar
            ]
        )
        nuisance = make_regressor(learner_cfg, seed=seed_offset + 1009 * fold_id)
        nuisance.fit(x[train_idx], targets)
        pred = nuisance.predict(x[test_idx])

        hhat = pred[:, 0]
        jhat = pred[:, 1 : 1 + d]
        mhat = pred[:, 1 + d]

        hhat_oof[test_idx] = hhat
        jhat_oof[test_idx, :] = jhat
        mhat_oof[test_idx] = mhat

        g_test = g_eval[test_idx]
        y_test = y[test_idx]
        phi_test = phi_t[test_idx, :]

        # Naive plug-in gradient score: jhat(X) * (hhat(X) - Y).
        phi_pi_oof[test_idx, :] = jhat * (hhat - y_test)[:, None]

        # Orthogonal OBiGrad/DR score.
        phi_dr_oof[test_idx, :] = (
            jhat * (g_test - y_test)[:, None]
            + (phi_test - jhat) * (hhat - mhat)[:, None]
        )

    # Oracle DR score, using analytic nuisance functions.
    phi_oracle = (
        true_j * (g_eval - y)[:, None] + (phi_t - true_j) * (true_h - true_m)[:, None]
    )

    plugin = phi_pi_oof.mean(axis=0)
    dr = phi_dr_oof.mean(axis=0)
    oracle = phi_oracle.mean(axis=0)

    cov_dr = np.cov(phi_dr_oof, rowvar=False)
    if d == 1:
        cov_dr = np.array([[float(cov_dr)]])
    se_dr = np.sqrt(np.maximum(np.diag(cov_dr), 0.0) / n)

    err_h = math.sqrt(float(np.mean((hhat_oof - true_h) ** 2)))
    err_j = math.sqrt(float(np.mean((jhat_oof - true_j) ** 2)))
    err_m = math.sqrt(float(np.mean((mhat_oof - true_m) ** 2)))
    product_bias_proxy = err_j * (err_h + err_m)

    return {
        "plugin": plugin,
        "dr": dr,
        "oracle": oracle,
        "se_dr": se_dr,
        "err_h": err_h,
        "err_j": err_j,
        "err_m": err_m,
        "product_bias_proxy": product_bias_proxy,
    }


def summarize_replications(rep_rows, d):
    """Aggregate Monte Carlo rows for one n."""
    out = {}
    for method in ["plugin", "dr", "oracle"]:
        l2_errors = np.array(
            [row[f"{method}_l2_error"] for row in rep_rows], dtype=float
        )
        rmse, rmse_se = rmse_and_se(l2_errors)
        out[f"{method}_l2_rmse"] = rmse
        out[f"{method}_l2_rmse_se"] = rmse_se
        out[f"{method}_l2_mae"] = float(np.mean(l2_errors))
        for k in range(d):
            errors_k = np.array(
                [row[f"{method}_err_{k}"] for row in rep_rows], dtype=float
            )
            out[f"{method}_coord{k}_bias"] = float(np.mean(errors_k))
            out[f"{method}_coord{k}_rmse"] = math.sqrt(float(np.mean(errors_k**2)))
    coverage = np.array([row["dr_mean_coord_coverage_95"] for row in rep_rows])
    out["dr_mean_coord_coverage_95"] = float(np.mean(coverage))
    out["dr_mean_coord_coverage_95_se"] = (
        float(np.std(coverage, ddof=1) / math.sqrt(len(coverage)))
        if len(coverage) > 1
        else 0.0
    )
    for name in ["err_h", "err_j", "err_m", "product_bias_proxy"]:
        out[name] = float(np.mean([row[name] for row in rep_rows]))
    return out


def rmse_and_se(errors):
    sq = np.asarray(errors, dtype=float) ** 2
    rmse = math.sqrt(float(np.mean(sq)))
    if sq.size <= 1 or rmse <= 0:
        return rmse, 0.0
    se_mse = float(np.std(sq, ddof=1) / math.sqrt(sq.size))
    return rmse, se_mse / (2.0 * rmse)


def run_experiment(
    dgp_cfg,
    learner_cfg,
    exp_cfg,
):
    os.makedirs(exp_cfg.outdir, exist_ok=True)

    omega_star = dgp_cfg.omega_star
    direction = np.linspace(1.0, -1.0, dgp_cfg.d)
    direction = direction / np.linalg.norm(direction)
    omega_eval = omega_star + exp_cfg.omega_eval_shift * direction
    psi_true = true_gradient(omega_eval, dgp_cfg)

    master_rng = np.random.default_rng(exp_cfg.seed)
    replications = []
    summary = []

    for n in exp_cfg.n_grid:
        rows_for_n = []
        print(f"n={n} reps={exp_cfg.reps}", flush=True)
        for rep in range(exp_cfg.reps):
            seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
            rng = np.random.default_rng(seed)
            data = simulate_data(n, dgp_cfg, rng)
            est = estimate_gradients_one_sample(
                data=data,
                omega_eval=omega_eval,
                dgp_cfg=dgp_cfg,
                learner_cfg=learner_cfg,
                n_folds=exp_cfg.n_folds,
                rng=rng,
                seed_offset=seed + 17,
            )

            row = {"n": float(n), "rep": float(rep), "seed": float(seed)}
            for method in ["plugin", "dr", "oracle"]:
                estimate = np.asarray(est[method], dtype=float)
                err = estimate - psi_true
                row[f"{method}_l2_error"] = float(np.linalg.norm(err))
                for k in range(dgp_cfg.d):
                    row[f"{method}_est_{k}"] = float(estimate[k])
                    row[f"{method}_err_{k}"] = float(err[k])

            se_dr = np.asarray(est["se_dr"], dtype=float)
            dr_est = np.asarray(est["dr"], dtype=float)
            cover = np.abs(dr_est - psi_true) <= 1.96 * se_dr
            row["dr_mean_coord_coverage_95"] = float(np.mean(cover))
            for k in range(dgp_cfg.d):
                row[f"dr_se_{k}"] = float(se_dr[k])
                row[f"dr_cover95_{k}"] = float(cover[k])
                row[f"psi_true_{k}"] = float(psi_true[k])

            for name in ["err_h", "err_j", "err_m", "product_bias_proxy"]:
                row[name] = float(est[name])

            replications.append(row)
            rows_for_n.append(row)

        s = summarize_replications(rows_for_n, dgp_cfg.d)
        s["n"] = float(n)
        summary.append(s)
        print(
            "  RMSE L2: "
            f"plugin={s['plugin_l2_rmse']:.4g}, "
            f"dr={s['dr_l2_rmse']:.4g}, "
            f"oracle={s['oracle_l2_rmse']:.4g}, "
            f"DR cov={s['dr_mean_coord_coverage_95']:.3f}",
            flush=True,
        )

    make_plots(summary, exp_cfg.outdir)
    write_latex_outputs(summary, exp_cfg.outdir)
    return summary, replications


def make_plots(summary, outdir):
    ns = np.array([row["n"] for row in summary], dtype=float)

    plt.figure(figsize=(7.0, 5.0))
    for method, label, marker in [
        ("plugin", "Plug-in", "o"),
        ("dr", "OBiGrad / orthogonal DR", "s"),
        ("oracle", "Oracle DR", "^"),
    ]:
        rmse = np.array([row[f"{method}_l2_rmse"] for row in summary], dtype=float)
        rmse_se = np.array(
            [row.get(f"{method}_l2_rmse_se", 0.0) for row in summary], dtype=float
        )
        plt.errorbar(
            ns,
            rmse,
            yerr=1.96 * rmse_se,
            marker=marker,
            linewidth=2,
            capsize=3,
            label=label,
        )

    oracle_rmse = np.array([row["oracle_l2_rmse"] for row in summary], dtype=float)
    anchor = oracle_rmse[-1] * math.sqrt(ns[-1])
    plt.loglog(
        ns,
        anchor / np.sqrt(ns),
        linestyle="--",
        linewidth=1.5,
        label=r"reference $n^{-1/2}$",
    )
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("sample size n")
    plt.ylabel(r"Monte Carlo RMSE of $\widehat\Psi_\omega$ (L2 norm)")
    plt.title("Fixed-omega gradient estimation")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure1_gradient_rmse.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    coverage = np.array(
        [row["dr_mean_coord_coverage_95"] for row in summary], dtype=float
    )
    coverage_se = np.array(
        [row.get("dr_mean_coord_coverage_95_se", 0.0) for row in summary], dtype=float
    )
    plt.errorbar(
        ns,
        coverage,
        yerr=1.96 * coverage_se,
        marker="s",
        linewidth=2,
        capsize=3,
        label="OBiGrad mean coordinate coverage",
    )
    plt.axhline(0.95, linestyle="--", linewidth=1.5, label="nominal 95%")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("sample size n")
    plt.ylabel("coverage")
    plt.title("OBiGrad Wald interval coverage")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure1_dr_coverage.png"), dpi=220)
    plt.close()


def write_latex_outputs(summary, outdir):
    main_rows = []
    appendix_rows = []
    for row in summary:
        n = int(row["n"])
        main_rows.append(
            [
                str(n),
                format_pm(row["plugin_l2_rmse"], row.get("plugin_l2_rmse_se")),
                format_pm(row["dr_l2_rmse"], row.get("dr_l2_rmse_se")),
                format_pm(row["oracle_l2_rmse"], row.get("oracle_l2_rmse_se")),
                format_pm(
                    row["dr_mean_coord_coverage_95"],
                    row.get("dr_mean_coord_coverage_95_se"),
                    digits=3,
                ),
                format_number(row["product_bias_proxy"], digits=3),
            ]
        )
        appendix_rows.append(
            [
                str(n),
                format_number(row["err_h"], digits=4),
                format_number(row["err_j"], digits=4),
                format_number(row["err_m"], digits=4),
                format_number(row["product_bias_proxy"], digits=4),
            ]
        )

    write_latex_table(
        os.path.join(outdir, "table_iv_figure1_gradient.tex"),
        r"Fixed-omega IV gradient estimation. Parentheses report Monte Carlo 95\% error bars for RMSE.",
        "tab:generated-iv-gradient",
        [r"$n$", "Plug-in", "OBiGrad", "Oracle DR", "DR coverage", "Product bias"],
        main_rows,
    )
    write_latex_table(
        os.path.join(outdir, "table_iv_figure1_nuisance.tex"),
        "Nuisance-learning diagnostics for the fixed-omega IV experiment.",
        "tab:generated-iv-nuisance",
        [r"$n$", r"$\|\hat h-h^\star\|$", r"$\|\hat j-j^\star\|$", r"$\|\hat m-m^\star\|$", "Product bias"],
        appendix_rows,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Figure 1 OBiGrad gradient experiment."
    )
    parser.add_argument("--n-grid", type=str, default="200,400,800,1600,3200")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=str, default="results/IV/figure1")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--sigma-t", type=float, default=math.sqrt(0.10))
    parser.add_argument("--sigma-y", type=float, default=0.25)
    parser.add_argument("--omega-eval-shift", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=2)
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
        outdir=args.outdir,
    )
    run_experiment(dgp_cfg, learner_cfg, exp_cfg)


if __name__ == "__main__":
    main()
