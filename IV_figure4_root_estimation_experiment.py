#!/usr/bin/env python3
"""
Figure 4 experiment: root/optimizer estimation.

The goal is to compare root estimation for the unregularized semiparametric
bilevel target against KBO-style fixed-regularization roots.

This script is self-contained but intentionally mirrors the scalar version of
the uploaded KBO code:

    C_hat = K(X_outer, X_inner) @ (K(X_inner, X_inner) + n * lambda * I)^(-1) T_inner
    omega_hat_KBO = <C_hat, Y_outer> / <C_hat, C_hat>.

The OBiGrad root uses the orthogonal score for the quadratic specialization:

    phi_omega(O; h, j, m)
      = (omega T - Y) j(X) + (T - j(X)) (h_omega(X) - m(X)).

For the easy scalar DGP,
    X ~ N(0, I_p),
    T = 2 sum_j X_j + eta,
    Y = omega_star * T + rho eta + eps_y,

we have
    j*(X) = E[T|X] = 2 sum_j X_j,
    m*(X) = E[Y|X] = omega_star j*(X),
    h*_omega(X) = omega j*(X),
    Psi_0(omega) = 4 p (omega - omega_star),

so the true root is omega_star.

Outputs
-------
    figure4_root_rmse_vs_n.png
    figure4_root_bias_sd_vs_n.png
    figure4_kbo_decomposition_vs_n.png
    table_iv_figure4_root_rmse.tex
    table_iv_figure4_kbo_population_roots.tex

Paper-level example
-------------------
    python IV_figure4_root_estimation_experiment.py \
      --n-grid 100,200,400,800,1600 \
      --reps 300 \
      --pop-n 2500 \
      --outdir results/IV/figure4
"""


import argparse
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_reporting import format_number, format_pm, write_latex_table


@dataclass(frozen=True)
class DGPConfig:
    p: int = 3
    sigma_t: float = math.sqrt(0.10)
    endog_strength: float = 0.5
    sigma_y: float = 0.10
    omega_star: float = 2.0


@dataclass(frozen=True)
class LearnerConfig:
    learner: str = "linear"  # "linear" or "rff"
    ridge_alpha: float = 1e-8
    fit_intercept: bool = True
    rff_dim: int = 512
    rff_sigma: float = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    n_grid: Tuple[int, ...] = (100, 200, 400, 800, 1600)
    reps: int = 300
    seed: int = 20260425
    k_folds: int = 2
    outdir: str = "results/IV/figure4"

    # KBO kernel and regularization.
    kernel_sigma: float = 2.0
    kbo_fixed_lambda: float = 1e-2
    kbo_decay_c: float = 0.05
    kbo_decay_alpha: float = 0.60

    # Population proxy for the KBO regularized root.
    pop_n: int = 2500


def parse_int_grid(text):
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("Grid must contain at least one integer.")
    if any(v < 20 for v in values):
        raise ValueError("Use sample sizes >= 20.")
    return values


def parse_float_grid(text):
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("Grid must contain at least one positive float.")
    if any(v <= 0 for v in values):
        raise ValueError("All float-grid values must be positive.")
    return values


def simulate_data(n, cfg, rng):
    x = rng.normal(size=(n, cfg.p))
    eta = rng.normal(scale=cfg.sigma_t, size=n)
    t = 2.0 * x.sum(axis=1) + eta
    eps = rng.normal(scale=cfg.sigma_y, size=n) if cfg.sigma_y > 0 else np.zeros(n)
    y = cfg.omega_star * t + cfg.endog_strength * eta + eps
    return {"X": x, "T": t, "Y": y, "eta": eta}


def true_j(x):
    return 2.0 * np.asarray(x).sum(axis=1)


def true_m(x, cfg):
    return cfg.omega_star * true_j(x)


def true_gradient(omega, cfg):
    return 4.0 * cfg.p * (omega - cfg.omega_star)


def safe_ratio(num, den, min_abs_den=1e-12):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < min_abs_den:
        return float("nan")
    return float(num / den)


class BaseRegressor:
    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class LinearRidgeRegressor(BaseRegressor):
    """Multi-output ridge regression on [1, X].

    This is an optional hand-aligned benchmark for the toy DGP, where
    j*(X)=2 sum_j X_j and m*(X)=omega_star j*(X) are linear in X.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.coef_ = None

    def _features(self, x):
        x = np.asarray(x, dtype=float)
        if self.cfg.fit_intercept:
            return np.column_stack([np.ones(x.shape[0]), x])
        return x

    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        z = self._features(x)
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
            raise RuntimeError("Regressor is not fitted.")
        pred = self._features(x) @ self.coef_
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred


class RFFRidgeRegressor(BaseRegressor):
    """Multi-output ridge regression on Gaussian random Fourier features."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.w_ = None
        self.b_ = None
        self.coef_ = None

    def _features(self, x, fit):
        x = np.asarray(x, dtype=float)
        if fit:
            self.w_ = self.rng.normal(
                scale=1.0 / self.cfg.rff_sigma, size=(x.shape[1], self.cfg.rff_dim)
            )
            self.b_ = self.rng.uniform(0.0, 2.0 * math.pi, size=self.cfg.rff_dim)
        if self.w_ is None or self.b_ is None:
            raise RuntimeError("RFF regressor is not fitted.")
        z = math.sqrt(2.0 / self.cfg.rff_dim) * np.cos(x @ self.w_ + self.b_)
        if self.cfg.fit_intercept:
            z = np.column_stack([np.ones(x.shape[0]), z])
        return z

    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
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
            raise RuntimeError("Regressor is not fitted.")
        pred = self._features(x, fit=False) @ self.coef_
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred


def make_regressor(cfg, rng):
    if cfg.learner == "linear":
        return LinearRidgeRegressor(cfg)
    if cfg.learner == "rff":
        return RFFRidgeRegressor(cfg, rng)
    raise ValueError(f"Unknown learner: {cfg.learner}")


def make_folds(n, k_folds, rng):
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2.")
    perm = rng.permutation(n)
    return [fold for fold in np.array_split(perm, k_folds) if fold.size > 0]


def autodml_plugin_oracle_roots(
    data,
    dgp_cfg,
    learner_cfg,
    k_folds,
    rng,
):
    """K-fold cross-fitted OBiGrad and plug-in roots.

    The nuisance regressions are:
        j*(X)=E[T|X],
        m*(X)=E[Y|X].

    The OBiGrad score is linear in omega:
        phi_omega = omega * [(2T-jhat)jhat]
                    - [Y*jhat + (T-jhat)*mhat].
    """

    x = data["X"]
    t = data["T"]
    y = data["Y"]
    n = x.shape[0]
    folds = make_folds(n, k_folds, rng)

    dr_den_parts = []
    dr_num_parts = []
    plugin_den_parts = []
    plugin_num_parts = []
    j_mse_parts = []
    m_mse_parts = []

    for val_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.nonzero(train_mask)[0]

        targets = np.column_stack([t[train_idx], y[train_idx]])
        fold_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int32).max)))
        reg = make_regressor(learner_cfg, fold_rng).fit(x[train_idx], targets)
        pred = reg.predict(x[val_idx])
        j_hat = pred[:, 0]
        m_hat = pred[:, 1]

        tv = t[val_idx]
        yv = y[val_idx]

        dr_den = (2.0 * tv - j_hat) * j_hat
        dr_num = yv * j_hat + (tv - j_hat) * m_hat
        dr_den_parts.append(dr_den)
        dr_num_parts.append(dr_num)

        plugin_den_parts.append(j_hat * j_hat)
        plugin_num_parts.append(yv * j_hat)

        jt = true_j(x[val_idx])
        mt = true_m(x[val_idx], dgp_cfg)
        j_mse_parts.append(float(np.mean((j_hat - jt) ** 2)))
        m_mse_parts.append(float(np.mean((m_hat - mt) ** 2)))

    dr_den_all = np.concatenate(dr_den_parts)
    dr_num_all = np.concatenate(dr_num_parts)
    plugin_den_all = np.concatenate(plugin_den_parts)
    plugin_num_all = np.concatenate(plugin_num_parts)

    jt_all = true_j(x)
    mt_all = true_m(x, dgp_cfg)

    oracle_dr_den = (2.0 * t - jt_all) * jt_all
    oracle_dr_num = y * jt_all + (t - jt_all) * mt_all
    oracle_plugin_den = jt_all * jt_all
    oracle_plugin_num = y * jt_all

    return {
        "OBiGrad": safe_ratio(float(np.mean(dr_num_all)), float(np.mean(dr_den_all))),
        "Plug-in": safe_ratio(
            float(np.mean(plugin_num_all)), float(np.mean(plugin_den_all))
        ),
        "Oracle DR": safe_ratio(
            float(np.mean(oracle_dr_num)), float(np.mean(oracle_dr_den))
        ),
        "Oracle plug-in": safe_ratio(
            float(np.mean(oracle_plugin_num)), float(np.mean(oracle_plugin_den))
        ),
        "j_rmse": math.sqrt(float(np.mean(j_mse_parts))),
        "m_rmse": math.sqrt(float(np.mean(m_mse_parts))),
        "autodml_den_mean": float(np.mean(dr_den_all)),
        "plugin_den_mean": float(np.mean(plugin_den_all)),
    }


def gaussian_kernel_gram(x, y, sigma):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    sqdist = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.exp(-sqdist / (2.0 * sigma**2))


class KBOScalarFactorization:
    """Reusable KBO kernel matrices for one or several lambdas.

    For this root experiment the default run uses only two lambda values per
    replication (fixed lambda and decreasing lambda). Repeated linear solves are
    faster and more stable than an eigendecomposition in that regime.
    """

    def __init__(self, x_inner, x_outer, kernel_sigma):
        self.x_inner = x_inner
        self.x_outer = x_outer
        self.kernel_sigma = kernel_sigma
        self.n = x_inner.shape[0]
        k11 = gaussian_kernel_gram(x_inner, x_inner, kernel_sigma)
        self.k11 = 0.5 * (k11 + k11.T)
        self.k21 = gaussian_kernel_gram(x_outer, x_inner, kernel_sigma)

    def predict(self, y_inner, lam):
        y_inner = np.asarray(y_inner, dtype=float)
        mat = self.k11 + self.n * lam * np.eye(self.n)
        try:
            alpha = np.linalg.solve(mat, y_inner)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(mat, y_inner, rcond=None)[0]
        return self.k21 @ alpha


def kbo_roots_from_split(
    inner,
    outer,
    lambdas,
    kernel_sigma,
):
    fact = KBOScalarFactorization(inner["X"], outer["X"], kernel_sigma)
    roots = {}
    for lam in lambdas:
        c_hat = fact.predict(inner["T"], float(lam))
        roots[float(lam)] = safe_ratio(
            float(np.mean(c_hat * outer["Y"])), float(np.mean(c_hat * c_hat))
        )
    return roots


def all_population_lambdas(exp_cfg):
    vals = [float(exp_cfg.kbo_fixed_lambda)]
    for n in exp_cfg.n_grid:
        vals.append(
            float(exp_cfg.kbo_decay_c * (float(n) ** (-exp_cfg.kbo_decay_alpha)))
        )
    return tuple(sorted(set(round(v, 16) for v in vals)))


def kbo_population_roots(
    dgp_cfg,
    exp_cfg,
    lambdas,
):
    rng = np.random.default_rng(exp_cfg.seed + 99173)
    n_inner = exp_cfg.pop_n // 2
    n_outer = exp_cfg.pop_n - n_inner

    x_inner = rng.normal(size=(n_inner, dgp_cfg.p))
    x_outer = rng.normal(size=(n_outer, dgp_cfg.p))
    j_inner = true_j(x_inner)
    m_outer = true_m(x_outer, dgp_cfg)

    fact = KBOScalarFactorization(x_inner, x_outer, exp_cfg.kernel_sigma)
    roots = {}
    for lam in lambdas:
        c_lam = fact.predict(j_inner, float(lam))
        roots[float(lam)] = safe_ratio(
            float(np.mean(c_lam * m_outer)), float(np.mean(c_lam * c_lam))
        )
    return roots


def rmse(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return math.sqrt(float(np.mean(arr * arr)))


def summarize(
    replications,
    dgp_cfg,
    population_root_by_n_method,
):
    n_values = sorted(set(int(float(r["n"])) for r in replications))
    methods = sorted(set(str(r["method"]) for r in replications))

    rows_out = []
    for n in n_values:
        for method in methods:
            rows = [
                r
                for r in replications
                if int(float(r["n"])) == n and str(r["method"]) == method
            ]
            if not rows:
                continue
            roots = np.asarray([float(r["root_hat"]) for r in rows], dtype=float)
            roots = roots[np.isfinite(roots)]
            if roots.size == 0:
                continue
            errors = roots - dgp_cfg.omega_star
            rmse_true, rmse_true_se = rmse_with_se(errors)
            bias_se = (
                float(np.std(errors, ddof=1) / math.sqrt(errors.size))
                if errors.size > 1
                else 0.0
            )
            sd_root = float(np.std(roots, ddof=1)) if roots.size > 1 else float("nan")
            out = {
                "n": float(n),
                "method": method,
                "n_reps_finite": float(roots.size),
                "mean_root": float(np.mean(roots)),
                "bias_to_true": float(np.mean(errors)),
                "abs_bias_to_true": float(abs(np.mean(errors))),
                "abs_bias_to_true_se": bias_se,
                "sd_root": sd_root,
                "sd_root_se": (
                    sd_root / math.sqrt(2.0 * (roots.size - 1.0))
                    if roots.size > 1 and math.isfinite(sd_root)
                    else 0.0
                ),
                "rmse_to_true": rmse_true,
                "rmse_to_true_se": rmse_true_se,
                "mae_to_true": float(np.mean(np.abs(errors))),
            }
            pop_root = population_root_by_n_method.get((n, method))
            if pop_root is not None and np.isfinite(pop_root):
                err_to_pop = roots - pop_root
                rmse_pop, rmse_pop_se = rmse_with_se(err_to_pop)
                out["population_root"] = float(pop_root)
                out["regularization_bias_abs"] = float(
                    abs(pop_root - dgp_cfg.omega_star)
                )
                out["regularization_bias_abs_se"] = 0.0
                out["rmse_to_population_root"] = rmse_pop
                out["rmse_to_population_root_se"] = rmse_pop_se
            # Nuisance RMSEs are defined only for OBiGrad/plugin rows but are useful
            # because they come from the same cross-fitted run.
            if rows and "j_rmse" in rows[0]:
                vals_j = [
                    float(r["j_rmse"])
                    for r in rows
                    if "j_rmse" in r and np.isfinite(float(r["j_rmse"]))
                ]
                vals_m = [
                    float(r["m_rmse"])
                    for r in rows
                    if "m_rmse" in r and np.isfinite(float(r["m_rmse"]))
                ]
                if vals_j:
                    out["mean_j_rmse"] = float(np.mean(vals_j))
                if vals_m:
                    out["mean_m_rmse"] = float(np.mean(vals_m))
            rows_out.append(out)
    return rows_out


def rmse_with_se(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    sq = values**2
    out = math.sqrt(float(np.mean(sq)))
    if values.size <= 1 or out <= 0:
        return out, 0.0
    se_mse = float(np.std(sq, ddof=1) / math.sqrt(values.size))
    return out, se_mse / (2.0 * out)


def make_plots(summary, outdir):
    order = [
        "OBiGrad",
        "Oracle DR",
        "Plug-in",
        "Oracle plug-in",
        "KBO fixed lambda",
        "KBO decaying lambda",
    ]
    methods = [m for m in order if any(str(r["method"]) == m for r in summary)]
    n_values = sorted(set(int(float(r["n"])) for r in summary))
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    eps = 1e-14

    def series(method, key):
        xs = []
        ys = []
        for n in n_values:
            rows = [
                r
                for r in summary
                if str(r["method"]) == method and int(float(r["n"])) == n
            ]
            if rows and key in rows[0]:
                val = float(rows[0][key])
                if np.isfinite(val):
                    xs.append(float(n))
                    ys.append(val)
        return np.asarray(xs), np.asarray(ys)

    def series_err(method, key, se_key):
        xs, ys = series(method, key)
        ses = []
        for x in xs:
            row = next(
                r
                for r in summary
                if str(r["method"]) == method and int(float(r["n"])) == int(x)
            )
            ses.append(float(row.get(se_key, 0.0)))
        return xs, ys, np.asarray(ses)

    plt.figure(figsize=(7.8, 5.4))
    for i, method in enumerate(methods):
        xs, ys, ses = series_err(method, "rmse_to_true", "rmse_to_true_se")
        if xs.size:
            plt.errorbar(
                xs,
                np.maximum(ys, eps),
                yerr=1.96 * ses,
                marker=markers[i % len(markers)],
                linewidth=2.0,
                capsize=3,
                label=method,
            )

    xs_auto, ys_auto = series("OBiGrad", "rmse_to_true")
    if xs_auto.size:
        n_ref = np.asarray(n_values, dtype=float)
        ref = ys_auto[-1] * np.sqrt(xs_auto[-1] / n_ref)
        plt.loglog(
            n_ref,
            np.maximum(ref, eps),
            linestyle="--",
            linewidth=1.4,
            label=r"$n^{-1/2}$ reference",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sample size n")
    plt.ylabel(r"root RMSE $|\widehat{\omega}-\omega^\star|$")
    plt.title("Root estimation for the unregularized bilevel target")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure4_root_rmse_vs_n.png"), dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for i, method in enumerate(methods):
        xs, ys, ses = series_err(method, "abs_bias_to_true", "abs_bias_to_true_se")
        if xs.size:
            axes[0].errorbar(
                xs,
                np.maximum(ys, eps),
                yerr=1.96 * ses,
                marker=markers[i % len(markers)],
                linewidth=2.0,
                capsize=3,
                label=method,
            )
    axes[0].set_xlabel("sample size n")
    axes[0].set_ylabel("absolute empirical bias")
    axes[0].set_title("Bias to true root")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)

    for i, method in enumerate(methods):
        xs, ys, ses = series_err(method, "sd_root", "sd_root_se")
        if xs.size:
            axes[1].errorbar(
                xs,
                np.maximum(ys, eps),
                yerr=1.96 * ses,
                marker=markers[i % len(markers)],
                linewidth=2.0,
                capsize=3,
                label=method,
            )
    axes[1].set_xlabel("sample size n")
    axes[1].set_ylabel("standard deviation")
    axes[1].set_title("Sampling variability")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figure4_root_bias_sd_vs_n.png"), dpi=220)
    plt.close(fig)

    kbo_methods = ["KBO fixed lambda", "KBO decaying lambda"]
    has_decomp = any(
        any(
            str(r["method"]) == method and "regularization_bias_abs" in r
            for r in summary
        )
        for method in kbo_methods
    )
    if has_decomp:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        for ax, method in zip(axes, kbo_methods):
            xs, total = series(method, "rmse_to_true")
            _, _, total_se = series_err(method, "rmse_to_true", "rmse_to_true_se")
            _, reg, reg_se = series_err(method, "regularization_bias_abs", "regularization_bias_abs_se")
            _, est, est_se = series_err(method, "rmse_to_population_root", "rmse_to_population_root_se")
            if xs.size:
                ax.errorbar(
                    xs,
                    np.maximum(total, eps),
                    yerr=1.96 * total_se,
                    marker="o",
                    linewidth=2.0,
                    capsize=3,
                    label="total RMSE to true root",
                )
            if reg.size:
                ax.errorbar(
                    xs,
                    np.maximum(reg, eps),
                    yerr=1.96 * reg_se,
                    marker="s",
                    linewidth=2.0,
                    capsize=3,
                    label="regularization bias",
                )
            if est.size:
                ax.errorbar(
                    xs,
                    np.maximum(est, eps),
                    yerr=1.96 * est_se,
                    marker="^",
                    linewidth=2.0,
                    capsize=3,
                    label="estimation RMSE to KBO target",
                )
            ax.set_xlabel("sample size n")
            ax.set_ylabel("root error")
            ax.set_title(method)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "figure4_kbo_decomposition_vs_n.png"), dpi=220)
        plt.close(fig)


def run_experiment(
    dgp_cfg,
    learner_cfg,
    exp_cfg,
):
    os.makedirs(exp_cfg.outdir, exist_ok=True)

    print("Computing KBO population root proxies...", flush=True)
    pop_lams = all_population_lambdas(exp_cfg)
    pop_roots_by_lam = kbo_population_roots(dgp_cfg, exp_cfg, pop_lams)

    population_rows = []
    for lam in pop_lams:
        root = pop_roots_by_lam[float(lam)]
        population_rows.append(
            {
                "lambda": float(lam),
                "population_root": float(root),
                "regularization_bias_abs": float(abs(root - dgp_cfg.omega_star)),
            }
        )

    population_root_by_n_method = {}
    for n in exp_cfg.n_grid:
        lam_decay = float(
            exp_cfg.kbo_decay_c * (float(n) ** (-exp_cfg.kbo_decay_alpha))
        )
        population_root_by_n_method[(n, "KBO fixed lambda")] = pop_roots_by_lam[
            round(float(exp_cfg.kbo_fixed_lambda), 16)
        ]
        population_root_by_n_method[(n, "KBO decaying lambda")] = pop_roots_by_lam[
            round(lam_decay, 16)
        ]

    rng_master = np.random.default_rng(exp_cfg.seed)
    replications = []

    for n in exp_cfg.n_grid:
        lam_decay = float(
            exp_cfg.kbo_decay_c * (float(n) ** (-exp_cfg.kbo_decay_alpha))
        )
        kbo_lams = [float(exp_cfg.kbo_fixed_lambda), lam_decay]
        kbo_lams = sorted(set(round(x, 16) for x in kbo_lams))

        print(
            f"Running n={n}, reps={exp_cfg.reps}, "
            f"fixed_lambda={exp_cfg.kbo_fixed_lambda:.2e}, decay_lambda={lam_decay:.2e}",
            flush=True,
        )

        for rep in range(exp_cfg.reps):
            seed_main = int(rng_master.integers(0, np.iinfo(np.int32).max))
            seed_inner = int(rng_master.integers(0, np.iinfo(np.int32).max))
            seed_outer = int(rng_master.integers(0, np.iinfo(np.int32).max))
            seed_cf = int(rng_master.integers(0, np.iinfo(np.int32).max))

            main_data = simulate_data(n, dgp_cfg, np.random.default_rng(seed_main))
            inner = simulate_data(n, dgp_cfg, np.random.default_rng(seed_inner))
            outer = simulate_data(n, dgp_cfg, np.random.default_rng(seed_outer))

            roots = autodml_plugin_oracle_roots(
                data=main_data,
                dgp_cfg=dgp_cfg,
                learner_cfg=learner_cfg,
                k_folds=exp_cfg.k_folds,
                rng=np.random.default_rng(seed_cf),
            )

            for method in ["OBiGrad", "Plug-in", "Oracle DR", "Oracle plug-in"]:
                replications.append(
                    {
                        "rep": float(rep),
                        "n": float(n),
                        "method": method,
                        "root_hat": float(roots[method]),
                        "lambda": float("nan"),
                        "seed_main": float(seed_main),
                        "j_rmse": float(roots["j_rmse"]),
                        "m_rmse": float(roots["m_rmse"]),
                    }
                )

            kbo_roots = kbo_roots_from_split(
                inner, outer, kbo_lams, exp_cfg.kernel_sigma
            )
            root_fixed = kbo_roots[round(float(exp_cfg.kbo_fixed_lambda), 16)]
            root_decay = kbo_roots[round(lam_decay, 16)]

            for method, root_hat, lam_used in [
                ("KBO fixed lambda", root_fixed, float(exp_cfg.kbo_fixed_lambda)),
                ("KBO decaying lambda", root_decay, lam_decay),
            ]:
                row = {
                    "rep": float(rep),
                    "n": float(n),
                    "method": method,
                    "root_hat": float(root_hat),
                    "lambda": float(lam_used),
                    "seed_inner": float(seed_inner),
                    "seed_outer": float(seed_outer),
                }
                pop_root = population_root_by_n_method.get((n, method))
                if pop_root is not None:
                    row["population_root"] = float(pop_root)
                    row["error_to_population_root"] = float(root_hat - pop_root)
                replications.append(row)

            if (rep + 1) % max(1, exp_cfg.reps // 10) == 0:
                print(f"  n={n}: completed {rep + 1}/{exp_cfg.reps}", flush=True)

    summary = summarize(replications, dgp_cfg, population_root_by_n_method)
    make_plots(summary, exp_cfg.outdir)
    write_latex_outputs(summary, population_rows, exp_cfg.outdir)

    print("Summary:", flush=True)
    for row in summary:
        print(
            f"  n={int(float(row['n'])):5d}  {str(row['method']):24s}  "
            f"RMSE={float(row['rmse_to_true']):.4g}  "
            f"bias={float(row['bias_to_true']):+.4g}  "
            f"sd={float(row['sd_root']):.4g}",
            flush=True,
        )

    return population_rows, replications, summary


def write_latex_outputs(summary, population_rows, outdir):
    order = ["Plug-in", "OBiGrad", "Oracle DR", "KBO fixed lambda", "KBO decaying lambda"]
    rows = []
    for n in sorted({int(float(row["n"])) for row in summary}):
        by_method = {
            str(row["method"]): row
            for row in summary
            if int(float(row["n"])) == n
        }
        row = [str(n)]
        for method in order:
            item = by_method.get(method)
            row.append(
                format_pm(item["rmse_to_true"], item.get("rmse_to_true_se"))
                if item
                else "--"
            )
        rows.append(row)

    pop_rows = [
        [
            format_number(row["lambda"], digits=5),
            format_number(row["population_root"], digits=4),
            format_number(row["regularization_bias_abs"], digits=4),
        ]
        for row in population_rows
    ]

    write_latex_table(
        os.path.join(outdir, "table_iv_figure4_root_rmse.tex"),
        r"Root-estimation RMSE. Parentheses report Monte Carlo 95\% error bars.",
        "tab:generated-iv-root-rmse",
        [r"$n$"] + order,
        rows,
    )
    write_latex_table(
        os.path.join(outdir, "table_iv_figure4_kbo_population_roots.tex"),
        "KBO population roots in the scalar root experiment.",
        "tab:generated-iv-root-kbo-population",
        [r"$\lambda$", "Population root", "Bias"],
        pop_rows,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Figure 4 root-estimation experiment."
    )
    parser.add_argument("--n-grid", type=str, default="100,200,400,800,1600")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--outdir", type=str, default="results/IV/figure4")
    parser.add_argument("--k-folds", type=int, default=2)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--omega-star", type=float, default=2.0)
    parser.add_argument("--sigma-t", type=float, default=math.sqrt(0.10))
    parser.add_argument("--endog-strength", type=float, default=0.5)
    parser.add_argument("--sigma-y", type=float, default=0.10)

    parser.add_argument("--learner", choices=["linear", "rff"], default="linear")
    parser.add_argument("--ridge-alpha", type=float, default=1e-8)
    parser.add_argument("--rff-dim", type=int, default=512)
    parser.add_argument("--rff-sigma", type=float, default=2.0)

    parser.add_argument("--kernel-sigma", type=float, default=2.0)
    parser.add_argument("--kbo-fixed-lambda", type=float, default=1e-2)
    parser.add_argument("--kbo-decay-c", type=float, default=0.05)
    parser.add_argument("--kbo-decay-alpha", type=float, default=0.60)
    parser.add_argument("--pop-n", type=int, default=2500)

    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be positive.")
    if args.k_folds < 2:
        raise ValueError("--k-folds must be at least 2.")
    if args.pop_n < 100:
        raise ValueError("--pop-n must be at least 100.")
    if args.kbo_fixed_lambda <= 0:
        raise ValueError("--kbo-fixed-lambda must be positive.")

    dgp_cfg = DGPConfig(
        p=args.p,
        sigma_t=args.sigma_t,
        endog_strength=args.endog_strength,
        sigma_y=args.sigma_y,
        omega_star=args.omega_star,
    )
    learner_cfg = LearnerConfig(
        learner=args.learner,
        ridge_alpha=args.ridge_alpha,
        rff_dim=args.rff_dim,
        rff_sigma=args.rff_sigma,
    )
    exp_cfg = ExperimentConfig(
        n_grid=parse_int_grid(args.n_grid),
        reps=args.reps,
        seed=args.seed,
        k_folds=args.k_folds,
        outdir=args.outdir,
        kernel_sigma=args.kernel_sigma,
        kbo_fixed_lambda=args.kbo_fixed_lambda,
        kbo_decay_c=args.kbo_decay_c,
        kbo_decay_alpha=args.kbo_decay_alpha,
        pop_n=args.pop_n,
    )
    run_experiment(dgp_cfg, learner_cfg, exp_cfg)


if __name__ == "__main__":
    main()
