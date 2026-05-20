#!/usr/bin/env python3
"""
Figure 3 experiment: KBO regularization bias versus the unregularized
semiparametric gradient target.

This script is self-contained. It adapts the kernel-ridge KBO computation used in
KBO-main/iv_regression.py from the uploaded KBO reproduction code:

    C_hat = K(X_outer, X_inner) @ (K(X_inner, X_inner) + n * lambda * I)^{-1} Phi(T_inner)
    grad  = C_hat.T @ (C_hat @ omega - Y_outer) / m

The experimental target, however, is the unregularized semiparametric gradient
from the note:

    Psi_0(omega) = grad_omega 0.5 E[(Y - h^*_omega(X))^2],
    h^*_omega(X) = E[g_omega(T) | X],
    g_omega(T) = omega^T phi(T),
    phi_l(T) = sin(T + l).

For the easy Gaussian sine-IV DGP, Psi_0(omega) is analytic. This lets us split
KBO's error to the unregularized target into:

    1. regularization bias: || Psi_lambda^KBO(omega) - Psi_0(omega) ||,
    2. fixed-lambda estimation error: || Psi_hat_lambda^KBO(omega)
                                      - Psi_lambda^KBO(omega) ||,
    3. total error to the semiparametric target: || Psi_hat_lambda^KBO(omega)
                                                 - Psi_0(omega) ||.

The script also computes a sample-split OBiGrad/orthogonal DR gradient estimator
for Psi_0(omega) as a reference horizontal line in the plot.

Outputs
-------
    figure3_kbo_regularization_bias.png
    figure3_kbo_bias_decomposition.png
    table_iv_figure3_kbo.tex
    table_iv_figure3_kbo_appendix.tex

Paper-level example
-------------------
    python IV_figure3_kbo_regularization_experiment.py \
        --n 600 \
        --pop-n 3000 \
        --reps 300 \
        --lambdas 1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1 \
        --outdir results/IV/figure3

Fast smoke test
---------------
    python IV_figure3_kbo_regularization_experiment.py \
        --n 200 --pop-n 800 --reps 20 \
        --lambdas 1e-4,3e-4,1e-3,3e-3,1e-2 \
        --outdir results/IV/figure3_smoke
"""


import argparse
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_reporting import format_lambda, format_number, format_pm, write_latex_table


@dataclass(frozen=True)
class DGPConfig:
    """Easy sine-IV DGP.

    X ~ N(0, I_p)
    T = 2 sum_j X_j + eta, eta ~ N(0, sigma_t^2)
    Y = omega_star^T phi(T) + endog_strength * eta + eps_y

    The endogeneity term is mean-zero conditional on X, so the conditional mean
    m^*(X) remains E[Y|X] = j^*(X)^T omega_star.
    """

    p: int = 3
    d: int = 4
    sigma_t: float = math.sqrt(0.10)
    endog_strength: float = 0.5
    sigma_y: float = 0.0
    omega_star_scale: float = 1.0

    @property
    def omega_star(self):
        base = np.arange(1, self.d + 1, dtype=float)
        return self.omega_star_scale * base / np.linalg.norm(base)


@dataclass(frozen=True)
class LearnerConfig:
    """OBiGrad nuisance learner configuration.

    Default ``sum_fourier`` matches the paper experiment: it uses observable
    Fourier features of sum(X), which is stable for the sine-IV benchmark.
    Set ``--learner rff`` for a more generic random-feature stress test.
    """

    kind: str = "sum_fourier"
    rff_dim: int = 512
    gamma: float = 2.0
    ridge_alpha: float = 1e-6
    fourier_max_freq: int = 8
    fit_intercept: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 400
    pop_n: int = 2000
    reps: int = 200
    seed: int = 12345
    kernel_sigma: float = 0.5
    omega_eval_shift: float = 0.35
    lambdas: Tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    outdir: str = "results/IV/figure3"


def parse_float_grid(text):
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("Grid must contain at least one number")
    if any(v <= 0 for v in values):
        raise ValueError("All grid values must be positive")
    return values


def phi_features(t, d):
    """phi_l(t)=sin(t+l), l=1,...,d. Returns shape (n,d)."""
    ell = np.arange(1, d + 1, dtype=float)
    return np.sin(t[:, None] + ell[None, :])


def true_conditional_phi(x, cfg):
    """j^*(X)=E[phi(T)|X] for T=2 sum(X)+eta, Gaussian eta."""
    ell = np.arange(1, cfg.d + 1, dtype=float)
    attenuation = math.exp(-0.5 * cfg.sigma_t**2)
    return attenuation * np.sin(2.0 * x.sum(axis=1)[:, None] + ell[None, :])


def true_A_matrix(cfg):
    """Analytic A=E[j^*(X)j^*(X)^T].

    S=sum_j X_j ~ N(0,p), a=2S, Var(a)=4p. With
    j_l(X)=c sin(a+l), c=exp(-sigma_t^2/2),

        E[j_l j_k] = c^2/2 { cos(l-k) - exp(-8p) cos(l+k) }.
    """
    ell = np.arange(1, cfg.d + 1, dtype=float)
    l_minus_k = ell[:, None] - ell[None, :]
    l_plus_k = ell[:, None] + ell[None, :]
    c2 = math.exp(-cfg.sigma_t**2)
    return 0.5 * c2 * (np.cos(l_minus_k) - math.exp(-8.0 * cfg.p) * np.cos(l_plus_k))


def true_gradient(omega, cfg):
    """Unregularized semiparametric target Psi_0(omega)."""
    return true_A_matrix(cfg) @ (omega - cfg.omega_star)


def simulate_data(n, cfg, rng):
    x = rng.normal(size=(n, cfg.p))
    eta = rng.normal(scale=cfg.sigma_t, size=n)
    t = 2.0 * x.sum(axis=1) + eta
    phi_t = phi_features(t, cfg.d)
    eps_y = rng.normal(scale=cfg.sigma_y, size=n) if cfg.sigma_y > 0 else np.zeros(n)
    y = phi_t @ cfg.omega_star + cfg.endog_strength * eta + eps_y
    return {"X": x, "T": t, "Phi": phi_t, "Y": y, "eta": eta}


def gaussian_kernel_gram(x, y, sigma):
    """Gaussian kernel matrix exp(-||x-y||^2/(2 sigma^2))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    sqdist = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.exp(-sqdist / (2.0 * sigma**2))


def kernel_ridge_predict_train_outer(
    x_inner,
    y_inner,
    x_outer,
    lam,
    kernel_sigma,
    precomputed=None,
):
    """Predict E[y_inner|X] at x_outer by KRR in the KBO parameterization.

    The KBO code regularizes with K + n * lambda * I. This function follows the
    same convention.
    """
    n = x_inner.shape[0]
    if precomputed is None:
        k11 = gaussian_kernel_gram(x_inner, x_inner, kernel_sigma)
        k21 = gaussian_kernel_gram(x_outer, x_inner, kernel_sigma)
    else:
        k11, k21 = precomputed
    mat = k11 + n * lam * np.eye(n)
    try:
        alpha = np.linalg.solve(mat, y_inner)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(mat, y_inner, rcond=None)[0]
    return k21 @ alpha


def kbo_gradient_from_split(
    inner,
    outer,
    omega_eval,
    lam,
    kernel_sigma,
    precomputed=None,
):
    """KBO plug-in gradient for a fixed lambda.

    This is the NumPy equivalent of IVRegression.Kbar_C_hat() followed by
    grad value(omega) in the uploaded KBO implementation.
    """
    c_hat = kernel_ridge_predict_train_outer(
        x_inner=inner["X"],
        y_inner=inner["Phi"],
        x_outer=outer["X"],
        lam=lam,
        kernel_sigma=kernel_sigma,
        precomputed=precomputed,
    )
    residual = c_hat @ omega_eval - outer["Y"]
    return c_hat.T @ residual / outer["X"].shape[0]


def kbo_population_proxy(
    cfg,
    exp_cfg,
    omega_eval,
):
    """Approximate the fixed-lambda KBO population gradient.

    To isolate regularization bias, the population proxy uses the denoised
    population regression targets j^*(X_inner) and m^*(X_outer). This approximates
    the regularized population regression problem rather than adding avoidable
    Monte Carlo response noise to the target itself.
    """
    rng = np.random.default_rng(exp_cfg.seed + 987654)
    n_inner = exp_cfg.pop_n // 2
    n_outer = exp_cfg.pop_n - n_inner
    x_inner = rng.normal(size=(n_inner, cfg.p))
    x_outer = rng.normal(size=(n_outer, cfg.p))

    j_inner = true_conditional_phi(x_inner, cfg)
    m_outer = true_conditional_phi(x_outer, cfg) @ cfg.omega_star

    k11 = gaussian_kernel_gram(x_inner, x_inner, exp_cfg.kernel_sigma)
    k21 = gaussian_kernel_gram(x_outer, x_inner, exp_cfg.kernel_sigma)

    rows = []
    grad_by_lam = {}
    psi0 = true_gradient(omega_eval, cfg)

    for lam in exp_cfg.lambdas:
        c_lam = kernel_ridge_predict_train_outer(
            x_inner=x_inner,
            y_inner=j_inner,
            x_outer=x_outer,
            lam=lam,
            kernel_sigma=exp_cfg.kernel_sigma,
            precomputed=(k11, k21),
        )
        grad_lam = c_lam.T @ (c_lam @ omega_eval - m_outer) / n_outer
        grad_by_lam[lam] = grad_lam
        row = {
            "lambda": float(lam),
            "regularization_bias_l2": float(np.linalg.norm(grad_lam - psi0)),
        }
        for k in range(cfg.d):
            row[f"pop_grad_lambda_{k}"] = float(grad_lam[k])
            row[f"psi0_{k}"] = float(psi0[k])
        rows.append(row)

    return rows, grad_by_lam


class SumFourierRidgeRegressor:
    """Ridge regression on Fourier features of S=sum_j X_j."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.s_mean_ = None
        self.coef_ = None

    def _features(self, x, fit):
        s = np.asarray(x, dtype=float).sum(axis=1)
        if fit:
            self.s_mean_ = float(s.mean())
        if self.s_mean_ is None:
            raise RuntimeError("Regressor is not fitted")
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
            raise RuntimeError("Regressor is not fitted")
        pred = self._features(x, fit=False) @ self.coef_
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred


class RFFRidgeRegressor:
    """Multi-output ridge regression on Gaussian random Fourier features."""

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
            raise RuntimeError("Regressor is not fitted")
        return (x - self.x_mean_) / self.x_scale_

    def _features(self, x, fit):
        if fit:
            xs = self._standardize_fit(np.asarray(x, dtype=float))
            rng = np.random.default_rng(self.seed)
            self.W_ = rng.normal(
                scale=math.sqrt(2.0 * self.cfg.gamma),
                size=(xs.shape[1], self.cfg.rff_dim),
            )
            self.b_ = rng.uniform(0.0, 2.0 * math.pi, size=self.cfg.rff_dim)
        else:
            xs = self._standardize_transform(np.asarray(x, dtype=float))
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Regressor is not fitted")
        z = math.sqrt(2.0 / self.cfg.rff_dim) * np.cos(xs @ self.W_ + self.b_)
        if self.cfg.fit_intercept:
            z = np.column_stack([np.ones(xs.shape[0]), z])
        return z

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
            raise RuntimeError("Regressor is not fitted")
        pred = self._features(x, fit=False) @ self.coef_
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred


def make_regressor(cfg, seed):
    if cfg.kind == "sum_fourier":
        return SumFourierRidgeRegressor(cfg)
    if cfg.kind == "rff":
        return RFFRidgeRegressor(cfg, seed=seed)
    raise ValueError(f"Unknown learner kind: {cfg.kind!r}")


def autodml_gradient_from_split(
    inner,
    outer,
    omega_eval,
    cfg,
    learner_cfg,
    seed,
):
    """Sample-split orthogonal DR gradient and oracle DR gradient.

    Train h, j, m on the inner sample and evaluate the orthogonal score on the
    outer sample. This is deliberately the same split structure as KBO.
    """
    g_inner = inner["Phi"] @ omega_eval
    targets = np.column_stack([g_inner, inner["Phi"], inner["Y"]])
    learner = make_regressor(learner_cfg, seed=seed).fit(inner["X"], targets)
    pred = learner.predict(outer["X"])

    h_hat = pred[:, 0]
    j_hat = pred[:, 1 : 1 + cfg.d]
    m_hat = pred[:, 1 + cfg.d]

    g_outer = outer["Phi"] @ omega_eval
    score = (
        j_hat * (g_outer - outer["Y"])[:, None]
        + (outer["Phi"] - j_hat) * (h_hat - m_hat)[:, None]
    )
    dr_grad = score.mean(axis=0)

    j_true = true_conditional_phi(outer["X"], cfg)
    h_true = j_true @ omega_eval
    m_true = j_true @ cfg.omega_star
    oracle_score = (
        j_true * (g_outer - outer["Y"])[:, None]
        + (outer["Phi"] - j_true) * (h_true - m_true)[:, None]
    )
    oracle_grad = oracle_score.mean(axis=0)

    return dr_grad, oracle_grad


def summarize(
    replications,
    population_rows,
    lambdas,
):
    pop_by_lam = {float(r["lambda"]): r for r in population_rows}
    summary = []
    for lam in lambdas:
        lam_f = float(lam)
        rows = [r for r in replications if abs(r["lambda"] - lam_f) <= 1e-18]
        if not rows:
            continue

        def rmse(name):
            vals = np.array([r[name] for r in rows], dtype=float)
            out = math.sqrt(float(np.mean(vals**2)))
            if len(vals) <= 1 or out <= 0:
                return out, 0.0
            se_mse = float(np.std(vals**2, ddof=1) / math.sqrt(len(vals)))
            return out, se_mse / (2.0 * out)

        kbo_est, kbo_est_se = rmse("kbo_error_to_lambda_l2")
        kbo_total, kbo_total_se = rmse("kbo_error_to_psi0_l2")
        autodml, autodml_se = rmse("autodml_error_to_psi0_l2")
        oracle, oracle_se = rmse("oracle_dr_error_to_psi0_l2")

        out = {
            "lambda": lam_f,
            "regularization_bias_l2": float(
                pop_by_lam[lam_f]["regularization_bias_l2"]
            ),
            "regularization_bias_l2_se": 0.0,
            "kbo_estimation_rmse_l2": kbo_est,
            "kbo_estimation_rmse_se": kbo_est_se,
            "kbo_total_rmse_l2": kbo_total,
            "kbo_total_rmse_se": kbo_total_se,
            "autodml_rmse_l2": autodml,
            "autodml_rmse_se": autodml_se,
            "oracle_dr_rmse_l2": oracle,
            "oracle_dr_rmse_se": oracle_se,
            "kbo_mean_error_to_lambda_l2": float(
                np.mean([r["kbo_error_to_lambda_l2"] for r in rows])
            ),
            "kbo_mean_error_to_psi0_l2": float(
                np.mean([r["kbo_error_to_psi0_l2"] for r in rows])
            ),
            "autodml_mean_error_to_psi0_l2": float(
                np.mean([r["autodml_error_to_psi0_l2"] for r in rows])
            ),
        }
        summary.append(out)
    return summary


def make_plots(summary, outdir):
    lambdas = np.array([r["lambda"] for r in summary], dtype=float)
    reg_bias = np.array([r["regularization_bias_l2"] for r in summary], dtype=float)
    reg_bias_se = np.array([r.get("regularization_bias_l2_se", 0.0) for r in summary], dtype=float)
    kbo_est = np.array([r["kbo_estimation_rmse_l2"] for r in summary], dtype=float)
    kbo_est_se = np.array([r.get("kbo_estimation_rmse_se", 0.0) for r in summary], dtype=float)
    kbo_total = np.array([r["kbo_total_rmse_l2"] for r in summary], dtype=float)
    kbo_total_se = np.array([r.get("kbo_total_rmse_se", 0.0) for r in summary], dtype=float)
    autodml = np.array([r["autodml_rmse_l2"] for r in summary], dtype=float)
    autodml_se = np.array([r.get("autodml_rmse_se", 0.0) for r in summary], dtype=float)
    oracle = np.array([r["oracle_dr_rmse_l2"] for r in summary], dtype=float)
    oracle_se = np.array([r.get("oracle_dr_rmse_se", 0.0) for r in summary], dtype=float)

    eps = 1e-12

    plt.figure(figsize=(7.4, 5.2))
    plt.errorbar(
        lambdas,
        np.maximum(kbo_total, eps),
        yerr=1.96 * kbo_total_se,
        marker="o",
        color="#1f4e79",
        linewidth=2.0,
        capsize=3,
        label=r"KBO total RMSE to $\nabla F_0(\omega)$",
    )
    plt.fill_between(
        lambdas,
        np.maximum(reg_bias - 1.96 * reg_bias_se, eps),
        np.maximum(reg_bias + 1.96 * reg_bias_se, eps),
        color="#d98c27",
        alpha=0.20,
        label=r"KBO component: regularization bias",
    )
    plt.plot(
        lambdas,
        np.maximum(reg_bias, eps),
        color="#d98c27",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
    )
    plt.fill_between(
        lambdas,
        np.maximum(kbo_est - 1.96 * kbo_est_se, eps),
        np.maximum(kbo_est + 1.96 * kbo_est_se, eps),
        color="#7b8fa1",
        alpha=0.22,
        label=r"KBO component: estimation error",
    )
    plt.plot(
        lambdas,
        np.maximum(kbo_est, eps),
        color="#7b8fa1",
        linestyle=":",
        linewidth=1.7,
        alpha=0.9,
    )
    plt.errorbar(
        lambdas,
        np.maximum(autodml, eps),
        yerr=1.96 * autodml_se,
        marker="D",
        color="#2c7a4b",
        linestyle="--",
        linewidth=1.8,
        capsize=3,
        label="OBiGrad RMSE to unregularized target",
    )
    plt.errorbar(
        lambdas,
        np.maximum(oracle, eps),
        yerr=1.96 * oracle_se,
        marker="P",
        color="#5f4b8b",
        linestyle=":",
        linewidth=1.8,
        capsize=3,
        label="Oracle DR RMSE",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"KBO regularization $\lambda$")
    plt.ylabel(r"gradient error, Euclidean norm")
    plt.title("KBO fixed-regularization bias versus unregularized target")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure3_kbo_regularization_bias.png"), dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].errorbar(
        lambdas,
        np.maximum(kbo_total, eps),
        yerr=1.96 * kbo_total_se,
        marker="o",
        color="#1f4e79",
        linewidth=2.0,
        capsize=3,
        label="KBO total",
    )
    axes[0].fill_between(
        lambdas,
        np.maximum(reg_bias - 1.96 * reg_bias_se, eps),
        np.maximum(reg_bias + 1.96 * reg_bias_se, eps),
        color="#d98c27",
        alpha=0.20,
        label="component: regularization bias",
    )
    axes[0].plot(
        lambdas,
        np.maximum(reg_bias, eps),
        color="#d98c27",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
    )
    axes[0].fill_between(
        lambdas,
        np.maximum(kbo_est - 1.96 * kbo_est_se, eps),
        np.maximum(kbo_est + 1.96 * kbo_est_se, eps),
        color="#7b8fa1",
        alpha=0.22,
        label="component: estimation error",
    )
    axes[0].plot(
        lambdas,
        np.maximum(kbo_est, eps),
        color="#7b8fa1",
        linestyle=":",
        linewidth=1.7,
        alpha=0.9,
    )
    axes[0].errorbar(
        lambdas,
        np.maximum(autodml, eps),
        yerr=1.96 * autodml_se,
        marker="D",
        color="#2c7a4b",
        linestyle="--",
        linewidth=1.6,
        capsize=3,
        label="OBiGrad",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("gradient error")
    axes[0].set_title("Bias decomposition")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=9)

    ratio_bias = reg_bias / np.maximum(kbo_total, eps)
    ratio_est = kbo_est / np.maximum(kbo_total, eps)
    axes[1].errorbar(
        lambdas,
        ratio_bias,
        yerr=1.96 * reg_bias_se / np.maximum(kbo_total, eps),
        marker="s",
        linewidth=2.0,
        capsize=3,
        label="bias / total RMSE",
    )
    axes[1].errorbar(
        lambdas,
        ratio_est,
        yerr=1.96 * kbo_est_se / np.maximum(kbo_total, eps),
        marker="^",
        linewidth=2.0,
        capsize=3,
        label="estimation / total RMSE",
    )
    axes[1].set_xscale("log")
    axes[1].axhline(1.0, linestyle=":", linewidth=1.2)
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("relative size")
    axes[1].set_title("Which term dominates?")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figure3_kbo_bias_decomposition.png"), dpi=220)
    plt.close(fig)


def run_experiment(
    cfg,
    learner_cfg,
    exp_cfg,
):
    os.makedirs(exp_cfg.outdir, exist_ok=True)

    omega_star = cfg.omega_star
    direction = np.linspace(1.0, -1.0, cfg.d)
    direction = direction / np.linalg.norm(direction)
    omega_eval = omega_star + exp_cfg.omega_eval_shift * direction
    psi0 = true_gradient(omega_eval, cfg)

    print("Computing fixed-lambda KBO population proxy...", flush=True)
    population_rows, grad_lambda = kbo_population_proxy(cfg, exp_cfg, omega_eval)

    rng_master = np.random.default_rng(exp_cfg.seed)
    replications = []

    print(f"Running replications: n={exp_cfg.n}, reps={exp_cfg.reps}", flush=True)
    for rep in range(exp_cfg.reps):
        seed_inner = int(rng_master.integers(0, np.iinfo(np.int32).max))
        seed_outer = int(rng_master.integers(0, np.iinfo(np.int32).max))
        inner = simulate_data(exp_cfg.n, cfg, np.random.default_rng(seed_inner))
        outer = simulate_data(exp_cfg.n, cfg, np.random.default_rng(seed_outer))

        # Precompute the two kernel matrices used by all lambda values in this replication.
        k11 = gaussian_kernel_gram(inner["X"], inner["X"], exp_cfg.kernel_sigma)
        k21 = gaussian_kernel_gram(outer["X"], inner["X"], exp_cfg.kernel_sigma)

        autodml_grad, oracle_dr_grad = autodml_gradient_from_split(
            inner=inner,
            outer=outer,
            omega_eval=omega_eval,
            cfg=cfg,
            learner_cfg=learner_cfg,
            seed=seed_inner + 17,
        )
        autodml_err = float(np.linalg.norm(autodml_grad - psi0))
        oracle_err = float(np.linalg.norm(oracle_dr_grad - psi0))

        for lam in exp_cfg.lambdas:
            kbo_grad = kbo_gradient_from_split(
                inner=inner,
                outer=outer,
                omega_eval=omega_eval,
                lam=lam,
                kernel_sigma=exp_cfg.kernel_sigma,
                precomputed=(k11, k21),
            )
            err_to_lambda = kbo_grad - grad_lambda[lam]
            err_to_psi0 = kbo_grad - psi0
            row = {
                "rep": float(rep),
                "n": float(exp_cfg.n),
                "lambda": float(lam),
                "seed_inner": float(seed_inner),
                "seed_outer": float(seed_outer),
                "kbo_error_to_lambda_l2": float(np.linalg.norm(err_to_lambda)),
                "kbo_error_to_psi0_l2": float(np.linalg.norm(err_to_psi0)),
                "autodml_error_to_psi0_l2": autodml_err,
                "oracle_dr_error_to_psi0_l2": oracle_err,
            }
            for k in range(cfg.d):
                row[f"kbo_grad_{k}"] = float(kbo_grad[k])
                row[f"kbo_pop_grad_lambda_{k}"] = float(grad_lambda[lam][k])
                row[f"autodml_grad_{k}"] = float(autodml_grad[k])
                row[f"oracle_dr_grad_{k}"] = float(oracle_dr_grad[k])
                row[f"psi0_{k}"] = float(psi0[k])
            replications.append(row)

        if (rep + 1) % max(1, exp_cfg.reps // 10) == 0:
            print(f"  completed {rep + 1}/{exp_cfg.reps}", flush=True)

    summary = summarize(replications, population_rows, exp_cfg.lambdas)
    make_plots(summary, exp_cfg.outdir)
    write_latex_outputs(summary, exp_cfg.outdir)

    print("Summary:", flush=True)
    for row in summary:
        print(
            f"  lambda={row['lambda']:.1e}  "
            f"KBO_total={row['kbo_total_rmse_l2']:.4g}  "
            f"reg_bias={row['regularization_bias_l2']:.4g}  "
            f"KBO_est={row['kbo_estimation_rmse_l2']:.4g}  "
            f"OBiGrad={row['autodml_rmse_l2']:.4g}",
            flush=True,
        )

    return population_rows, replications, summary


def write_latex_outputs(summary, outdir):
    rows = []
    appendix_rows = []
    for row in summary:
        rows.append(
            [
                format_lambda(row["lambda"]),
                format_pm(row["kbo_total_rmse_l2"], row.get("kbo_total_rmse_se")),
                format_number(row["regularization_bias_l2"], digits=4),
                format_pm(row["kbo_estimation_rmse_l2"], row.get("kbo_estimation_rmse_se")),
                format_pm(row["autodml_rmse_l2"], row.get("autodml_rmse_se")),
            ]
        )
        appendix_rows.append(
            [
                format_lambda(row["lambda"]),
                format_number(row["kbo_mean_error_to_psi0_l2"], digits=4),
                format_number(row["kbo_mean_error_to_lambda_l2"], digits=4),
                format_number(row["autodml_mean_error_to_psi0_l2"], digits=4),
                format_pm(row["oracle_dr_rmse_l2"], row.get("oracle_dr_rmse_se")),
            ]
        )

    write_latex_table(
        os.path.join(outdir, "table_iv_figure3_kbo.tex"),
        r"KBO gradient-error decomposition. Parentheses report Monte Carlo 95\% error bars for RMSE.",
        "tab:generated-iv-kbo-gradient",
        [r"$\lambda$", "KBO total", "Reg. bias", "KBO estimation", "OBiGrad"],
        rows,
    )
    write_latex_table(
        os.path.join(outdir, "table_iv_figure3_kbo_appendix.tex"),
        "Additional KBO and OBiGrad diagnostics for the IV regularization experiment.",
        "tab:generated-iv-kbo-gradient-appendix",
        [r"$\lambda$", "KBO mean total", "KBO mean estimation", "OBiGrad mean", "Oracle DR"],
        appendix_rows,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Figure 3 KBO regularization-bias experiment."
    )
    parser.add_argument(
        "--n", type=int, default=400, help="inner and outer sample size per replication"
    )
    parser.add_argument(
        "--pop-n", type=int, default=2000, help="population proxy total sample size"
    )
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=str, default="results/IV/figure3")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--sigma-t", type=float, default=math.sqrt(0.10))
    parser.add_argument("--endog-strength", type=float, default=0.5)
    parser.add_argument("--sigma-y", type=float, default=0.0)
    parser.add_argument("--kernel-sigma", type=float, default=0.5)
    parser.add_argument("--omega-eval-shift", type=float, default=0.35)
    parser.add_argument(
        "--learner", type=str, default="sum_fourier", choices=["sum_fourier", "rff"]
    )
    parser.add_argument("--rff-dim", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument(
        "--lambdas",
        type=str,
        default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--fourier-max-freq", type=int, default=8)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.n < 20:
        raise ValueError("--n should be at least 20")
    if args.pop_n < 100:
        raise ValueError("--pop-n should be at least 100")

    dgp_cfg = DGPConfig(
        p=args.p,
        d=args.d,
        sigma_t=args.sigma_t,
        endog_strength=args.endog_strength,
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
        n=args.n,
        pop_n=args.pop_n,
        reps=args.reps,
        seed=args.seed,
        kernel_sigma=args.kernel_sigma,
        omega_eval_shift=args.omega_eval_shift,
        lambdas=parse_float_grid(args.lambdas),
        outdir=args.outdir,
    )
    run_experiment(dgp_cfg, learner_cfg, exp_cfg)


if __name__ == "__main__":
    main()
