#!/usr/bin/env python3
"""
Figure 5 experiment: outer optimization with the OBiGrad gradient estimator.

This experiment uses the scalar IV data-generating process from
``IV_figure4_root_estimation_experiment.py``:

    X ~ N(0, I_p),
    T = 2 sum_j X_j + eta,
    Y = omega_star * T + rho eta + eps_y.

For the unregularized population target,

    Psi_0(omega) = 4 p (omega - omega_star),

so gradient descent should converge to ``omega_star``. The script learns

    j*(X) = E[T | X],    m*(X) = E[Y | X]

by cross-fitting and evaluates the scalar OBiGrad score at every outer iterate:

    phi_omega(O; h, j, m)
      = (omega T - Y) j(X)
        + (T - j(X)) (omega j(X) - m(X)).

The nuisance predictions do not depend on ``omega``, so one set of cross-fitted
predictions defines a stable empirical outer objective throughout optimization.

Outputs
-------
    figure5_optimization_trace.csv
    figure5_omega_trajectory.png
    figure5_objective_gap.png

Example
-------
    python IV_figure5_obigrad_optimization_experiment.py \
      --n 800 \
      --omega-init -1.0 \
      --steps 40 \
      --step-size 0.05 \
      --outdir results/IV/figure5
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    n: int = 800
    seed: int = 20260425
    k_folds: int = 2
    omega_init: float = -1.0
    iterations: int = 40
    step_size: float = 0.05
    outdir: str = "results/IV/figure5"


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


def true_objective_gap(omega, cfg):
    return 2.0 * cfg.p * (omega - cfg.omega_star) ** 2


class BaseRegressor:
    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class LinearRidgeRegressor(BaseRegressor):
    """Multi-output ridge regression on ``[1, X]`` as in Figure 4."""

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


def cross_fitted_nuisance_predictions(data, dgp_cfg, learner_cfg, k_folds, rng):
    """Cross-fit predictions of ``j*(X)`` and ``m*(X)`` on one dataset."""

    x = data["X"]
    t = data["T"]
    y = data["Y"]
    n = x.shape[0]
    j_hat = np.empty(n, dtype=float)
    m_hat = np.empty(n, dtype=float)

    for val_idx in make_folds(n, k_folds, rng):
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.nonzero(train_mask)[0]
        targets = np.column_stack([t[train_idx], y[train_idx]])
        fold_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int32).max)))
        reg = make_regressor(learner_cfg, fold_rng).fit(x[train_idx], targets)
        pred = reg.predict(x[val_idx])
        j_hat[val_idx] = pred[:, 0]
        m_hat[val_idx] = pred[:, 1]

    diagnostics = {
        "j_rmse": math.sqrt(float(np.mean((j_hat - true_j(x)) ** 2))),
        "m_rmse": math.sqrt(float(np.mean((m_hat - true_m(x, dgp_cfg)) ** 2))),
    }
    return j_hat, m_hat, diagnostics


def obigrad_gradient(omega, data, j_hat, m_hat):
    """Evaluate the cross-fitted scalar OBiGrad gradient at ``omega``."""

    t = data["T"]
    y = data["Y"]
    h_hat = omega * j_hat
    scores = (omega * t - y) * j_hat + (t - j_hat) * (h_hat - m_hat)
    return float(np.mean(scores))


def optimize_with_obigrad(data, dgp_cfg, learner_cfg, exp_cfg):
    rng = np.random.default_rng(exp_cfg.seed + 1)
    j_hat, m_hat, diagnostics = cross_fitted_nuisance_predictions(
        data, dgp_cfg, learner_cfg, exp_cfg.k_folds, rng
    )

    omega = float(exp_cfg.omega_init)
    trace = []
    for iteration in range(exp_cfg.iterations + 1):
        grad_hat = obigrad_gradient(omega, data, j_hat, m_hat)
        trace.append(
            {
                "iteration": iteration,
                "omega": omega,
                "gradient_hat": grad_hat,
                "true_gradient": true_gradient(omega, dgp_cfg),
                "objective_gap": true_objective_gap(omega, dgp_cfg),
                "distance_to_omega_star": abs(omega - dgp_cfg.omega_star),
            }
        )
        if iteration < exp_cfg.iterations:
            omega = omega - exp_cfg.step_size * grad_hat

    return trace, diagnostics


def write_trace_csv(trace, outdir):
    path = os.path.join(outdir, "figure5_optimization_trace.csv")
    columns = [
        "iteration",
        "omega",
        "gradient_hat",
        "true_gradient",
        "objective_gap",
        "distance_to_omega_star",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(trace)
    return path


def make_plots(trace, dgp_cfg, outdir):
    iterations = np.asarray([row["iteration"] for row in trace], dtype=float)
    omega = np.asarray([row["omega"] for row in trace], dtype=float)
    gap = np.asarray([row["objective_gap"] for row in trace], dtype=float)

    plt.figure(figsize=(7.8, 5.2))
    plt.plot(iterations, omega, marker="o", linewidth=2.0, label="OBiGrad GD")
    plt.axhline(
        dgp_cfg.omega_star,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=r"$\omega^\star$",
    )
    plt.xlabel("outer iteration")
    plt.ylabel(r"$\omega_t$")
    plt.title("Outer optimization trajectory with OBiGrad")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure5_omega_trajectory.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7.8, 5.2))
    plt.semilogy(iterations, np.maximum(gap, 1e-16), marker="o", linewidth=2.0)
    plt.xlabel("outer iteration")
    plt.ylabel(r"$F(\omega_t) - F(\omega^\star)$")
    plt.title("Population objective gap along OBiGrad updates")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure5_objective_gap.png"), dpi=220)
    plt.close()


def run_experiment(dgp_cfg, learner_cfg, exp_cfg):
    os.makedirs(exp_cfg.outdir, exist_ok=True)
    data = simulate_data(exp_cfg.n, dgp_cfg, np.random.default_rng(exp_cfg.seed))
    trace, diagnostics = optimize_with_obigrad(data, dgp_cfg, learner_cfg, exp_cfg)

    write_trace_csv(trace, exp_cfg.outdir)
    make_plots(trace, dgp_cfg, exp_cfg.outdir)

    final = trace[-1]
    print(
        f"Final omega: {final['omega']:.8f}; "
        f"distance to omega_star={final['distance_to_omega_star']:.8f}"
    )
    print(
        f"Cross-fitted nuisance RMSE: "
        f"j={diagnostics['j_rmse']:.6g}, m={diagnostics['m_rmse']:.6g}"
    )
    return trace, diagnostics


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Figure 5 OBiGrad outer-optimization experiment."
    )
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--outdir", type=str, default="results/IV/figure5")
    parser.add_argument("--k-folds", type=int, default=2)
    parser.add_argument("--omega-init", type=float, default=-1.0)
    parser.add_argument(
        "--steps",
        "--iterations",
        dest="iterations",
        type=int,
        default=40,
        help="Number of outer gradient-descent updates.",
    )
    parser.add_argument("--step-size", type=float, default=0.05)

    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--omega-star", type=float, default=2.0)
    parser.add_argument("--sigma-t", type=float, default=math.sqrt(0.10))
    parser.add_argument("--endog-strength", type=float, default=0.5)
    parser.add_argument("--sigma-y", type=float, default=0.10)

    parser.add_argument("--learner", choices=["linear", "rff"], default="linear")
    parser.add_argument("--ridge-alpha", type=float, default=1e-8)
    parser.add_argument("--rff-dim", type=int, default=512)
    parser.add_argument("--rff-sigma", type=float, default=2.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.n < 20:
        raise ValueError("--n must be at least 20.")
    if args.k_folds < 2:
        raise ValueError("--k-folds must be at least 2.")
    if args.k_folds > args.n:
        raise ValueError("--k-folds cannot exceed --n.")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive.")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive.")

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
        n=args.n,
        seed=args.seed,
        k_folds=args.k_folds,
        omega_init=args.omega_init,
        iterations=args.iterations,
        step_size=args.step_size,
        outdir=args.outdir,
    )
    run_experiment(dgp_cfg, learner_cfg, exp_cfg)


if __name__ == "__main__":
    main()
