#!/usr/bin/env python3
"""
Outer gradient descent using OBiGrad in the projected Bellman experiment.

This script is an optimization companion to ``Bellman_b1_obigrad_experiment``.
It uses the same fitted-Q-evaluation DGP, four-dimensional value-function
parameter, nuisance learner, and cross-fitted orthogonal gradient estimator:

    g_omega(Z) = R + gamma * omega^T phi(S')
    F(omega) = 0.5 E[(Y - E[g_omega(Z) | S,A])^2]

The population optimizer is the known vector ``omega_star``. At each outer
iteration, the existing B1 OBiGrad estimator is evaluated at the current
``omega`` and gradient descent performs

    omega_{t+1} = omega_t - step_size * grad_hat_t.

The same sampled dataset and cross-fitting split seeds are used throughout the
trajectory, so the output tracks optimization on one empirical problem.

Outputs
-------
    bellman_obigrad_optimization_trace.csv
    bellman_obigrad_omega_trajectory.png
    bellman_obigrad_gradient_magnitude.png
    bellman_obigrad_bellman_error.png
    run_config.json

Example
-------
    python Bellman_obigrad_optimization_experiment.py \
      --n 500 \
      --steps 30 \
      --step-size 0.05 \
      --omega-init 0.0 \
      --seed 123 \
      --outdir results/Bellman/optimization_demo

Higher-accuracy run
-------------------
    python Bellman_obigrad_optimization_experiment.py \
      --n 50000 \
      --steps 1000 \
      --step-size 1.0 \
      --omega-init 0.0 \
      --seed 123 \
      --outdir results/Bellman/optimization_high_accuracy
"""

import argparse
import csv
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Bellman_b1_obigrad_experiment import (
    DGPConfig,
    LearnerConfig,
    MultiOutputRegressor,
    make_folds,
    simulate_data,
    subset,
    true_A_matrix,
    true_gradient,
)


@dataclass(frozen=True)
class OptimizationConfig:
    n: int = 500
    steps: int = 30
    step_size: float = 0.05
    omega_init: float = 0.0
    seed: int = 123
    folds: int = 2
    outdir: str = "results/Bellman/optimization_demo"
    quadrature_nodes: int = 200


def bellman_error(omega, dgp, a_matrix):
    """Return the population outer objective gap ``F(omega)-F(omega_star)``."""

    delta = np.asarray(omega, dtype=float) - np.asarray(dgp.omega_star, dtype=float)
    return 0.5 * float(delta @ a_matrix @ delta)


def vector_text(values):
    return "[" + ", ".join(f"{float(value):.10g}" for value in values) + "]"


def cross_fitted_score_terms(data, dgp, learner, cfg):
    """Build the affine B1 OBiGrad score ``s_i(omega)=A_i omega+c_i``.

    B1 fits ``(g_omega, dg, Y)`` at a fixed omega. Since its ridge learner is
    linear in its targets and ``g_omega=R+dg @ omega``, fitting
    ``(R, dg, Y)`` once produces the identical score for every outer iterate.

    The unexpanded doubly robust / orthogonal score is

        j_hat * (R + dg @ omega - Y)
        + (dg - j_hat) * (h0_hat + j_hat @ omega - m_hat).

    The second line is the orthogonal correction; a plug-in gradient would not
    contain it.
    """

    n = len(data["Y"])
    d = len(dgp.omega_star)
    slope = np.zeros((n, d, d), dtype=float)
    intercept = np.zeros((n, d), dtype=float)
    dg_all = dgp.gamma * data["PhiSp"]
    rng = np.random.default_rng(cfg.seed + 1009)
    fold_ids = make_folds(n, cfg.folds, rng)
    all_idx = np.arange(n)

    for fold_id, test_idx in enumerate(fold_ids):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        train = subset(data, train_idx)
        dg_train = dgp.gamma * train["PhiSp"]
        targets = np.column_stack([train["R"], dg_train, train["Y"]])
        model = MultiOutputRegressor(
            learner, dgp, seed=cfg.seed + 2003 + 7919 * (fold_id + 1)
        )
        model.fit(train["X"], targets)
        pred = model.predict(data["X"][test_idx])
        h0_hat = pred[:, 0]
        j_hat = pred[:, 1 : 1 + d]
        m_hat = pred[:, 1 + d]
        dg = dg_all[test_idx]

        # Affine expansion of the doubly robust score shown in the docstring.
        intercept[test_idx] = (
            j_hat * (data["R"][test_idx] - data["Y"][test_idx])[:, None]
            + (dg - j_hat) * (h0_hat - m_hat)[:, None]
        )
        slope[test_idx] = (
            np.einsum("ni,nj->nij", j_hat, dg)
            + np.einsum("ni,nj->nij", dg - j_hat, j_hat)
        )

    return slope, intercept


def estimate_obigrad_at_omega(omega, slope, intercept, dgp, cfg):
    """Evaluate the cross-fitted B1 OBiGrad score at one outer iterate."""

    scores = intercept + np.einsum("nij,j->ni", slope, omega)
    grad_hat = scores.mean(axis=0)
    cov = np.cov(scores, rowvar=False)
    grad_se = np.sqrt(np.maximum(np.diag(cov), 0.0) / scores.shape[0])
    grad_true = true_gradient(omega, dgp, nodes=cfg.quadrature_nodes)
    return grad_hat, grad_se, grad_true


def optimize(data, dgp, learner, cfg):
    omega_star = np.asarray(dgp.omega_star, dtype=float)
    omega = np.full(omega_star.shape, cfg.omega_init, dtype=float)
    a_matrix = true_A_matrix(dgp, nodes=cfg.quadrature_nodes)
    slope, intercept = cross_fitted_score_terms(data, dgp, learner, cfg)
    trace = []

    for step in range(cfg.steps + 1):
        grad_hat, grad_se, grad_true = estimate_obigrad_at_omega(
            omega, slope, intercept, dgp, cfg
        )
        distance = float(np.linalg.norm(omega - omega_star))
        row = {
            "step": step,
            "omega": vector_text(omega),
            "grad_hat": vector_text(grad_hat),
            "grad_se": vector_text(grad_se),
            "grad_norm": float(np.linalg.norm(grad_hat)),
            "true_grad_norm": float(np.linalg.norm(grad_true)),
            "bellman_error": bellman_error(omega, dgp, a_matrix),
            "distance_to_target": distance,
        }
        for k in range(len(omega)):
            row[f"omega_{k}"] = float(omega[k])
            row[f"grad_hat_{k}"] = float(grad_hat[k])
            row[f"grad_se_{k}"] = float(grad_se[k])
            row[f"omega_star_{k}"] = float(omega_star[k])
        trace.append(row)
        if step < cfg.steps:
            omega = omega - cfg.step_size * grad_hat

    return trace


def write_trace_csv(trace, outdir):
    path = outdir / "bellman_obigrad_optimization_trace.csv"
    fieldnames = list(trace[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)
    return path


def save_plots(trace, dgp, outdir):
    steps = np.asarray([row["step"] for row in trace], dtype=float)
    d = len(dgp.omega_star)
    omega = np.asarray(
        [[row[f"omega_{k}"] for k in range(d)] for row in trace], dtype=float
    )
    grad_norm = np.asarray([row["grad_norm"] for row in trace], dtype=float)
    errors = np.asarray([row["bellman_error"] for row in trace], dtype=float)
    omega_star = np.asarray(dgp.omega_star, dtype=float)

    omega_path = outdir / "bellman_obigrad_omega_trajectory.png"
    plt.figure(figsize=(8.0, 5.4))
    for k in range(d):
        plt.plot(steps, omega[:, k], marker="o", linewidth=1.8, label=rf"$\omega_{{{k}}}$")
        plt.axhline(
            omega_star[k],
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            color=f"C{k}",
        )
    plt.xlabel("outer optimization step")
    plt.ylabel("parameter value")
    plt.title("Projected Bellman optimization: OBiGrad trajectory")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(omega_path, dpi=220)
    plt.close()

    gradient_path = outdir / "bellman_obigrad_gradient_magnitude.png"
    plt.figure(figsize=(7.6, 5.2))
    plt.semilogy(steps, np.maximum(grad_norm, 1e-16), marker="o", linewidth=2.0)
    plt.xlabel("outer optimization step")
    plt.ylabel(r"$\|\widehat{\nabla F}(\omega_t)\|_2$")
    plt.title("OBiGrad magnitude along optimization")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(gradient_path, dpi=220)
    plt.close()

    error_path = outdir / "bellman_obigrad_bellman_error.png"
    plt.figure(figsize=(7.6, 5.2))
    plt.semilogy(steps, np.maximum(errors, 1e-16), marker="o", linewidth=2.0)
    plt.xlabel("outer optimization step")
    plt.ylabel(r"$F(\omega_t)-F(\omega^\star)$")
    plt.title("Population projected Bellman objective gap")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(error_path, dpi=220)
    plt.close()
    return [omega_path, gradient_path, error_path]


def write_run_config(dgp, learner, cfg, outdir):
    payload = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": {"MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR")},
        "dgp": asdict(dgp),
        "learner": asdict(learner),
        "optimization": asdict(cfg),
    }
    path = outdir / "run_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def run_experiment(dgp, learner, cfg):
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = write_run_config(dgp, learner, cfg, outdir)
    data = simulate_data(cfg.n, dgp, np.random.default_rng(cfg.seed))
    trace = optimize(data, dgp, learner, cfg)
    csv_path = write_trace_csv(trace, outdir)
    plot_paths = save_plots(trace, dgp, outdir)

    first = trace[0]
    final = trace[-1]
    best = min(trace, key=lambda row: row["bellman_error"])
    print(f"Initial omega: {first['omega']}")
    print(f"Final omega:   {final['omega']}")
    print(f"Final gradient estimate: {final['grad_hat']}")
    print(f"Final gradient norm: {final['grad_norm']:.8g}")
    print(f"Final Bellman objective gap: {final['bellman_error']:.8g}")
    print(
        f"Minimum Bellman objective gap: {best['bellman_error']:.8g} "
        f"at step {best['step']}"
    )
    print(f"Final distance to omega_star: {final['distance_to_target']:.8g}")
    print(f"Run configuration: {config_path}")
    print(f"CSV trace: {csv_path}")
    for path in plot_paths:
        print(f"Plot: {path}")
    return trace, csv_path, plot_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Outer optimization using B1's projected Bellman OBiGrad estimator."
    )
    parser.add_argument("--n", type=int, default=OptimizationConfig.n)
    parser.add_argument("--steps", type=int, default=OptimizationConfig.steps)
    parser.add_argument("--step-size", type=float, default=OptimizationConfig.step_size)
    parser.add_argument("--omega-init", type=float, default=OptimizationConfig.omega_init)
    parser.add_argument("--seed", type=int, default=OptimizationConfig.seed)
    parser.add_argument("--outdir", type=str, default=OptimizationConfig.outdir)
    parser.add_argument("--folds", type=int, default=OptimizationConfig.folds)
    parser.add_argument(
        "--learner", choices=["bellman_basis", "rff"], default=LearnerConfig.learner
    )
    parser.add_argument("--ridge", type=float, default=LearnerConfig.ridge)
    parser.add_argument("--rff-dim", type=int, default=LearnerConfig.rff_dim)
    parser.add_argument("--rff-gamma", type=float, default=LearnerConfig.rff_gamma)
    parser.add_argument(
        "--quadrature-nodes", type=int, default=OptimizationConfig.quadrature_nodes
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.n < 20:
        raise ValueError("--n must be at least 20")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.folds < 2 or args.folds > args.n:
        raise ValueError("--folds must be between 2 and --n")

    dgp = DGPConfig()
    learner = LearnerConfig(
        learner=args.learner,
        ridge=args.ridge,
        rff_dim=args.rff_dim,
        rff_gamma=args.rff_gamma,
    )
    cfg = OptimizationConfig(
        n=args.n,
        steps=args.steps,
        step_size=args.step_size,
        omega_init=args.omega_init,
        seed=args.seed,
        folds=args.folds,
        outdir=args.outdir,
        quadrature_nodes=args.quadrature_nodes,
    )
    run_experiment(dgp, learner, cfg)


if __name__ == "__main__":
    main()
