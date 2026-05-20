#!/usr/bin/env python3
"""
Experiment B3: projected Bellman regression root estimation.

This script implements a theory-aligned fitted Bellman regression experiment
with a quadratic inner loss and quadratic outer loss. The population root is
known analytically.

DGP
---
    S ~ N(0, 1),       A ~ Bernoulli(1/2)
    S' = transition_rho * S + action_shift * A + xi,
         xi ~ N(0, sigma_next^2)
    R = sin(S) + reward_action * A + reward_interaction * S A + eps_R
    Y = R + gamma * V_{omega_star}(S') + eps_Y

Bellman target
--------------
    g_omega(Z) = R + gamma * V_omega(S')
    V_omega(s') = omega^T phi(s')
    phi(s') = (sin(s'), cos(s'))

Quadratic inner problem
-----------------------
    h*_omega(S,A) = E[g_omega(Z) | S,A]
                  = h0*(S,A) + j*(S,A)^T omega
    j*(S,A)      = E[gamma * phi(S') | S,A]

Quadratic outer problem
-----------------------
    F(omega) = 0.5 E[(Y - h*_omega(S,A))^2]

Since E[Y | S,A] = h0*(S,A) + j*(S,A)^T omega_star, the population
root is omega_star whenever E[j*j^T] is nonsingular.

Methods
-------
    plug-in root
    OBiGrad / orthogonal DR root
    oracle DR root
    oracle plug-in root

Outputs
-------
    bellman_b3_root_rmse_vs_n.png
    bellman_b3_root_bias_vs_n.png
    table_bellman_b3_root_rmse.tex
    table_bellman_b3_root_appendix.tex
    bellman_b3_root_bias_vs_n.png
Example
-------
    python Bellman_b3_root_experiment.py --reps 200 --outdir results/Bellman/b3
"""


import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Keep linear algebra predictable on shared CPUs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_reporting import format_number, format_pm, write_latex_table


@dataclass(frozen=True)
class DGPConfig:
    gamma: float = 0.80
    transition_rho: float = 0.70
    action_shift: float = 0.50
    sigma_next: float = 0.20
    reward_action: float = 0.50
    reward_interaction: float = 0.25
    sigma_reward: float = 0.10
    sigma_y: float = 0.10
    omega_star_0: float = 0.65
    omega_star_1: float = -0.45

    @property
    def omega_star(self):
        return np.array([self.omega_star_0, self.omega_star_1], dtype=float)


@dataclass(frozen=True)
class LearnerConfig:
    ridge_alpha: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    n_grid: Tuple[int, ...] = (200, 400, 800, 1600, 3200)
    reps: int = 200
    folds: int = 2
    seed: int = 20260425
    outdir: str = "results/Bellman/b3"
    min_eig: float = 1e-10


def parse_int_grid(s):
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if not vals or any(v < 20 for v in vals):
        raise ValueError("--n-grid must contain integers at least 20")
    return vals


def phi_next(s_next):
    return np.column_stack([np.sin(s_next), np.cos(s_next)])


def simulate_data(n, dgp, rng):
    S = rng.normal(size=n)
    A = rng.binomial(1, 0.5, size=n).astype(float)
    xi = rng.normal(scale=dgp.sigma_next, size=n)
    S_next = dgp.transition_rho * S + dgp.action_shift * A + xi
    eps_R = rng.normal(scale=dgp.sigma_reward, size=n)
    R = np.sin(S) + dgp.reward_action * A + dgp.reward_interaction * S * A + eps_R
    eps_Y = rng.normal(scale=dgp.sigma_y, size=n)
    Phi_next = phi_next(S_next)
    Y = R + dgp.gamma * (Phi_next @ dgp.omega_star) + eps_Y
    X = np.column_stack([S, A])
    Gfeat = dgp.gamma * Phi_next
    return {
        "S": S,
        "A": A,
        "X": X,
        "S_next": S_next,
        "Phi_next": Phi_next,
        "Gfeat": Gfeat,
        "R": R,
        "Y": Y,
    }


def transition_mean(S, A, dgp):
    return dgp.transition_rho * S + dgp.action_shift * A


def h0_true(S, A, dgp):
    return np.sin(S) + dgp.reward_action * A + dgp.reward_interaction * S * A


def j_true(S, A, dgp):
    mu = transition_mean(S, A, dgp)
    attenuation = math.exp(-0.5 * dgp.sigma_next**2)
    e_sin = attenuation * np.sin(mu)
    e_cos = attenuation * np.cos(mu)
    return dgp.gamma * np.column_stack([e_sin, e_cos])


def m_true(S, A, dgp):
    return h0_true(S, A, dgp) + j_true(S, A, dgp) @ dgp.omega_star


def design_matrix(X, dgp, learner):
    S = X[:, 0]
    A = X[:, 1]
    cols = [
        np.ones_like(S),
        S,
        A,
        S * A,
        S**2,
        (S**2) * A,
        S**3,
        (S**3) * A,
    ]
    for freq in (0.5, 0.75, 1.0, 1.5, 2.0):
        sf = np.sin(freq * S)
        cf = np.cos(freq * S)
        cols.extend([sf, cf, sf * A, cf * A])
    for knot in (-2.0, -1.0, 0.0, 1.0, 2.0):
        bump = np.exp(-0.5 * ((S - knot) / 0.8) ** 2)
        cols.extend([bump, bump * A])
    return np.column_stack(cols)


def ridge_fit_predict(
    X_train,
    Y_train,
    X_eval,
    dgp,
    learner,
):
    Y = np.asarray(Y_train, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Z = design_matrix(X_train, dgp, learner)
    Ze = design_matrix(X_eval, dgp, learner)
    gram = Z.T @ Z
    penalty = learner.ridge_alpha * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0
    rhs = Z.T @ Y
    try:
        beta = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    pred = Ze @ beta
    return pred[:, 0] if pred.shape[1] == 1 else pred


def safe_solve(A, b, min_eig):
    """Solve A x = b robustly without forcing symmetry.

    The OBiGrad root equation is generally affine with a non-symmetric
    coefficient matrix, even though its population limit is well-conditioned in
    this DGP. Symmetrizing this matrix can create spurious singularities, so we
    solve the original system and fall back to least squares when needed.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        svals = np.linalg.svd(A, compute_uv=False)
        if float(np.min(svals)) < min_eig:
            return np.linalg.lstsq(A, b, rcond=min_eig)[0]
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=min_eig)[0]


def make_folds(n, k, rng):
    idx = rng.permutation(n)
    return [fold.astype(int) for fold in np.array_split(idx, k)]


def accumulate_coefficients(
    X_eval,
    R_eval,
    Y_eval,
    Gfeat_eval,
    h0_hat,
    j_hat,
    m_hat,
):
    """Return coefficient pairs for plugin and OBiGrad root equations.

    Plug-in score: A_pi omega - b_pi = 0.
    DR score:     A_dr omega + c_dr = 0.
    """
    n = X_eval.shape[0]
    # Plug-in: mean_j j^T omega + mean_j(h0-Y) = 0
    A_pi = (j_hat.T @ j_hat) / n
    b_pi = np.mean(j_hat * (Y_eval - h0_hat)[:, None], axis=0)

    # OBiGrad score is affine: A_dr omega + c_dr.
    A_dr = np.zeros((j_hat.shape[1], j_hat.shape[1]), dtype=float)
    for i in range(n):
        j = j_hat[i]
        gf = Gfeat_eval[i]
        A_dr += np.outer(j, gf) + np.outer(gf - j, j)
    A_dr /= n
    c_dr = np.mean(
        j_hat * (R_eval - Y_eval)[:, None]
        + (Gfeat_eval - j_hat) * (h0_hat - m_hat)[:, None],
        axis=0,
    )
    return A_pi, b_pi, A_dr, c_dr


def roots_one_rep(n, dgp, learner, exp, rng):
    data = simulate_data(n, dgp, rng)
    folds = make_folds(n, exp.folds, rng)
    d = len(dgp.omega_star)

    A_pi_sum = np.zeros((d, d))
    b_pi_sum = np.zeros(d)
    A_dr_sum = np.zeros((d, d))
    c_dr_sum = np.zeros(d)
    total_eval = 0

    all_idx = np.arange(n)
    for eval_idx in folds:
        train_idx = np.setdiff1d(all_idx, eval_idx, assume_unique=False)
        targets = np.column_stack(
            [data["R"][train_idx], data["Gfeat"][train_idx, :], data["Y"][train_idx]]
        )
        pred = ridge_fit_predict(
            data["X"][train_idx], targets, data["X"][eval_idx], dgp, learner
        )
        h0_hat = pred[:, 0]
        j_hat = pred[:, 1 : 1 + d]
        m_hat = pred[:, 1 + d]
        A_pi, b_pi, A_dr, c_dr = accumulate_coefficients(
            X_eval=data["X"][eval_idx],
            R_eval=data["R"][eval_idx],
            Y_eval=data["Y"][eval_idx],
            Gfeat_eval=data["Gfeat"][eval_idx, :],
            h0_hat=h0_hat,
            j_hat=j_hat,
            m_hat=m_hat,
        )
        w = len(eval_idx)
        A_pi_sum += w * A_pi
        b_pi_sum += w * b_pi
        A_dr_sum += w * A_dr
        c_dr_sum += w * c_dr
        total_eval += w

    A_pi_hat = A_pi_sum / total_eval
    b_pi_hat = b_pi_sum / total_eval
    A_dr_hat = A_dr_sum / total_eval
    c_dr_hat = c_dr_sum / total_eval

    omega_pi = safe_solve(A_pi_hat, b_pi_hat, exp.min_eig)
    omega_dr = safe_solve(A_dr_hat, -c_dr_hat, exp.min_eig)

    # Oracle roots on the same full sample.
    S, A = data["S"], data["A"]
    j0 = j_true(S, A, dgp)
    h00 = h0_true(S, A, dgp)
    m0 = m_true(S, A, dgp)
    Gfeat = data["Gfeat"]
    Y = data["Y"]
    R = data["R"]

    A_or_pi = (j0.T @ j0) / n
    b_or_pi = np.mean(j0 * (Y - h00)[:, None], axis=0)
    omega_or_pi = safe_solve(A_or_pi, b_or_pi, exp.min_eig)

    A_or_dr = np.zeros((d, d), dtype=float)
    for i in range(n):
        j = j0[i]
        gf = Gfeat[i]
        A_or_dr += np.outer(j, gf) + np.outer(gf - j, j)
    A_or_dr /= n
    c_or_dr = np.mean(
        j0 * (R - Y)[:, None] + (Gfeat - j0) * (h00 - m0)[:, None], axis=0
    )
    omega_or_dr = safe_solve(A_or_dr, -c_or_dr, exp.min_eig)

    # Nuisance diagnostics on all data with cross-fitted predictions are expensive to store; skip by default.
    return {
        "plugin": omega_pi,
        "autodml": omega_dr,
        "oracle_dr": omega_or_dr,
        "oracle_plugin": omega_or_pi,
    }


def summarize(rep_rows):
    methods = ["plugin", "autodml", "oracle_dr", "oracle_plugin"]
    ns = sorted({int(r["n"]) for r in rep_rows})
    out = []
    for n in ns:
        for method in methods:
            sub = [r for r in rep_rows if int(r["n"]) == n and r["method"] == method]
            errs = np.array([float(r["error_norm"]) for r in sub], dtype=float)
            signed0 = np.array([float(r["error_0"]) for r in sub], dtype=float)
            signed1 = np.array([float(r["error_1"]) for r in sub], dtype=float)
            sq = errs**2
            rmse = float(np.sqrt(np.mean(sq)))
            se_rmse = (
                float(np.std(sq, ddof=1) / math.sqrt(len(sq)) / (2 * rmse))
                if len(sq) > 1 and rmse > 0
                else 0.0
            )
            out.append(
                {
                    "n": float(n),
                    "method": method,
                    "reps": float(len(sub)),
                    "rmse": rmse,
                    "rmse_se": se_rmse,
                    "rmse_ci95_low": max(0.0, rmse - 1.96 * se_rmse),
                    "rmse_ci95_high": rmse + 1.96 * se_rmse,
                    "mean_abs_error": float(np.mean(errs)),
                    "median_abs_error": float(np.median(errs)),
                    "q90_abs_error": float(np.quantile(errs, 0.90)),
                    "bias_norm": float(
                        np.linalg.norm([np.mean(signed0), np.mean(signed1)])
                    ),
                    "bias_norm_se": (
                        float(
                            np.sqrt(
                                np.var(signed0, ddof=1) + np.var(signed1, ddof=1)
                            )
                            / math.sqrt(len(sub))
                        )
                        if len(sub) > 1
                        else 0.0
                    ),
                    "bias_0": float(np.mean(signed0)),
                    "bias_1": float(np.mean(signed1)),
                    "sd_0": (
                        float(np.std(signed0, ddof=1)) if len(sub) > 1 else float("nan")
                    ),
                    "sd_1": (
                        float(np.std(signed1, ddof=1)) if len(sub) > 1 else float("nan")
                    ),
                }
            )
    return out


def plot_rmse(summary, outdir):
    labels = {
        "plugin": "Plug-in root",
        "autodml": "OBiGrad / DR root",
        "oracle_dr": "Oracle DR root",
        "oracle_plugin": "Oracle plug-in root",
    }
    plt.figure(figsize=(7.2, 5.0))
    for method in ["plugin", "autodml", "oracle_dr", "oracle_plugin"]:
        sub = sorted(
            [r for r in summary if r["method"] == method], key=lambda x: float(x["n"])
        )
        n = np.array([float(r["n"]) for r in sub])
        y = np.array([float(r["rmse"]) for r in sub])
        yerr = 1.96 * np.array([float(r["rmse_se"]) for r in sub])
        plt.errorbar(
            n, y, yerr=yerr, marker="o", linewidth=2, capsize=3, label=labels[method]
        )
    # n^-1/2 reference anchored at oracle DR first point.
    oracle = sorted(
        [r for r in summary if r["method"] == "oracle_dr"], key=lambda x: float(x["n"])
    )
    if oracle:
        n_ref = np.array([float(r["n"]) for r in oracle])
        y0 = float(oracle[0]["rmse"])
        n0 = float(oracle[0]["n"])
        plt.plot(
            n_ref,
            y0 * np.sqrt(n0 / n_ref),
            linestyle="--",
            linewidth=1.8,
            label=r"reference $n^{-1/2}$",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sample size n")
    plt.ylabel(r"root RMSE $\|\hat\omega-\omega^\star\|_2$")
    plt.title("Projected Bellman regression root estimation")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "bellman_b3_root_rmse_vs_n.png", dpi=220)
    plt.close()


def plot_bias(summary, outdir):
    labels = {
        "plugin": "Plug-in root",
        "autodml": "OBiGrad / DR root",
        "oracle_dr": "Oracle DR root",
        "oracle_plugin": "Oracle plug-in root",
    }
    plt.figure(figsize=(7.2, 5.0))
    for method in ["plugin", "autodml", "oracle_dr", "oracle_plugin"]:
        sub = sorted(
            [r for r in summary if r["method"] == method], key=lambda x: float(x["n"])
        )
        n = np.array([float(r["n"]) for r in sub])
        y = np.array([float(r["bias_norm"]) for r in sub])
        se = np.array([float(r.get("bias_norm_se", 0.0)) for r in sub])
        plt.errorbar(
            n,
            np.maximum(y, 1e-12),
            yerr=1.96 * se,
            marker="o",
            linewidth=2,
            capsize=3,
            label=labels[method],
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sample size n")
    plt.ylabel(r"empirical bias norm $\|E\hat\omega-\omega^\star\|_2$")
    plt.title("Projected Bellman regression root bias")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "bellman_b3_root_bias_vs_n.png", dpi=220)
    plt.close()


def run_experiment(dgp, learner, exp):
    outdir = Path(exp.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(exp.seed)
    rep_rows = []
    omega_star = dgp.omega_star

    for n in exp.n_grid:
        print(f"n={n}, reps={exp.reps}", flush=True)
        for rep in range(exp.reps):
            seed = int(rng_master.integers(0, np.iinfo(np.int32).max))
            rng = np.random.default_rng(seed)
            roots = roots_one_rep(n, dgp, learner, exp, rng)
            for method, omega_hat in roots.items():
                err_vec = np.asarray(omega_hat) - omega_star
                rep_rows.append(
                    {
                        "n": float(n),
                        "rep": float(rep),
                        "seed": float(seed),
                        "method": method,
                        "omega_hat_0": float(omega_hat[0]),
                        "omega_hat_1": float(omega_hat[1]),
                        "omega_star_0": float(omega_star[0]),
                        "omega_star_1": float(omega_star[1]),
                        "error_0": float(err_vec[0]),
                        "error_1": float(err_vec[1]),
                        "error_norm": float(np.linalg.norm(err_vec)),
                    }
                )
        # small progress summary
        partial = summarize([r for r in rep_rows if int(r["n"]) == n])
        msg = ", ".join([f"{r['method']} RMSE={float(r['rmse']):.4g}" for r in partial])
        print("  " + msg, flush=True)

    summary = summarize(rep_rows)
    plot_rmse(summary, outdir)
    plot_bias(summary, outdir)
    write_latex_outputs(summary, outdir)
    return rep_rows, summary


def write_latex_outputs(summary, outdir):
    rows = []
    appendix_rows = []
    order = ["plugin", "autodml", "oracle_dr", "oracle_plugin"]
    labels = {
        "plugin": "Plug-in",
        "autodml": "OBiGrad",
        "oracle_dr": "Oracle DR",
        "oracle_plugin": "Oracle plug-in",
    }
    for n in sorted({int(row["n"]) for row in summary}):
        by_method = {
            row["method"]: row for row in summary if int(row["n"]) == n
        }
        rows.append(
            [str(n)]
            + [
                format_pm(by_method[method]["rmse"], by_method[method].get("rmse_se"))
                for method in order
            ]
        )
        for method in order:
            item = by_method[method]
            appendix_rows.append(
                [
                    str(n),
                    labels[method],
                    format_number(item["bias_norm"], digits=4),
                    format_number(item["mean_abs_error"], digits=4),
                    format_number(item["median_abs_error"], digits=4),
                    format_number(item["q90_abs_error"], digits=4),
                ]
            )
    write_latex_table(
        outdir / "table_bellman_b3_root_rmse.tex",
        r"Projected Bellman root-estimation RMSE. Parentheses report Monte Carlo 95\% error bars.",
        "tab:generated-bellman-b3-root-rmse",
        [r"$n$", "Plug-in", "OBiGrad", "Oracle DR", "Oracle plug-in"],
        rows,
    )
    write_latex_table(
        outdir / "table_bellman_b3_root_appendix.tex",
        "Additional projected Bellman root-estimation diagnostics.",
        "tab:generated-bellman-b3-root-appendix",
        [r"$n$", "Method", "Bias norm", "Mean abs.", "Median abs.", r"90\% abs."],
        appendix_rows,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Projected Bellman regression root estimation experiment B3."
    )
    parser.add_argument(
        "--n-grid", type=parse_int_grid, default=ExperimentConfig.n_grid
    )
    parser.add_argument("--reps", type=int, default=ExperimentConfig.reps)
    parser.add_argument("--folds", type=int, default=ExperimentConfig.folds)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--outdir", type=str, default=ExperimentConfig.outdir)
    parser.add_argument("--ridge-alpha", type=float, default=LearnerConfig.ridge_alpha)
    parser.add_argument("--sigma-y", type=float, default=DGPConfig.sigma_y)
    parser.add_argument("--sigma-reward", type=float, default=DGPConfig.sigma_reward)
    return parser


def main():
    args = build_parser().parse_args()
    dgp = DGPConfig(sigma_y=args.sigma_y, sigma_reward=args.sigma_reward)
    learner = LearnerConfig(ridge_alpha=args.ridge_alpha)
    exp = ExperimentConfig(
        n_grid=args.n_grid,
        reps=args.reps,
        folds=args.folds,
        seed=args.seed,
        outdir=args.outdir,
    )
    run_experiment(dgp, learner, exp)


if __name__ == "__main__":
    main()
