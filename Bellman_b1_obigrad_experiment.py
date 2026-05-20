#!/usr/bin/env python3
"""
Experiment B1: projected Bellman regression gradient estimation.

This is a fitted-Bellman-regression analogue of the IV experiments in the paper.
The lower-level loss is quadratic:

    h*_omega = argmin_h 0.5 E[(h(S,A) - g_omega(S,A,R,S'))^2]
             = E[g_omega(Z) | S,A]

with Bellman target

    g_omega(Z) = R + gamma V_omega(S'),   V_omega(s)=omega^T phi(s),
    phi(s) = (sin s, cos s, s, s^2).

The outer loss is also quadratic:

    F(omega) = 0.5 E[(Y - h*_omega(S,A))^2].

The script compares:
    - plug-in functional hypergradient,
    - OBiGrad / orthogonal DR gradient,
    - oracle DR gradient.

It produces PNG figures and generated LaTeX tables with Monte Carlo error bars.
"""


import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Keep BLAS from oversubscribing on repeated small regressions.
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

Z975 = 1.959963984540054


@dataclass(frozen=True)
class DGPConfig:
    rho: float = 0.7
    tau: float = 0.5
    sigma_s: float = 0.2
    sigma_r: float = 0.1
    sigma_y: float = 0.25
    gamma: float = 0.8
    p_action: float = 0.5
    omega_star: Tuple[float, ...] = (0.55, -0.35, 0.25, 0.15)
    omega_eval_shift: float = 0.35


@dataclass(frozen=True)
class LearnerConfig:
    learner: str = "bellman_basis"  # choices: bellman_basis, rff
    ridge: float = 5e-1
    rff_dim: int = 96
    rff_gamma: float = 0.75
    basis_degree: int = 3
    fit_intercept: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    n_grid: Tuple[int, ...] = (200, 400, 800, 1600, 3200)
    reps: int = 200
    folds: int = 2
    seed: int = 20260425
    outdir: str = "results/Bellman/b1"
    quadrature_nodes: int = 200


def parse_int_grid(s):
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if not vals or any(v < 20 for v in vals):
        raise ValueError("--n-grid must contain integers >= 20")
    return vals


def phi(s):
    s = np.asarray(s, dtype=float)
    return np.column_stack([np.sin(s), np.cos(s), s, s * s])


def dphi_mean(mu, sigma_s):
    """E[phi(S') | S,A] when S'|S,A ~ N(mu, sigma_s^2)."""
    atten = math.exp(-0.5 * sigma_s * sigma_s)
    return np.column_stack(
        [
            atten * np.sin(mu),
            atten * np.cos(mu),
            mu,
            mu * mu + sigma_s * sigma_s,
        ]
    )


def base_reward_mean(s, a):
    return np.sin(s) + 0.5 * a + 0.25 * s * a


def simulate_data(n, cfg, rng):
    s = rng.normal(size=n)
    a = rng.binomial(1, cfg.p_action, size=n).astype(float)
    xi = rng.normal(scale=cfg.sigma_s, size=n)
    sp = cfg.rho * s + cfg.tau * a + xi
    eps_r = rng.normal(scale=cfg.sigma_r, size=n)
    r = base_reward_mean(s, a) + eps_r
    phip = phi(sp)
    omega_star = np.asarray(cfg.omega_star, dtype=float)
    g_star = r + cfg.gamma * (phip @ omega_star)
    eps_y = rng.normal(scale=cfg.sigma_y, size=n)
    y = g_star + eps_y
    x = np.column_stack([s, a])
    return {"S": s, "A": a, "Sp": sp, "R": r, "PhiSp": phip, "X": x, "Y": y}


def true_nuisances(data, omega, cfg):
    s = data["S"]
    a = data["A"]
    mu = cfg.rho * s + cfg.tau * a
    mean_phi = dphi_mean(mu, cfg.sigma_s)
    j = cfg.gamma * mean_phi
    base = base_reward_mean(s, a)
    h = base + j @ omega
    m = base + j @ np.asarray(cfg.omega_star, dtype=float)
    return h, j, m


def make_eval_omega(cfg):
    omega_star = np.asarray(cfg.omega_star, dtype=float)
    direction = np.array([1.0, -0.5, 0.35, -0.25], dtype=float)
    direction = direction / np.linalg.norm(direction)
    return omega_star + cfg.omega_eval_shift * direction


def true_A_matrix(cfg, nodes=200):
    """Compute E[j*(X) j*(X)^T] by Gauss-Hermite quadrature over S and exact sum over A."""
    xs, ws = np.polynomial.hermite.hermgauss(nodes)
    s_vals = math.sqrt(2.0) * xs  # S ~ N(0,1)
    weights = ws / math.sqrt(math.pi)
    A_mat = np.zeros((4, 4), dtype=float)
    for aval, prob in [(0.0, 1.0 - cfg.p_action), (1.0, cfg.p_action)]:
        mu = cfg.rho * s_vals + cfg.tau * aval
        j = cfg.gamma * dphi_mean(mu, cfg.sigma_s)
        A_mat += prob * ((j.T * weights) @ j)
    return A_mat


def true_gradient(omega, cfg, nodes=200):
    return true_A_matrix(cfg, nodes=nodes) @ (
        omega - np.asarray(cfg.omega_star, dtype=float)
    )


class MultiOutputRegressor:
    def __init__(self, learner_cfg, dgp_cfg, seed):
        self.cfg = learner_cfg
        self.dgp = dgp_cfg
        self.seed = int(seed)
        self.beta_ = None
        self.x_mean_ = None
        self.x_scale_ = None
        self.W_ = None
        self.b_ = None

    def _standardize_fit(self, x):
        self.x_mean_ = x.mean(axis=0, keepdims=True)
        self.x_scale_ = x.std(axis=0, keepdims=True)
        self.x_scale_ = np.where(self.x_scale_ < 1e-12, 1.0, self.x_scale_)
        return (x - self.x_mean_) / self.x_scale_

    def _standardize_transform(self, x):
        if self.x_mean_ is None or self.x_scale_ is None:
            raise RuntimeError("model is not fitted")
        return (x - self.x_mean_) / self.x_scale_

    def _features_bellman_basis(self, x, fit):
        s = x[:, 0]
        a = x[:, 1]
        cols = []
        if self.cfg.fit_intercept:
            cols.append(np.ones_like(s))
        cols += [s, a, s * a, s * s, s * s * a, s**3, a * s**3]
        for freq in (0.5, 0.75, 1.0, 1.5, 2.0):
            sf = np.sin(freq * s)
            cf = np.cos(freq * s)
            cols += [sf, cf, a * sf, a * cf]
        for knot in (-2.0, -1.0, 0.0, 1.0, 2.0):
            bump = np.exp(-0.5 * ((s - knot) / 0.8) ** 2)
            cols += [bump, a * bump]
        return np.column_stack(cols)

    def _features_rff(self, x, fit):
        if fit:
            xs = self._standardize_fit(x)
            rng = np.random.default_rng(self.seed)
            self.W_ = rng.normal(
                scale=math.sqrt(2.0 * self.cfg.rff_gamma),
                size=(xs.shape[1], self.cfg.rff_dim),
            )
            self.b_ = rng.uniform(0.0, 2.0 * math.pi, size=self.cfg.rff_dim)
        else:
            xs = self._standardize_transform(x)
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("RFF parameters not initialized")
        z = math.sqrt(2.0 / self.cfg.rff_dim) * np.cos(xs @ self.W_ + self.b_)
        if self.cfg.fit_intercept:
            z = np.column_stack([np.ones(x.shape[0]), z])
        return z

    def _features(self, x, fit):
        x = np.asarray(x, dtype=float)
        if self.cfg.learner == "bellman_basis":
            return self._features_bellman_basis(x, fit=fit)
        if self.cfg.learner == "rff":
            return self._features_rff(x, fit=fit)
        raise ValueError(f"unknown learner: {self.cfg.learner}")

    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        z = self._features(x, fit=True)
        G = z.T @ z
        penalty = self.cfg.ridge * np.eye(G.shape[0])
        if self.cfg.fit_intercept:
            penalty[0, 0] = 0.0
        rhs = z.T @ y
        try:
            self.beta_ = np.linalg.solve(G + penalty, rhs)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(G + penalty, rhs, rcond=None)[0]
        return self

    def predict(self, x):
        if self.beta_ is None:
            raise RuntimeError("model is not fitted")
        z = self._features(x, fit=False)
        pred = z @ self.beta_
        return pred[:, 0] if pred.shape[1] == 1 else pred


def make_folds(n, k, rng):
    idx = rng.permutation(n)
    return [fold.astype(int) for fold in np.array_split(idx, k)]


def subset(data, idx):
    return {key: val[idx] for key, val in data.items()}


def nuisance_training_targets(data, omega_eval, cfg):
    g_eval = data["R"] + cfg.gamma * (data["PhiSp"] @ omega_eval)
    dg_eval = cfg.gamma * data["PhiSp"]
    y = data["Y"]
    return np.column_stack([g_eval, dg_eval, y])


def estimate_one(data, omega_eval, psi_true, dgp, learner, folds, rng, seed):
    n = len(data["Y"])
    d = len(omega_eval)
    phi_plugin = np.zeros((n, d))
    phi_dr = np.zeros((n, d))
    phi_oracle = np.zeros((n, d))
    jhat_all = np.zeros((n, d))
    hhat_all = np.zeros(n)
    mhat_all = np.zeros(n)

    g_eval_all = data["R"] + dgp.gamma * (data["PhiSp"] @ omega_eval)
    dg_eval_all = dgp.gamma * data["PhiSp"]
    h0_all, j0_all, m0_all = true_nuisances(data, omega_eval, dgp)
    fold_ids = make_folds(n, folds, rng)
    all_idx = np.arange(n)

    for fold_id, test_idx in enumerate(fold_ids):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        train = subset(data, train_idx)
        targets = nuisance_training_targets(train, omega_eval, dgp)
        model = MultiOutputRegressor(learner, dgp, seed=seed + 7919 * (fold_id + 1))
        model.fit(train["X"], targets)
        pred = model.predict(data["X"][test_idx])
        hhat = pred[:, 0]
        jhat = pred[:, 1 : 1 + d]
        mhat = pred[:, 1 + d]
        y = data["Y"][test_idx]
        g = g_eval_all[test_idx]
        dg = dg_eval_all[test_idx]

        phi_plugin[test_idx] = jhat * (hhat - y)[:, None]
        phi_dr[test_idx] = (
            jhat * (g - y)[:, None] + (dg - jhat) * (hhat - mhat)[:, None]
        )
        j0 = j0_all[test_idx]
        h0 = h0_all[test_idx]
        m0 = m0_all[test_idx]
        phi_oracle[test_idx] = j0 * (g - y)[:, None] + (dg - j0) * (h0 - m0)[:, None]

        jhat_all[test_idx] = jhat
        hhat_all[test_idx] = hhat
        mhat_all[test_idx] = mhat

    def est_stats(scores, prefix):
        est = scores.mean(axis=0)
        err = est - psi_true
        cov = np.cov(scores, rowvar=False)
        if d == 1:
            cov = np.array([[float(cov)]])
        se = np.sqrt(np.maximum(np.diag(cov), 0.0) / n)
        cover = np.abs(err) <= Z975 * se
        out = {
            f"{prefix}_l2_error": float(np.linalg.norm(err)),
            f"{prefix}_mean_abs_coord_error": float(np.mean(np.abs(err))),
            f"{prefix}_mean_se": float(np.mean(se)),
            f"{prefix}_mean_coord_coverage95": float(np.mean(cover)),
        }
        for k in range(d):
            out[f"{prefix}_est_{k}"] = float(est[k])
            out[f"{prefix}_err_{k}"] = float(err[k])
            out[f"{prefix}_se_{k}"] = float(se[k])
            out[f"{prefix}_cover95_{k}"] = float(cover[k])
        return out

    out = {}
    out.update(est_stats(phi_plugin, "plugin"))
    out.update(est_stats(phi_dr, "dr"))
    out.update(est_stats(phi_oracle, "oracle"))
    out["err_h"] = float(np.sqrt(np.mean((hhat_all - h0_all) ** 2)))
    out["err_j"] = float(np.sqrt(np.mean((jhat_all - j0_all) ** 2)))
    out["err_m"] = float(np.sqrt(np.mean((mhat_all - m0_all) ** 2)))
    out["product_bias_proxy"] = out["err_j"] * (out["err_h"] + out["err_m"])
    return out


def rmse_and_se(errors):
    sq = np.asarray(errors, dtype=float) ** 2
    mse = float(np.mean(sq))
    rmse = math.sqrt(mse)
    if len(sq) <= 1 or rmse <= 0:
        return rmse, 0.0
    se_mse = float(np.std(sq, ddof=1) / math.sqrt(len(sq)))
    se_rmse = se_mse / (2.0 * rmse)
    return rmse, se_rmse


def summarize(rows, d):
    methods = ["plugin", "dr", "oracle"]
    ns = sorted({int(row["n"]) for row in rows})
    summary = []
    for n in ns:
        sub = [row for row in rows if int(row["n"]) == n]
        for method in methods:
            l2 = np.array([row[f"{method}_l2_error"] for row in sub], dtype=float)
            rmse, rmse_se = rmse_and_se(l2)
            summary.append(
                {
                    "n": float(n),
                    "method": method,
                    "l2_rmse": rmse,
                    "l2_rmse_se": rmse_se,
                    "mean_l2_error": float(np.mean(l2)),
                    "mean_abs_coord_error": float(
                        np.mean([row[f"{method}_mean_abs_coord_error"] for row in sub])
                    ),
                    "mean_se": float(
                        np.mean([row[f"{method}_mean_se"] for row in sub])
                    ),
                    "mean_coord_coverage95": float(
                        np.mean([row[f"{method}_mean_coord_coverage95"] for row in sub])
                    ),
                    "mean_coord_coverage95_se": (
                        float(
                            np.std(
                                [row[f"{method}_mean_coord_coverage95"] for row in sub],
                                ddof=1,
                            )
                            / math.sqrt(len(sub))
                        )
                        if len(sub) > 1
                        else 0.0
                    ),
                }
            )
        summary.append(
            {
                "n": float(n),
                "method": "nuisance",
                "err_h": float(np.mean([row["err_h"] for row in sub])),
                "err_j": float(np.mean([row["err_j"] for row in sub])),
                "err_m": float(np.mean([row["err_m"] for row in sub])),
                "product_bias_proxy": float(
                    np.mean([row["product_bias_proxy"] for row in sub])
                ),
            }
        )
    return summary


def plot_rmse(summary, outdir):
    labels = {"plugin": "Plug-in", "dr": "OBiGrad / DR", "oracle": "Oracle DR"}
    plt.figure(figsize=(7.2, 5.0))
    for method in ["plugin", "dr", "oracle"]:
        sub = sorted(
            [row for row in summary if row["method"] == method], key=lambda r: r["n"]
        )
        n = np.array([row["n"] for row in sub], dtype=float)
        y = np.array([row["l2_rmse"] for row in sub], dtype=float)
        se = np.array([row["l2_rmse_se"] for row in sub], dtype=float)
        plt.errorbar(
            n,
            y,
            yerr=1.96 * se,
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=labels[method],
        )
    oracle = sorted(
        [row for row in summary if row["method"] == "oracle"], key=lambda r: r["n"]
    )
    if oracle:
        n = np.array([row["n"] for row in oracle], dtype=float)
        y0 = oracle[0]["l2_rmse"]
        n0 = oracle[0]["n"]
        plt.plot(
            n,
            y0 * np.sqrt(n0 / n),
            linestyle=":",
            linewidth=1.8,
            label=r"reference $n^{-1/2}$",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sample size n")
    plt.ylabel(r"gradient RMSE $\|\widehat\Psi_\omega-\Psi_\omega\|_2$")
    plt.title("Projected Bellman regression: fixed-gradient estimation")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bellman_b1_gradient_rmse.png", dpi=240)
    plt.close()


def plot_coverage(summary, outdir):
    labels = {"plugin": "Plug-in", "dr": "OBiGrad / DR", "oracle": "Oracle DR"}
    plt.figure(figsize=(7.2, 5.0))
    for method in ["plugin", "dr", "oracle"]:
        sub = sorted(
            [row for row in summary if row["method"] == method], key=lambda r: r["n"]
        )
        n = np.array([row["n"] for row in sub], dtype=float)
        y = np.array([row["mean_coord_coverage95"] for row in sub], dtype=float)
        se = np.array(
            [row.get("mean_coord_coverage95_se", 0.0) for row in sub], dtype=float
        )
        plt.errorbar(
            n,
            y,
            yerr=1.96 * se,
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=labels[method],
        )
    plt.axhline(0.95, linestyle=":", linewidth=1.8, label="nominal 95%")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("sample size n")
    plt.ylabel("mean coordinate coverage")
    plt.title("Projected Bellman regression: Wald coverage")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bellman_b1_coverage.png", dpi=240)
    plt.close()


def run_experiment(dgp, learner, exp):
    outdir = Path(exp.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    omega_eval = make_eval_omega(dgp)
    psi_true = true_gradient(omega_eval, dgp, nodes=exp.quadrature_nodes)
    rng_master = np.random.default_rng(exp.seed)
    rows = []
    print("Running projected Bellman regression experiment")
    print(
        f"  n_grid={exp.n_grid}, reps={exp.reps}, learner={learner.learner}, rff_dim={learner.rff_dim}"
    )
    print(f"  psi_true={np.array2string(psi_true, precision=4)}")
    for n in exp.n_grid:
        print(f"n={n}", flush=True)
        for rep in range(exp.reps):
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            data = simulate_data(n, dgp, rng)
            est = estimate_one(
                data,
                omega_eval,
                psi_true,
                dgp,
                learner,
                exp.folds,
                rng,
                seed=seed + 1000,
            )
            est.update({"n": float(n), "rep": float(rep), "seed": float(seed)})
            rows.append(est)
        # progress summary
        partial = [row for row in rows if int(row["n"]) == n]
        for method in ["plugin", "dr", "oracle"]:
            l2 = np.array([row[f"{method}_l2_error"] for row in partial])
            rmse, se = rmse_and_se(l2)
            print(f"  {method:7s}: RMSE={rmse:.4g} (MC se {se:.2g})")

    summary = summarize(rows, d=len(omega_eval))
    plot_rmse(summary, outdir)
    plot_coverage(summary, outdir)
    write_latex_outputs(summary, outdir)
    return rows, summary


def write_latex_outputs(summary, outdir):
    rows = []
    nuisance_rows = []
    for n in sorted({int(row["n"]) for row in summary if row["method"] != "nuisance"}):
        by_method = {
            row["method"]: row
            for row in summary
            if int(row["n"]) == n and row["method"] != "nuisance"
        }
        nuisance = next(
            row for row in summary if int(row["n"]) == n and row["method"] == "nuisance"
        )
        rows.append(
            [
                str(n),
                format_pm(by_method["plugin"]["l2_rmse"], by_method["plugin"].get("l2_rmse_se")),
                format_pm(by_method["dr"]["l2_rmse"], by_method["dr"].get("l2_rmse_se")),
                format_pm(by_method["oracle"]["l2_rmse"], by_method["oracle"].get("l2_rmse_se")),
                format_pm(
                    by_method["dr"]["mean_coord_coverage95"],
                    by_method["dr"].get("mean_coord_coverage95_se"),
                    digits=3,
                ),
                format_number(nuisance["product_bias_proxy"], digits=3),
            ]
        )
        nuisance_rows.append(
            [
                str(n),
                format_number(nuisance["err_h"], digits=4),
                format_number(nuisance["err_j"], digits=4),
                format_number(nuisance["err_m"], digits=4),
                format_number(nuisance["product_bias_proxy"], digits=4),
            ]
        )
    write_latex_table(
        outdir / "table_bellman_b1_gradient.tex",
        r"Projected Bellman gradient estimation. Parentheses report Monte Carlo 95\% error bars.",
        "tab:generated-bellman-b1-gradient",
        [r"$n$", "Plug-in", "OBiGrad", "Oracle DR", "DR coverage", "Product bias"],
        rows,
    )
    write_latex_table(
        outdir / "table_bellman_b1_nuisance.tex",
        "Nuisance-learning diagnostics for projected Bellman gradient estimation.",
        "tab:generated-bellman-b1-nuisance",
        [r"$n$", r"$\|\hat h-h^\star\|$", r"$\|\hat j-j^\star\|$", r"$\|\hat m-m^\star\|$", "Product bias"],
        nuisance_rows,
    )


def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment B1: projected Bellman regression OBiGrad gradient estimation."
    )
    p.add_argument("--n-grid", type=parse_int_grid, default=ExperimentConfig.n_grid)
    p.add_argument("--reps", type=int, default=ExperimentConfig.reps)
    p.add_argument("--folds", type=int, default=ExperimentConfig.folds)
    p.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    p.add_argument("--outdir", type=str, default=ExperimentConfig.outdir)
    p.add_argument(
        "--learner", choices=["bellman_basis", "rff"], default=LearnerConfig.learner
    )
    p.add_argument("--ridge", type=float, default=LearnerConfig.ridge)
    p.add_argument("--rff-dim", type=int, default=LearnerConfig.rff_dim)
    p.add_argument("--rff-gamma", type=float, default=LearnerConfig.rff_gamma)
    p.add_argument("--gamma", type=float, default=DGPConfig.gamma)
    p.add_argument("--rho", type=float, default=DGPConfig.rho)
    p.add_argument("--tau", type=float, default=DGPConfig.tau)
    p.add_argument("--sigma-s", type=float, default=DGPConfig.sigma_s)
    p.add_argument("--sigma-r", type=float, default=DGPConfig.sigma_r)
    p.add_argument("--sigma-y", type=float, default=DGPConfig.sigma_y)
    p.add_argument("--omega-eval-shift", type=float, default=DGPConfig.omega_eval_shift)
    return p


def main():
    args = build_parser().parse_args()
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    dgp = DGPConfig(
        rho=args.rho,
        tau=args.tau,
        sigma_s=args.sigma_s,
        sigma_r=args.sigma_r,
        sigma_y=args.sigma_y,
        gamma=args.gamma,
        omega_eval_shift=args.omega_eval_shift,
    )
    learner = LearnerConfig(
        learner=args.learner,
        ridge=args.ridge,
        rff_dim=args.rff_dim,
        rff_gamma=args.rff_gamma,
    )
    exp = ExperimentConfig(
        n_grid=args.n_grid,
        reps=args.reps,
        folds=args.folds,
        seed=args.seed,
        outdir=args.outdir,
    )
    run_experiment(dgp, learner, exp)
    print(f"Outputs written to {exp.outdir}")


if __name__ == "__main__":
    main()
