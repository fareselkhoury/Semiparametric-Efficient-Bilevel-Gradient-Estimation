#!/usr/bin/env python3
"""
Experiment B2: Wald calibration for the projected Bellman-regression example.

This is the inference companion to Experiment B1. It uses the same quadratic
projected Bellman DGP with analytic ground truth, and reports coordinate-wise
95\\% Wald coverage, interval lengths, and studentized-error diagnostics for:
  1. plug-in normal intervals,
  2. OBiGrad / orthogonal DR intervals,
  3. oracle DR intervals.

The nuisance learner is a multi-output ridge regression on a fixed nonlinear
basis of X=(S,A). This keeps the experiment fast and stable while still learning
conditional means from data rather than using oracle nuisance functions.
"""


import argparse
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid BLAS oversubscription in repeated small ridge solves.
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
    gamma: float = 0.90
    rho: float = 0.70
    tau: float = 0.50
    sigma_snext: float = 0.20
    sigma_reward: float = 0.10
    sigma_y: float = 0.25
    beta_behavior: float = 0.50
    omega_shift: float = 0.35

    @property
    def omega_star(self):
        v = np.array([1.0, -0.6, 0.35, 0.15], dtype=float)
        return v / np.linalg.norm(v)

    @property
    def omega_eval(self):
        direction = np.array([0.6, 0.2, -0.5, 0.3], dtype=float)
        direction = direction / np.linalg.norm(direction)
        return self.omega_star + self.omega_shift * direction


@dataclass(frozen=True)
class LearnerConfig:
    ridge_alpha: float = 5e-2
    fit_intercept: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    n_grid: Tuple[int, ...] = (200, 400, 800, 1600, 3200)
    reps: int = 200
    folds: int = 2
    seed: int = 20260425
    quadrature_nodes: int = 200
    diagnostic_coord: int = 0
    outdir: str = "results/Bellman/b2"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def parse_n_grid(text):
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals or min(vals) < 50:
        raise ValueError("--n-grid must contain integers >= 50")
    return vals


def phi_features(sp):
    sp = np.asarray(sp, dtype=float)
    return np.column_stack([np.sin(sp), np.cos(sp), sp, sp**2])


def conditional_phi_mean(s, a, cfg):
    s = np.asarray(s, dtype=float)
    a = np.asarray(a, dtype=float)
    mu = cfg.rho * s + cfg.tau * a
    sig2 = cfg.sigma_snext**2
    attenuation = math.exp(-0.5 * sig2)
    return np.column_stack(
        [
            attenuation * np.sin(mu),
            attenuation * np.cos(mu),
            mu,
            mu**2 + sig2,
        ]
    )


def reward_mean(s, a):
    return np.sin(s) + 0.5 * a + 0.25 * s * a


def true_j(s, a, cfg):
    return cfg.gamma * conditional_phi_mean(s, a, cfg)


def true_h(s, a, omega, cfg):
    return reward_mean(s, a) + true_j(s, a, cfg) @ omega


def true_m(s, a, cfg):
    return reward_mean(s, a) + true_j(s, a, cfg) @ cfg.omega_star


def true_A_matrix(cfg, nodes=200):
    gh_x, gh_w = np.polynomial.hermite.hermgauss(nodes)
    s = math.sqrt(2.0) * gh_x
    weights = gh_w / math.sqrt(math.pi)
    pi = sigmoid(cfg.beta_behavior * s)
    A = np.zeros((4, 4), dtype=float)
    for a_val, prob in [(0.0, 1.0 - pi), (1.0, pi)]:
        a = np.full_like(s, a_val)
        j = true_j(s, a, cfg)
        A += (j * (weights * prob)[:, None]).T @ j
    return A


def true_gradient(cfg, nodes=200):
    return true_A_matrix(cfg, nodes) @ (cfg.omega_eval - cfg.omega_star)


def simulate_data(n, cfg, rng):
    s = rng.normal(size=n)
    pi = sigmoid(cfg.beta_behavior * s)
    a = rng.binomial(1, pi).astype(float)
    sp = cfg.rho * s + cfg.tau * a + rng.normal(scale=cfg.sigma_snext, size=n)
    r = reward_mean(s, a) + rng.normal(scale=cfg.sigma_reward, size=n)
    phi_sp = phi_features(sp)
    dg = cfg.gamma * phi_sp
    g_eval = r + dg @ cfg.omega_eval
    y = r + dg @ cfg.omega_star + rng.normal(scale=cfg.sigma_y, size=n)
    return {
        "S": s,
        "A": a,
        "Sp": sp,
        "R": r,
        "X": np.column_stack([s, a]),
        "dg": dg,
        "g_eval": g_eval,
        "Y": y,
    }


def basis_features(X, cfg, fit_intercept=True):
    """Observable-only nonlinear basis for conditional means."""
    s = X[:, 0]
    a = X[:, 1]
    cols = []
    if fit_intercept:
        cols.append(np.ones_like(s))
    cols.extend(
        [
            s,
            a,
            s * a,
            s**2,
            a * s**2,
            s**3,
            a * s**3,
        ]
    )
    for freq in (0.5, 0.75, 1.0, 1.5, 2.0):
        sf = np.sin(freq * s)
        cf = np.cos(freq * s)
        cols.extend([sf, cf, a * sf, a * cf])
    for knot in (-2.0, -1.0, 0.0, 1.0, 2.0):
        bump = np.exp(-0.5 * ((s - knot) / 0.8) ** 2)
        cols.extend([bump, a * bump])
    return np.column_stack(cols)


class BasisRidgeMultiOutput:
    def __init__(self, cfg, dgp_cfg):
        self.cfg = cfg
        self.dgp_cfg = dgp_cfg
        self.beta_ = None

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        Z = basis_features(
            np.asarray(X, dtype=float), self.dgp_cfg, self.cfg.fit_intercept
        )
        G = Z.T @ Z
        penalty = self.cfg.ridge_alpha * np.eye(G.shape[0])
        if self.cfg.fit_intercept:
            penalty[0, 0] = 0.0
        rhs = Z.T @ Y
        try:
            self.beta_ = np.linalg.solve(G + penalty, rhs)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(G + penalty, rhs, rcond=None)[0]
        return self

    def predict(self, X):
        if self.beta_ is None:
            raise RuntimeError("model not fitted")
        Z = basis_features(
            np.asarray(X, dtype=float), self.dgp_cfg, self.cfg.fit_intercept
        )
        pred = Z @ self.beta_
        return pred[:, 0] if pred.shape[1] == 1 else pred


def make_folds(n, k, rng):
    idx = rng.permutation(n)
    return [fold.astype(int) for fold in np.array_split(idx, k)]


def score_stats(scores, estimate, psi_true):
    n = scores.shape[0]
    se = np.sqrt(np.maximum(np.var(scores, axis=0, ddof=1), 0.0) / n)
    err = estimate - psi_true
    cover = (
        (estimate - Z975 * se <= psi_true) & (psi_true <= estimate + Z975 * se)
    ).astype(float)
    length = 2.0 * Z975 * se
    z = np.full_like(err, np.nan, dtype=float)
    mask = se > 0
    z[mask] = err[mask] / se[mask]
    return {"se": se, "err": err, "cover": cover, "length": length, "z": z}


def estimate_one_dataset(data, dgp_cfg, learner_cfg, folds, rng, psi_true):
    n = data["X"].shape[0]
    d = 4
    fold_ids = make_folds(n, folds, rng)
    all_idx = np.arange(n)

    scores = {
        "plugin": np.zeros((n, d)),
        "dr": np.zeros((n, d)),
        "oracle": np.zeros((n, d)),
    }
    hhat_all = np.zeros(n)
    jhat_all = np.zeros((n, d))
    mhat_all = np.zeros(n)

    s, a = data["S"], data["A"]
    j0_all = true_j(s, a, dgp_cfg)
    h0_all = true_h(s, a, dgp_cfg.omega_eval, dgp_cfg)
    m0_all = true_m(s, a, dgp_cfg)

    for test_idx in fold_ids:
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        targets = np.column_stack(
            [data["g_eval"][train_idx], data["dg"][train_idx, :], data["Y"][train_idx]]
        )
        model = BasisRidgeMultiOutput(learner_cfg, dgp_cfg).fit(
            data["X"][train_idx], targets
        )
        pred = model.predict(data["X"][test_idx])
        hhat = pred[:, 0]
        jhat = pred[:, 1 : 1 + d]
        mhat = pred[:, 1 + d]

        g = data["g_eval"][test_idx]
        dg = data["dg"][test_idx, :]
        y = data["Y"][test_idx]
        j0 = j0_all[test_idx, :]
        h0 = h0_all[test_idx]
        m0 = m0_all[test_idx]

        scores["plugin"][test_idx, :] = jhat * (hhat - y)[:, None]
        scores["dr"][test_idx, :] = (
            jhat * (g - y)[:, None] + (dg - jhat) * (hhat - mhat)[:, None]
        )
        scores["oracle"][test_idx, :] = (
            j0 * (g - y)[:, None] + (dg - j0) * (h0 - m0)[:, None]
        )
        hhat_all[test_idx] = hhat
        jhat_all[test_idx, :] = jhat
        mhat_all[test_idx] = mhat

    estimates = {method: mat.mean(axis=0) for method, mat in scores.items()}
    stats = {
        method: score_stats(scores[method], estimates[method], psi_true)
        for method in scores
    }
    nuisance = {
        "err_h": float(np.sqrt(np.mean((hhat_all - h0_all) ** 2))),
        "err_j": float(np.sqrt(np.mean((jhat_all - j0_all) ** 2))),
        "err_m": float(np.sqrt(np.mean((mhat_all - m0_all) ** 2))),
    }
    nuisance["product_bias_proxy"] = nuisance["err_j"] * (
        nuisance["err_h"] + nuisance["err_m"]
    )
    return estimates, stats, nuisance


def mean_se(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(vals.size))


def summarize(rep_rows, z_rows):
    out = []
    ns = sorted(set(int(r["n"]) for r in z_rows))
    for n in ns:
        for method in ["plugin", "dr", "oracle"]:
            sub = [r for r in z_rows if int(r["n"]) == n and r["method"] == method]
            err = np.array([r["error"] for r in sub], dtype=float)
            cover = np.array([r["cover95"] for r in sub], dtype=float)
            length = np.array([r["ci_length95"] for r in sub], dtype=float)
            se_vals = np.array([r["se"] for r in sub], dtype=float)
            z = np.array([r["z"] for r in sub], dtype=float)
            z = z[np.isfinite(z)]
            row = {"n": float(n), "method": method, "num_coord_rep": float(len(sub))}
            row["coverage"], row["coverage_se"] = mean_se(cover)
            row["ci_length_mean"], row["ci_length_se"] = mean_se(length)
            row["mean_se"], row["mean_se_mc_se"] = mean_se(se_vals)
            row["bias"], row["bias_se"] = mean_se(err)
            sq_err = err**2
            row["rmse"] = float(math.sqrt(np.mean(sq_err)))
            row["rmse_se"] = (
                float(np.std(sq_err, ddof=1) / math.sqrt(len(sq_err)) / (2.0 * row["rmse"]))
                if len(sq_err) > 1 and row["rmse"] > 0
                else 0.0
            )
            row["z_mean"], row["z_mean_se"] = mean_se(z)
            row["z_sd"] = float(np.std(z, ddof=1)) if z.size > 1 else float("nan")
            row["z_std"] = row["z_sd"]
            row["z_q025"] = float(np.quantile(z, 0.025)) if z.size else float("nan")
            row["z_q500"] = float(np.quantile(z, 0.500)) if z.size else float("nan")
            row["z_q975"] = float(np.quantile(z, 0.975)) if z.size else float("nan")
            row["z_abs_gt_196"] = (
                float(np.mean(np.abs(z) > Z975)) if z.size else float("nan")
            )
            rep_covs = [
                rr[f"{method}_mean_coord_coverage95"]
                for rr in rep_rows
                if int(rr["n"]) == n
            ]
            row["coverage_rep_se"] = (
                float(np.std(rep_covs, ddof=1) / math.sqrt(len(rep_covs)))
                if len(rep_covs) > 1
                else 0.0
            )
            rep_n = [rr for rr in rep_rows if int(rr["n"]) == n]
            if rep_n:
                for nuisance_name in ["err_h", "err_j", "err_m", "product_bias_proxy"]:
                    vals = [rr[nuisance_name] for rr in rep_n if nuisance_name in rr]
                    row[f"{nuisance_name}_mean"] = (
                        float(np.mean(vals)) if vals else float("nan")
                    )
            out.append(row)
    return out


def save_coverage_plot(summary, outdir):
    label_map = {
        "plugin": "Plug-in normal interval",
        "dr": "OBiGrad / orthogonal DR",
        "oracle": "Oracle DR",
    }
    marker_map = {"plugin": "o", "dr": "s", "oracle": "^"}
    plt.figure(figsize=(7.0, 5.0))
    for method in ["dr", "oracle", "plugin"]:
        rows = sorted(
            [r for r in summary if r["method"] == method], key=lambda x: x["n"]
        )
        n = np.array([r["n"] for r in rows], dtype=float)
        y = np.array([r["coverage"] for r in rows], dtype=float)
        err = 1.96 * np.array([r["coverage_rep_se"] for r in rows], dtype=float)
        plt.errorbar(
            n,
            y,
            yerr=err,
            marker=marker_map[method],
            linewidth=2,
            capsize=3,
            label=label_map[method],
        )
    plt.axhline(0.95, linestyle="--", linewidth=1.5, label="nominal 95%")
    plt.xscale("log")
    plt.ylim(0.88, 1.01)
    plt.xlabel("sample size n")
    plt.ylabel("empirical coordinate-wise coverage")
    plt.title("Fitted Bellman regression: Wald interval coverage")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bellman_b2_coverage_vs_n.png", dpi=240)
    plt.savefig(outdir / "bellman_b2_coverage_vs_n.pdf")
    plt.close()


def save_length_plot(summary, outdir):
    label_map = {
        "plugin": "Plug-in normal interval",
        "dr": "OBiGrad / orthogonal DR",
        "oracle": "Oracle DR",
    }
    marker_map = {"plugin": "o", "dr": "s", "oracle": "^"}
    plt.figure(figsize=(7.0, 5.0))
    ref_n = None
    ref_y = None
    for method in ["dr", "oracle", "plugin"]:
        rows = sorted(
            [r for r in summary if r["method"] == method], key=lambda x: x["n"]
        )
        n = np.array([r["n"] for r in rows], dtype=float)
        y = np.array([r["ci_length_mean"] for r in rows], dtype=float)
        err = 1.96 * np.array([r["ci_length_se"] for r in rows], dtype=float)
        plt.errorbar(
            n,
            y,
            yerr=err,
            marker=marker_map[method],
            linewidth=2,
            capsize=3,
            label=label_map[method],
        )
        if method == "oracle":
            ref_n = n
            ref_y = y[0]
            n0 = n[0]
    if ref_n is not None:
        plt.plot(
            ref_n,
            ref_y * np.sqrt(n0 / ref_n),
            linestyle="--",
            linewidth=1.5,
            label=r"reference $n^{-1/2}$",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sample size n")
    plt.ylabel("mean 95% CI length")
    plt.title("Fitted Bellman regression: Wald interval length")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bellman_b2_ci_length_vs_n.png", dpi=240)
    plt.savefig(outdir / "bellman_b2_ci_length_vs_n.pdf")
    plt.close()


def get_z(z_rows, n, coord, method):
    vals = np.array(
        [
            r["z"]
            for r in z_rows
            if int(r["n"]) == n and int(r["coord"]) == coord and r["method"] == method
        ],
        dtype=float,
    )
    return vals[np.isfinite(vals)]


def save_studentized_plots(z_rows, outdir, coord, max_n):
    z_dr = get_z(z_rows, max_n, coord, "dr")
    z_or = get_z(z_rows, max_n, coord, "oracle")
    bins = max(14, int(math.sqrt(max(len(z_dr), 1))))

    plt.figure(figsize=(7.0, 5.0))
    plt.hist(
        z_dr,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="OBiGrad / orthogonal DR",
    )
    plt.hist(z_or, bins=bins, density=True, alpha=0.35, label="Oracle DR")
    xs = np.linspace(-3.5, 3.5, 500)
    plt.plot(
        xs,
        np.exp(-0.5 * xs**2) / math.sqrt(2.0 * math.pi),
        linestyle="--",
        linewidth=2,
        label="standard normal density",
    )
    plt.xlabel("studentized error")
    plt.ylabel("density")
    plt.title(
        f"Fitted Bellman regression: studentized errors, coord {coord}, n={max_n}"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    base = f"bellman_b2_studentized_hist_coord{coord}_n{max_n}"
    plt.savefig(outdir / f"{base}.png", dpi=240)
    plt.savefig(outdir / f"{base}.pdf")
    plt.close()

    nd = statistics.NormalDist()
    plt.figure(figsize=(6.0, 6.0))
    all_min, all_max = 0.0, 0.0
    for method, marker, label in [
        ("dr", "o", "OBiGrad / orthogonal DR"),
        ("oracle", "^", "Oracle DR"),
    ]:
        z = np.sort(get_z(z_rows, max_n, coord, method))
        probs = (np.arange(1, z.size + 1) - 0.5) / z.size
        q = np.array([nd.inv_cdf(float(p)) for p in probs])
        all_min = min(all_min, float(np.min(q)), float(np.min(z)))
        all_max = max(all_max, float(np.max(q)), float(np.max(z)))
        plt.scatter(q, z, s=22, alpha=0.75, marker=marker, label=label)
    lo = min(-3.0, all_min) - 0.05
    hi = max(3.0, all_max) + 0.05
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.8, label="standard normal")
    plt.xlabel("theoretical standard-normal quantile")
    plt.ylabel("empirical studentized-error quantile")
    plt.title(f"Fitted Bellman regression: QQ diagnostic, coord {coord}, n={max_n}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    base = f"bellman_b2_studentized_qq_coord{coord}_n{max_n}"
    plt.savefig(outdir / f"{base}.png", dpi=240)
    plt.savefig(outdir / f"{base}.pdf")
    plt.close()


def run_experiment(dgp_cfg, learner_cfg, exp_cfg):
    outdir = Path(exp_cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    psi_true = true_gradient(dgp_cfg, exp_cfg.quadrature_nodes)
    A_true = true_A_matrix(dgp_cfg, exp_cfg.quadrature_nodes)

    rep_rows = []
    z_rows = []
    master = np.random.default_rng(exp_cfg.seed)
    print("Fitted Bellman regression B2: Wald calibration", flush=True)
    print("true gradient:", psi_true, flush=True)

    for n in exp_cfg.n_grid:
        print(f"n={n}, reps={exp_cfg.reps}", flush=True)
        for rep in range(exp_cfg.reps):
            seed = int(master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            data = simulate_data(n, dgp_cfg, rng)
            estimates, stats, nuisance = estimate_one_dataset(
                data, dgp_cfg, learner_cfg, exp_cfg.folds, rng, psi_true
            )
            rep_row = {"n": float(n), "rep": float(rep), "seed": float(seed)}
            rep_row.update(nuisance)
            for method in ["plugin", "dr", "oracle"]:
                err = stats[method]["err"]
                rep_row[f"{method}_l2_error"] = float(np.linalg.norm(err))
                rep_row[f"{method}_mean_coord_coverage95"] = float(
                    np.mean(stats[method]["cover"])
                )
                rep_row[f"{method}_mean_ci_length95"] = float(
                    np.mean(stats[method]["length"])
                )
                for k in range(4):
                    z_rows.append(
                        {
                            "n": float(n),
                            "rep": float(rep),
                            "seed": float(seed),
                            "method": method,
                            "coord": float(k),
                            "psi_true": float(psi_true[k]),
                            "estimate": float(estimates[method][k]),
                            "error": float(stats[method]["err"][k]),
                            "se": float(stats[method]["se"][k]),
                            "z": float(stats[method]["z"][k]),
                            "cover95": float(stats[method]["cover"][k]),
                            "ci_length95": float(stats[method]["length"][k]),
                        }
                    )
            rep_rows.append(rep_row)
        tmp = summarize(
            [r for r in rep_rows if int(r["n"]) == n],
            [r for r in z_rows if int(r["n"]) == n],
        )
        dr = next(r for r in tmp if r["method"] == "dr")
        oracle = next(r for r in tmp if r["method"] == "oracle")
        print(
            f"  coverage dr={dr['coverage']:.3f}, oracle={oracle['coverage']:.3f}; length dr={dr['ci_length_mean']:.4g}",
            flush=True,
        )

    summary = summarize(rep_rows, z_rows)
    save_coverage_plot(summary, outdir)
    save_length_plot(summary, outdir)
    max_n = max(exp_cfg.n_grid)
    save_studentized_plots(z_rows, outdir, exp_cfg.diagnostic_coord, max_n)
    write_latex_outputs(summary, outdir)

    print("Wrote", outdir, flush=True)
    return rep_rows, z_rows, summary


def write_latex_outputs(summary, outdir):
    rows = []
    z_rows = []
    for n in sorted({int(row["n"]) for row in summary}):
        by_method = {
            row["method"]: row for row in summary if int(row["n"]) == n
        }
        dr = by_method["dr"]
        rows.append(
            [
                str(n),
                format_pm(dr["coverage"], dr.get("coverage_se"), digits=3),
                format_pm(dr["ci_length_mean"], dr.get("ci_length_se"), digits=4),
                format_pm(dr["rmse"], dr.get("rmse_se"), digits=4),
                format_number(dr["z_abs_gt_196"], digits=3),
                format_number(dr["product_bias_proxy_mean"], digits=3),
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
            ]
        )
    write_latex_table(
        outdir / "table_bellman_b2_wald.tex",
        "Projected Bellman Wald calibration.",
        "tab:generated-bellman-b2-wald",
        [r"$n$", "Coverage", "Length", "RMSE", r"$|Z|>1.96$", "Product bias"],
        rows,
    )
    write_latex_table(
        outdir / "table_bellman_b2_studentized.tex",
        "Studentized-error diagnostics for projected Bellman Wald intervals.",
        "tab:generated-bellman-b2-studentized",
        [r"$n$", "Mean", "SD", r"2.5\%", "Median", r"97.5\%"],
        z_rows,
    )


def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment B2: fitted Bellman regression inference diagnostics."
    )
    p.add_argument("--n-grid", type=parse_n_grid, default=ExperimentConfig.n_grid)
    p.add_argument("--reps", type=int, default=ExperimentConfig.reps)
    p.add_argument("--folds", type=int, default=ExperimentConfig.folds)
    p.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    p.add_argument("--outdir", type=str, default=ExperimentConfig.outdir)
    p.add_argument("--ridge-alpha", type=float, default=LearnerConfig.ridge_alpha)
    p.add_argument("--gamma", type=float, default=DGPConfig.gamma)
    p.add_argument("--sigma-y", type=float, default=DGPConfig.sigma_y)
    p.add_argument(
        "--quadrature-nodes", type=int, default=ExperimentConfig.quadrature_nodes
    )
    p.add_argument(
        "--coord",
        "--diagnostic-coord",
        dest="diagnostic_coord",
        type=int,
        default=ExperimentConfig.diagnostic_coord,
    )
    return p


def main():
    args = build_parser().parse_args()
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if not (0 <= args.diagnostic_coord < 4):
        raise ValueError("--diagnostic-coord must be in {0,1,2,3}")
    dgp_cfg = DGPConfig(gamma=args.gamma, sigma_y=args.sigma_y)
    learner_cfg = LearnerConfig(ridge_alpha=args.ridge_alpha)
    exp_cfg = ExperimentConfig(
        n_grid=args.n_grid,
        reps=args.reps,
        folds=args.folds,
        seed=args.seed,
        outdir=args.outdir,
        quadrature_nodes=args.quadrature_nodes,
        diagnostic_coord=args.diagnostic_coord,
    )
    run_experiment(dgp_cfg, learner_cfg, exp_cfg)


if __name__ == "__main__":
    main()
