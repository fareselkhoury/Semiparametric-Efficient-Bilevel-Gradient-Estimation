#!/usr/bin/env python3
"""
Experiment B4: projected Bellman regression KBO regularization-bias decomposition.

Compares OBiGrad, Oracle DR, and KBO-RFF fixed-lambda gradients against the
unregularized projected Bellman gradient target Psi_0.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_reporting import format_lambda, format_number, format_pm, write_latex_table


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
class OBiGradConfig:
    ridge: float = 5e-1
    folds: int = 2
    learner: str = "bellman_basis"  # bellman_basis or rff
    rff_dim: int = 128
    rff_gamma: float = 0.75


@dataclass(frozen=True)
class KBOConfig:
    rff_dim: int = 256
    kernel_gamma: float = 0.35
    lambdas: Tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 600
    reps: int = 200
    pop_n: int = 12000
    seed: int = 20260425
    quadrature_nodes: int = 240
    outdir: str = "results/Bellman/b4"


def parse_lambdas(s):
    vals = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("positive lambda grid required")
    return vals


def phi(s):
    s = np.asarray(s, float)
    return np.column_stack([np.sin(s), np.cos(s), s, s * s])


def cond_phi_mean(mu, sig):
    att = math.exp(-0.5 * sig * sig)
    return np.column_stack(
        [att * np.sin(mu), att * np.cos(mu), mu, mu * mu + sig * sig]
    )


def base_reward_mean(s, a):
    return np.sin(s) + 0.5 * a + 0.25 * s * a


def simulate(n, cfg, rng):
    s = rng.normal(size=n)
    a = rng.binomial(1, cfg.p_action, size=n).astype(float)
    sp = cfg.rho * s + cfg.tau * a + rng.normal(scale=cfg.sigma_s, size=n)
    r = base_reward_mean(s, a) + rng.normal(scale=cfg.sigma_r, size=n)
    phip = phi(sp)
    omstar = np.asarray(cfg.omega_star, float)
    y = r + cfg.gamma * (phip @ omstar) + rng.normal(scale=cfg.sigma_y, size=n)
    return {
        "S": s,
        "A": a,
        "Sp": sp,
        "R": r,
        "PhiSp": phip,
        "X": np.column_stack([s, a]),
        "Y": y,
    }


def true_nuis(data, omega, cfg):
    mu = cfg.rho * data["S"] + cfg.tau * data["A"]
    j = cfg.gamma * cond_phi_mean(mu, cfg.sigma_s)
    base = base_reward_mean(data["S"], data["A"])
    omstar = np.asarray(cfg.omega_star, float)
    return base + j @ omega, j, base + j @ omstar


def eval_omega(cfg):
    omstar = np.asarray(cfg.omega_star, float)
    v = np.array([1.0, -0.5, 0.35, -0.25])
    return omstar + cfg.omega_eval_shift * v / np.linalg.norm(v)


def true_A(cfg, nodes):
    xs, ws = np.polynomial.hermite.hermgauss(nodes)
    s = math.sqrt(2.0) * xs
    w = ws / math.sqrt(math.pi)
    A = np.zeros((4, 4))
    for a, p in [(0.0, 1.0 - cfg.p_action), (1.0, cfg.p_action)]:
        mu = cfg.rho * s + cfg.tau * a
        j = cfg.gamma * cond_phi_mean(mu, cfg.sigma_s)
        A += p * ((j.T * w) @ j)
    return A


def true_grad(omega, cfg, nodes):
    return true_A(cfg, nodes) @ (omega - np.asarray(cfg.omega_star, float))


class BasisReg:
    def __init__(self, cfg, dgp, seed=0):
        self.cfg, self.dgp, self.seed = cfg, dgp, int(seed)
        self.beta = None
        self.mean = None
        self.scale = None
        self.W = None
        self.b = None

    def basis(self, x, fit=False):
        x = np.asarray(x, float)
        s = x[:, 0]
        a = x[:, 1]
        if self.cfg.learner == "bellman_basis":
            cols = [
                np.ones_like(s),
                s,
                a,
                s * a,
                s * s,
                s * s * a,
                s**3,
                a * s**3,
            ]
            for freq in (0.5, 0.75, 1.0, 1.5, 2.0):
                sf = np.sin(freq * s)
                cf = np.cos(freq * s)
                cols += [sf, cf, a * sf, a * cf]
            for knot in (-2.0, -1.0, 0.0, 1.0, 2.0):
                bump = np.exp(-0.5 * ((s - knot) / 0.8) ** 2)
                cols += [bump, a * bump]
            return np.column_stack(cols)
        if fit:
            self.mean = x.mean(0, keepdims=True)
            self.scale = x.std(0, keepdims=True)
            self.scale = np.where(self.scale < 1e-12, 1, self.scale)
            xs = (x - self.mean) / self.scale
            rng = np.random.default_rng(self.seed)
            self.W = rng.normal(
                scale=math.sqrt(2 * self.cfg.rff_gamma),
                size=(x.shape[1], self.cfg.rff_dim),
            )
            self.b = rng.uniform(0, 2 * math.pi, size=self.cfg.rff_dim)
        else:
            xs = (x - self.mean) / self.scale
        z = math.sqrt(2 / self.cfg.rff_dim) * np.cos(xs @ self.W + self.b)
        return np.column_stack([np.ones(x.shape[0]), z])

    def fit(self, x, y):
        y = np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
        Z = self.basis(x, fit=True)
        G = Z.T @ Z
        rhs = Z.T @ y
        pen = self.cfg.ridge * np.eye(G.shape[0])
        pen[0, 0] = 0
        self.beta = np.linalg.solve(G + pen, rhs)
        return self

    def predict(self, x):
        pred = self.basis(x, fit=False) @ self.beta
        return pred[:, 0] if pred.shape[1] == 1 else pred


def folds(n, k, rng):
    return [z.astype(int) for z in np.array_split(rng.permutation(n), k)]


def targets(data, omega, cfg):
    g = data["R"] + cfg.gamma * (data["PhiSp"] @ omega)
    dg = cfg.gamma * data["PhiSp"]
    return np.column_stack([g, dg, data["Y"]])


def autodml(data, omega, psi0, dgp, acfg, rng, seed):
    n = len(data["Y"])
    d = len(omega)
    allidx = np.arange(n)
    g = data["R"] + dgp.gamma * (data["PhiSp"] @ omega)
    dg = dgp.gamma * data["PhiSp"]
    h0, j0, m0 = true_nuis(data, omega, dgp)
    sdr = np.zeros((n, d))
    sor = np.zeros((n, d))
    splug = np.zeros((n, d))
    hhat_all = np.zeros(n)
    jhat_all = np.zeros((n, d))
    mhat_all = np.zeros(n)
    for fid, te in enumerate(folds(n, acfg.folds, rng)):
        tr = np.setdiff1d(allidx, te, assume_unique=False)
        model = BasisReg(acfg, dgp, seed + 1009 * (fid + 1)).fit(
            data["X"][tr], targets({k: v[tr] for k, v in data.items()}, omega, dgp)
        )
        pred = model.predict(data["X"][te])
        hhat = pred[:, 0]
        jhat = pred[:, 1 : 1 + d]
        mhat = pred[:, 1 + d]
        hhat_all[te] = hhat
        jhat_all[te] = jhat
        mhat_all[te] = mhat
        splug[te] = jhat * (hhat - data["Y"][te])[:, None]
        sdr[te] = (
            jhat * (g[te] - data["Y"][te])[:, None]
            + (dg[te] - jhat) * (hhat - mhat)[:, None]
        )
        sor[te] = (
            j0[te] * (g[te] - data["Y"][te])[:, None]
            + (dg[te] - j0[te]) * (h0[te] - m0[te])[:, None]
        )
    grad_dr = sdr.mean(0)
    grad_or = sor.mean(0)
    grad_plugin = splug.mean(0)
    diag = {
        "autodml_error_to_psi0": float(np.linalg.norm(grad_dr - psi0)),
        "oracle_dr_error_to_psi0": float(np.linalg.norm(grad_or - psi0)),
        "plugin_error_to_psi0": float(np.linalg.norm(grad_plugin - psi0)),
        "err_h": float(np.sqrt(np.mean((hhat_all - h0) ** 2))),
        "err_j": float(np.sqrt(np.mean((jhat_all - j0) ** 2))),
        "err_m": float(np.sqrt(np.mean((mhat_all - m0) ** 2))),
    }
    diag["product_bias_proxy"] = diag["err_j"] * (diag["err_h"] + diag["err_m"])
    return grad_dr, grad_or, grad_plugin, diag


class RFFMap:
    def __init__(self, dim, D, gamma, seed):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=math.sqrt(2 * gamma), size=(dim, D))
        self.b = rng.uniform(0, 2 * math.pi, size=D)
        self.D = D

    def transform(self, x):
        return math.sqrt(2 / self.D) * np.cos(np.asarray(x, float) @ self.W + self.b)


def ridge_beta(Z, Y, lam):
    n = Z.shape[0]
    G = Z.T @ Z
    rhs = Z.T @ Y
    return np.linalg.solve(G + n * lam * np.eye(G.shape[0]), rhs)


def kbo_grad(inner, outer, omega, lam, rff, cfg):
    Z1 = rff.transform(inner["X"])
    Z2 = rff.transform(outer["X"])
    Ytar = np.column_stack([inner["R"], cfg.gamma * inner["PhiSp"]])
    beta = ridge_beta(Z1, Ytar, lam)
    pred = Z2 @ beta
    rhat = pred[:, 0]
    C = pred[:, 1:]
    return C.T @ (rhat + C @ omega - outer["Y"]) / len(outer["Y"])


def kbo_pop_targets(pin, pout, omega, lambdas, rff, cfg):
    Z1 = rff.transform(pin["X"])
    Z2 = rff.transform(pout["X"])
    _, jin, _ = true_nuis(pin, omega, cfg)
    base = base_reward_mean(pin["S"], pin["A"])
    _, _, mout = true_nuis(pout, omega, cfg)
    Ytar = np.column_stack([base, jin])
    out = {}
    for lam in lambdas:
        beta = ridge_beta(Z1, Ytar, lam)
        pred = Z2 @ beta
        rhat = pred[:, 0]
        C = pred[:, 1:]
        out[float(lam)] = C.T @ (rhat + C @ omega - mout) / len(mout)
    return out


def rmse_se(vals):
    vals = np.asarray(list(vals), float)
    sq = vals**2
    rmse = math.sqrt(float(np.mean(sq)))
    if len(vals) < 2 or rmse == 0:
        return rmse, 0.0
    return rmse, float(np.std(sq, ddof=1) / math.sqrt(len(sq)) / (2 * rmse))


def summarize(rows, lambdas):
    out = []
    for lam in lambdas:
        sub = [r for r in rows if abs(r["lambda"] - lam) < 1e-20]
        total, se_total = rmse_se(r["kbo_total_error_to_psi0"] for r in sub)
        est, se_est = rmse_se(r["kbo_estimation_error_to_psilambda"] for r in sub)
        auto, se_auto = rmse_se(r["autodml_error_to_psi0"] for r in sub)
        oracle, se_or = rmse_se(r["oracle_dr_error_to_psi0"] for r in sub)
        plug, se_plug = rmse_se(r["plugin_error_to_psi0"] for r in sub)
        out.append(
            {
                "lambda": float(lam),
                "kbo_total_rmse_l2": total,
                "kbo_total_rmse_se": se_total,
                "kbo_estimation_rmse_l2": est,
                "kbo_estimation_rmse_se": se_est,
                "regularization_bias_l2": sub[0][
                    "regularization_bias_psilambda_to_psi0"
                ],
                "autodml_rmse_l2": auto,
                "autodml_rmse_se": se_auto,
                "oracle_dr_rmse_l2": oracle,
                "oracle_dr_rmse_se": se_or,
                "plugin_rmse_l2": plug,
                "plugin_rmse_se": se_plug,
                "product_bias_proxy_mean": float(
                    np.mean([r["product_bias_proxy"] for r in sub])
                ),
            }
        )
    return out


def plot_all(summary, outdir):
    lam = np.array([r["lambda"] for r in summary])
    total = np.array([r["kbo_total_rmse_l2"] for r in summary])
    total_se = np.array([r["kbo_total_rmse_se"] for r in summary])
    est = np.array([r["kbo_estimation_rmse_l2"] for r in summary])
    est_se = np.array([r["kbo_estimation_rmse_se"] for r in summary])
    bias = np.array([r["regularization_bias_l2"] for r in summary])
    auto = np.array([r["autodml_rmse_l2"] for r in summary])
    auto_se = np.array([r["autodml_rmse_se"] for r in summary])
    oracle = np.array([r["oracle_dr_rmse_l2"] for r in summary])
    oracle_se = np.array([r["oracle_dr_rmse_se"] for r in summary])
    plugin = np.array([r["plugin_rmse_l2"] for r in summary])
    plugin_se = np.array([r["plugin_rmse_se"] for r in summary])
    for name, full in [
        ("bellman_b4_kbo_bias_decomposition.png", True),
        ("bellman_b4_kbo_total_error.png", False),
    ]:
        plt.figure(figsize=(7.3, 5.1))
        plt.errorbar(
            lam,
            total,
            yerr=1.96 * total_se,
            marker="o",
            color="#1f4e79",
            capsize=3,
            linewidth=2,
            label="KBO total error to $\\Psi_0$",
        )
        if full:
            plt.fill_between(
                lam,
                np.maximum(est - 1.96 * est_se, 1e-12),
                est + 1.96 * est_se,
                color="#7b8fa1",
                alpha=0.22,
                label="KBO component: estimation error",
            )
            plt.plot(
                lam,
                np.maximum(est, 1e-12),
                color="#7b8fa1",
                linestyle=":",
                linewidth=1.7,
                alpha=0.9,
            )
            plt.fill_between(
                lam,
                np.maximum(bias, 1e-12),
                np.maximum(bias, 1e-12),
                color="#d98c27",
                alpha=0.20,
                label="KBO component: regularization bias",
            )
            plt.plot(
                lam,
                np.maximum(bias, 1e-12),
                color="#d98c27",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
            )
        am = float(np.mean(auto))
        ase = 1.96 * float(math.sqrt(np.mean(auto_se**2)))
        om = float(np.mean(oracle))
        ose = 1.96 * float(math.sqrt(np.mean(oracle_se**2)))
        plt.axhline(
            am,
            linestyle="--",
            linewidth=2,
            color="#2c7a4b",
            label="OBiGrad to $\\Psi_0$",
        )
        plt.fill_between(lam, max(am - ase, 1e-12), am + ase, alpha=0.12)
        pm = float(np.mean(plugin))
        pse = 1.96 * float(math.sqrt(np.mean(plugin_se**2)))
        plt.axhline(
            pm,
            linestyle="-.",
            linewidth=2,
            color="#8f3f2f",
            label="Plug-in to $\\Psi_0$",
        )
        plt.fill_between(lam, max(pm - pse, 1e-12), pm + pse, alpha=0.10)
        plt.axhline(
            om,
            linestyle=":",
            linewidth=2,
            color="#5f4b8b",
            label="Oracle DR to $\\Psi_0$",
        )
        plt.fill_between(lam, max(om - ose, 1e-12), om + ose, alpha=0.10)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both", alpha=0.25)
        plt.xlabel(r"KBO regularization $\lambda$")
        plt.ylabel("gradient error")
        plt.title("Projected Bellman regression: KBO regularization bias")
        plt.legend(fontsize=8.5)
        plt.tight_layout()
        plt.savefig(outdir / name, dpi=240)
        plt.close()


def run(dgp, acfg, kcfg, exp):
    outdir = Path(exp.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    omega = eval_omega(dgp)
    psi0 = true_grad(omega, dgp, exp.quadrature_nodes)
    rff = RFFMap(2, kcfg.rff_dim, kcfg.kernel_gamma, exp.seed + 999)
    rngpop = np.random.default_rng(exp.seed + 99991)
    pin = simulate(exp.pop_n, dgp, rngpop)
    pout = simulate(exp.pop_n, dgp, rngpop)
    pop = kbo_pop_targets(pin, pout, omega, kcfg.lambdas, rff, dgp)
    pop_rows = []
    for lam, g in pop.items():
        row = {"lambda": lam, "regularization_bias_l2": float(np.linalg.norm(g - psi0))}
        row.update({f"psilambda_{k}": float(g[k]) for k in range(len(g))})
        pop_rows.append(row)
    print("Running B4 projected Bellman KBO regularization experiment", flush=True)
    print(
        f"n={exp.n}, reps={exp.reps}, pop_n={exp.pop_n}, lambdas={kcfg.lambdas}",
        flush=True,
    )
    print(f"psi0={np.array2string(psi0, precision=4)}", flush=True)
    rng = np.random.default_rng(exp.seed)
    rows = []
    for rep in range(exp.reps):
        if rep % max(1, exp.reps // 10) == 0:
            print(f"rep {rep}/{exp.reps}", flush=True)
        seed = int(rng.integers(0, 2**31 - 1))
        rr = np.random.default_rng(seed)
        inner = simulate(exp.n, dgp, rr)
        outer = simulate(exp.n, dgp, rr)
        pooled = {
            k: (
                np.concatenate([inner[k], outer[k]])
                if inner[k].ndim == 1
                else np.vstack([inner[k], outer[k]])
            )
            for k in inner
        }
        gauto, gor, gplug, diag = autodml(
            pooled, omega, psi0, dgp, acfg, rr, seed + 5000
        )
        auto_err = float(np.linalg.norm(gauto - psi0))
        oracle_err = float(np.linalg.norm(gor - psi0))
        plugin_err = float(np.linalg.norm(gplug - psi0))
        for lam in kcfg.lambdas:
            gk = kbo_grad(inner, outer, omega, float(lam), rff, dgp)
            gp = pop[float(lam)]
            row = {
                "rep": float(rep),
                "seed": float(seed),
                "n_inner": float(exp.n),
                "n_total_autodml": float(2 * exp.n),
                "lambda": float(lam),
                "kbo_total_error_to_psi0": float(np.linalg.norm(gk - psi0)),
                "kbo_estimation_error_to_psilambda": float(np.linalg.norm(gk - gp)),
                "regularization_bias_psilambda_to_psi0": float(
                    np.linalg.norm(gp - psi0)
                ),
                "autodml_error_to_psi0": auto_err,
                "oracle_dr_error_to_psi0": oracle_err,
                "plugin_error_to_psi0": plugin_err,
                **diag,
            }
            for k in range(len(psi0)):
                row[f"psi0_{k}"] = float(psi0[k])
                row[f"kbo_grad_{k}"] = float(gk[k])
                row[f"psilambda_{k}"] = float(gp[k])
                row[f"autodml_grad_{k}"] = float(gauto[k])
                row[f"oracle_dr_grad_{k}"] = float(gor[k])
                row[f"plugin_grad_{k}"] = float(gplug[k])
            rows.append(row)
    summary = summarize(rows, kcfg.lambdas)
    plot_all(summary, outdir)
    write_latex_outputs(summary, outdir)
    return summary


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
                format_pm(row["plugin_rmse_l2"], row.get("plugin_rmse_se")),
            ]
        )
        appendix_rows.append(
            [
                format_lambda(row["lambda"]),
                format_pm(row["oracle_dr_rmse_l2"], row.get("oracle_dr_rmse_se")),
                format_number(row["product_bias_proxy_mean"], digits=4),
            ]
        )
    write_latex_table(
        outdir / "table_bellman_b4_kbo.tex",
        r"Projected Bellman KBO gradient-error decomposition. Parentheses report Monte Carlo 95\% error bars.",
        "tab:generated-bellman-b4-kbo",
        [r"$\lambda$", "KBO total", "Reg. bias", "KBO estimation", "OBiGrad", "Plug-in"],
        rows,
    )
    write_latex_table(
        outdir / "table_bellman_b4_appendix.tex",
        "Additional projected Bellman KBO diagnostics.",
        "tab:generated-bellman-b4-appendix",
        [r"$\lambda$", "Oracle DR", "Product bias"],
        appendix_rows,
    )


def build_parser():
    p = argparse.ArgumentParser(
        description="B4: projected Bellman KBO regularization-bias decomposition."
    )
    p.add_argument("--n", type=int, default=ExperimentConfig.n)
    p.add_argument("--reps", type=int, default=ExperimentConfig.reps)
    p.add_argument("--pop-n", type=int, default=ExperimentConfig.pop_n)
    p.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    p.add_argument("--outdir", type=str, default=ExperimentConfig.outdir)
    p.add_argument("--lambdas", type=parse_lambdas, default=KBOConfig.lambdas)
    p.add_argument("--kbo-rff-dim", type=int, default=KBOConfig.rff_dim)
    p.add_argument("--kbo-kernel-gamma", type=float, default=KBOConfig.kernel_gamma)
    p.add_argument(
        "--obigrad-learner",
        "--autodml-learner",
        dest="obigrad_learner",
        choices=["bellman_basis", "rff"],
        default=OBiGradConfig.learner,
    )
    p.add_argument(
        "--obigrad-ridge",
        "--autodml-ridge",
        dest="obigrad_ridge",
        type=float,
        default=OBiGradConfig.ridge,
    )
    p.add_argument(
        "--obigrad-rff-dim",
        "--autodml-rff-dim",
        dest="obigrad_rff_dim",
        type=int,
        default=OBiGradConfig.rff_dim,
    )
    p.add_argument("--gamma", type=float, default=DGPConfig.gamma)
    p.add_argument("--rho", type=float, default=DGPConfig.rho)
    p.add_argument("--tau", type=float, default=DGPConfig.tau)
    p.add_argument("--sigma-s", type=float, default=DGPConfig.sigma_s)
    p.add_argument("--sigma-r", type=float, default=DGPConfig.sigma_r)
    p.add_argument("--sigma-y", type=float, default=DGPConfig.sigma_y)
    return p


def main():
    a = build_parser().parse_args()
    dgp = DGPConfig(
        rho=a.rho,
        tau=a.tau,
        sigma_s=a.sigma_s,
        sigma_r=a.sigma_r,
        sigma_y=a.sigma_y,
        gamma=a.gamma,
    )
    acfg = OBiGradConfig(
        learner=a.obigrad_learner, ridge=a.obigrad_ridge, rff_dim=a.obigrad_rff_dim
    )
    kcfg = KBOConfig(
        rff_dim=a.kbo_rff_dim, kernel_gamma=a.kbo_kernel_gamma, lambdas=a.lambdas
    )
    exp = ExperimentConfig(
        n=a.n, reps=a.reps, pop_n=a.pop_n, seed=a.seed, outdir=a.outdir
    )
    summary = run(dgp, acfg, kcfg, exp)
    print("\nSummary:", flush=True)
    for r in summary:
        print(
            f"lambda={r['lambda']:.1e} | "
            f"KBO total={r['kbo_total_rmse_l2']:.4g} | "
            f"KBO est={r['kbo_estimation_rmse_l2']:.4g} | "
            f"reg bias={r['regularization_bias_l2']:.4g} | "
            f"OBiGrad={r['autodml_rmse_l2']:.4g} | "
            f"Plug-in={r['plugin_rmse_l2']:.4g} | "
            f"Oracle={r['oracle_dr_rmse_l2']:.4g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
