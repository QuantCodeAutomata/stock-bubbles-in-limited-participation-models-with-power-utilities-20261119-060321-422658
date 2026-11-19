"""Experiment 2: Lambda is a strict local martingale under Q_2 (Lemma 3.1).

Part A applies the Delbaen-Shirakawa (2002) integral test (Eq. 28) to the
autonomous diffusion for Lambda_t = ((1-Y_t)/Y_t)^gamma under Q_2 (Eq. 27)
and checks that int^infty lambda/a(lambda)^2 dlambda is finite.

Part B simulates the state process Y and the dividend D under P and under
Q_2, computes L_{1,t} = S^(0)_t Z1_t / Z1_0 (Eq. 25) and Lambda_t/Lambda_0,
and checks E[L_{1,t}] < 1 and E^{Q_2}[Lambda_t/Lambda_0] < 1 for t > 0, the
Monte Carlo counterpart of the strict local martingale property.

Run: python -m exp.exp2
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import generate_synthetic as gs
from src import config as cfg
from src import equilibrium as eq
from src import martingale_tests as mt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_integral_test(h_func) -> dict:
    """Part A: Delbaen-Shirakawa integral test, Eq. (28)."""
    value, err = mt.integral_test(cfg.GAMMA, cfg.SIGMA_D, h_func, cfg.LAMBDA_CUTOFF, cfg.LAMBDA_UPPER)
    lam_grid = np.geomspace(cfg.LAMBDA_CUTOFF, cfg.LAMBDA_CUTOFF * 1e6, 50)
    a_over_lambda = mt.asymptotic_a_over_lambda(cfg.GAMMA, cfg.SIGMA_D, h_func, lam_grid)
    asymptotic_limit = float(a_over_lambda[-1])
    tail_exponent = mt.integrand_tail_exponent(cfg.GAMMA)
    return {
        "integral_value": float(value),
        "integral_abs_error": float(err),
        "is_finite": bool(np.isfinite(value)),
        "asymptotic_a_over_lambda_pow": asymptotic_limit,
        "sigma_D": cfg.SIGMA_D,
        "tail_exponent": tail_exponent,
        "lower_cutoff": cfg.LAMBDA_CUTOFF,
    }


def run_monte_carlo(h_func, r_interp, delta2_interp, n_steps: int, n_paths: int) -> dict:
    """Part B: E[L_{1,t}] under P and E^{Q_2}[Lambda_t/Lambda_0] under Q_2."""
    nu_func = lambda y: eq.nu(y, cfg.GAMMA, cfg.SIGMA_D, h_func)

    t0 = time.time()
    p_results, Z1_0 = gs.simulate_P_paths(
        Y0=cfg.Y0, D0=cfg.D0, gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, beta=cfg.BETA,
        h_func=h_func, r_func=r_interp, delta2_func=delta2_interp,
        T=cfg.T_HORIZON, n_steps=n_steps, n_paths=n_paths, checkpoints=cfg.CHECKPOINTS, seed=cfg.MC_SEED,
        boundary_eps=cfg.MC_BOUNDARY_EPS,
    )
    p_elapsed = time.time() - t0

    t0 = time.time()
    q2_results = gs.simulate_Q2_lambda_paths(
        Y0=cfg.Y0, gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, h_func=h_func, nu_func=nu_func,
        T=cfg.T_HORIZON, n_steps=n_steps, n_paths=n_paths, checkpoints=cfg.CHECKPOINTS, seed=cfg.MC_SEED + 1,
        boundary_eps=cfg.MC_BOUNDARY_EPS,
    )
    q2_elapsed = time.time() - t0

    checkpoints = sorted(p_results.keys())
    E_L1 = []
    se_L1 = []
    E_Lambda_ratio = []
    se_Lambda_ratio = []
    for t in checkpoints:
        L1_t = p_results[t]["S0"] * p_results[t]["Z1"] / Z1_0
        E_L1.append(float(np.mean(L1_t)))
        se_L1.append(float(np.std(L1_t) / np.sqrt(n_paths)))
        lam_ratio = q2_results[t]
        E_Lambda_ratio.append(float(np.mean(lam_ratio)))
        se_Lambda_ratio.append(float(np.std(lam_ratio) / np.sqrt(n_paths)))

    return {
        "checkpoints": list(checkpoints),
        "E_L1": E_L1,
        "se_L1": se_L1,
        "E_Lambda_ratio_Q2": E_Lambda_ratio,
        "se_Lambda_ratio_Q2": se_Lambda_ratio,
        "n_steps": n_steps,
        "n_paths": n_paths,
        "p_elapsed_sec": p_elapsed,
        "q2_elapsed_sec": q2_elapsed,
    }


def main(full: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_grid = cfg.N_GRID_PAPER if full else cfg.N_GRID_RUN
    n_steps = cfg.N_STEPS_PAPER if full else cfg.N_STEPS_RUN
    n_paths = cfg.N_PATHS_PAPER if full else cfg.N_PATHS_RUN

    result = eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=cfg.EPS_Y, n_grid=n_grid, h_power=cfg.H_POWER, tol=cfg.BVP_TOL, max_nodes=cfg.BVP_MAX_NODES,
    )
    h_func = result["h_func"]

    integral_test_result = run_integral_test(h_func)
    print(
        f"Integral test (Eq. 28): I = {integral_test_result['integral_value']:.6e} "
        f"(+/- {integral_test_result['integral_abs_error']:.2e}), finite={integral_test_result['is_finite']}"
    )

    mc_result = run_monte_carlo(h_func, result["r_interp"], result["delta2_interp"], n_steps, n_paths)
    for t, e_l1, e_lam in zip(mc_result["checkpoints"], mc_result["E_L1"], mc_result["E_Lambda_ratio_Q2"]):
        print(f"t={t:5.2f}  E[L1,t]={e_l1:.6f}  E^Q2[Lambda_t/Lambda_0]={e_lam:.6f}")

    strictly_below_one_L1 = all(e < 1.0 for e in mc_result["E_L1"])
    strictly_below_one_Lambda = all(e < 1.0 for e in mc_result["E_Lambda_ratio_Q2"])
    decreasing_L1 = all(
        mc_result["E_L1"][i + 1] <= mc_result["E_L1"][i] + 3 * mc_result["se_L1"][i]
        for i in range(len(mc_result["E_L1"]) - 1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(mc_result["checkpoints"], mc_result["E_L1"], yerr=mc_result["se_L1"], marker="o", capsize=3)
    axes[0].axhline(1.0, color="r", linestyle="--", label="1 (true-martingale benchmark)")
    axes[0].set_title(r"$E[L_{1,t}]$ under $P$ (Eq. 25)")
    axes[0].set_xlabel("t (years)")
    axes[0].legend()
    axes[1].errorbar(
        mc_result["checkpoints"], mc_result["E_Lambda_ratio_Q2"], yerr=mc_result["se_Lambda_ratio_Q2"],
        marker="o", capsize=3, color="tab:green",
    )
    axes[1].axhline(1.0, color="r", linestyle="--", label="1 (true-martingale benchmark)")
    axes[1].set_title(r"$E^{Q_2}[\Lambda_t/\Lambda_0]$ (Eq. 26-27)")
    axes[1].set_xlabel("t (years)")
    axes[1].legend()
    fig.suptitle("Experiment 2: Monte Carlo evidence for the strict local martingale property")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_strict_local_martingale.png", dpi=150)
    plt.close(fig)

    metrics = {
        "parameters": {
            "gamma": cfg.GAMMA, "sigma_D": cfg.SIGMA_D, "mu_D": cfg.MU_D, "beta": cfg.BETA,
            "D0": cfg.D0, "Y0": cfg.Y0, "varrho": cfg.RHO, "h_power": cfg.H_POWER,
            "T_horizon": cfg.T_HORIZON, "n_steps": n_steps, "n_paths": n_paths, "seed": cfg.MC_SEED,
        },
        "integral_test": integral_test_result,
        "monte_carlo": mc_result,
        "verdict": {
            "E_L1_strictly_below_1": strictly_below_one_L1,
            "E_Lambda_ratio_Q2_strictly_below_1": strictly_below_one_Lambda,
            "E_L1_decreasing_within_mc_noise": decreasing_L1,
            "integral_test_finite": integral_test_result["is_finite"],
        },
    }
    with open(RESULTS_DIR / "exp2_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Saved results/exp2_strict_local_martingale.png and results/exp2_metrics.json")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
