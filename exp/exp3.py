"""Experiment 3: the equilibrium stock price is a bubble (Theorem 2.2).

Part A verifies that the finite-variation drift of the deflated cum-dividend
price process, sigma_S(y)(kappa(y) - eta), vanishes identically in y only
for eta = kappa(Y_t): no state-price deflator other than Z^{kappa} even
produces a local martingale (Section 3.3, p. 8).

Part B simulates Y, D under P, computes the stock price S_t (Eq. 20) and the
process M_t = Z1_t S_t + int_0^t Z1_s D_s ds, and checks E[M_t] < M_0 for
t > 0: even with eta = kappa(Y_t), the deflated cum-dividend price is only a
strict local martingale.

Part C checks the lower-bound inequality M_t/Z1_0 >= c_T * L_{1,t} on [0,T]
used in the proof (Section 3.3, p. 8-9), from which the strict local
martingale property of L_1 (Experiment 2) implies the one of M.

Run: python -m exp.exp3
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
from src import bubble
from src import config as cfg
from src import equilibrium as eq

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_drift_check(result: dict) -> dict:
    """Part A: sigma_S(y)*(kappa(y)-eta) on a grid, for eta = kappa(y) + offset."""
    y_grid = result["y_grid"]
    sigma_S_vals = result["sigma_S"]
    kappa_vals = result["kappa"]

    rows = []
    for offset in cfg.ETA_OFFSETS:
        eta = kappa_vals + offset
        drift = bubble.deflated_price_drift(sigma_S_vals, kappa_vals, eta)
        rows.append({
            "eta_minus_kappa_offset": offset,
            "max_abs_drift": float(np.max(np.abs(drift))),
            "min_abs_drift": float(np.min(np.abs(drift))),
        })
    only_zero_at_kappa = rows[[r["eta_minus_kappa_offset"] for r in rows].index(0.0)]["max_abs_drift"] < 1e-10
    all_others_nonzero = all(r["min_abs_drift"] > 1e-8 for r in rows if r["eta_minus_kappa_offset"] != 0.0)
    return {"rows": rows, "only_eta_equals_kappa_annihilates_drift": bool(only_zero_at_kappa and all_others_nonzero)}


def run_monte_carlo(result: dict, n_steps: int, n_paths: int) -> dict:
    """Parts B & C: simulate M_t, L_{1,t}, X-bar_{2,t} and check E[M_t] < M_0
    and the lower bound M_t/Z1_0 >= c_T * L_{1,t}."""
    h_func = result["h_func"]
    v_interp = result["v_interp"]
    g_interp = result["g_interp"]

    t0 = time.time()
    p_results, Z1_0 = gs.simulate_P_paths(
        Y0=cfg.Y0, D0=cfg.D0, gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, beta=cfg.BETA,
        h_func=h_func, r_func=result["r_interp"], delta2_func=result["delta2_interp"],
        T=cfg.T_HORIZON, n_steps=n_steps, n_paths=n_paths, checkpoints=cfg.CHECKPOINTS, seed=cfg.MC_SEED,
        boundary_eps=cfg.MC_BOUNDARY_EPS,
    )
    elapsed = time.time() - t0

    S0_val = cfg.D0 * (float(v_interp(cfg.Y0)) + float(g_interp(cfg.Y0)) * (1 - cfg.Y0) ** cfg.GAMMA)
    M0 = Z1_0 * S0_val
    X2_bar_0 = cfg.D0 * float(g_interp(cfg.Y0)) * (1 - cfg.Y0) ** cfg.GAMMA

    checkpoints = sorted(p_results.keys())
    E_M = []
    se_M = []
    bound_holds_fraction = []
    for t in checkpoints:
        data = p_results[t]
        S_t = bubble.stock_price(data["D"], data["Y"], v_interp, g_interp, cfg.GAMMA)
        M_t = bubble.M_process(data["Z1"], S_t, data["int_Z1_D"])
        E_M.append(float(np.mean(M_t)))
        se_M.append(float(np.std(M_t) / np.sqrt(n_paths)))

        L1_t = data["S0"] * data["Z1"] / Z1_0
        c_T = bubble.c_T_bound(X2_bar_0, cfg.T_HORIZON, cfg.G0)
        bound_satisfied = (M_t / Z1_0) >= (c_T * L1_t - 1e-9)
        bound_holds_fraction.append(float(np.mean(bound_satisfied)))

    return {
        "checkpoints": list(checkpoints),
        "E_M": E_M,
        "se_M": se_M,
        "M0": M0,
        "X2_bar_0": X2_bar_0,
        "c_T": bubble.c_T_bound(X2_bar_0, cfg.T_HORIZON, cfg.G0),
        "bound_holds_fraction": bound_holds_fraction,
        "n_steps": n_steps,
        "n_paths": n_paths,
        "elapsed_sec": elapsed,
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

    drift_check = run_drift_check(result)
    print("Drift check (Section 3.3, p.8):")
    for row in drift_check["rows"]:
        print(f"  eta - kappa(y) = {row['eta_minus_kappa_offset']:+.2f}: "
              f"max|drift|={row['max_abs_drift']:.4e}, min|drift|={row['min_abs_drift']:.4e}")

    mc_result = run_monte_carlo(result, n_steps, n_paths)
    print(f"M_0 = {mc_result['M0']:.6f}")
    for t, e_m, se in zip(mc_result["checkpoints"], mc_result["E_M"], mc_result["se_M"]):
        print(f"t={t:5.2f}  E[M_t]={e_m:.6f} (+/- {se:.2e})  E[M_t]/M_0={e_m/mc_result['M0']:.6f}")
    print(f"c_T={mc_result['c_T']:.6e}, fraction of paths where M_t/Z1_0 >= c_T*L1_t: "
          f"{mc_result['bound_holds_fraction']}")

    strictly_below_M0 = all(e < mc_result["M0"] for e in mc_result["E_M"])
    bound_holds = all(frac > 0.999 for frac in mc_result["bound_holds_fraction"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ratio = np.array(mc_result["E_M"]) / mc_result["M0"]
    se_ratio = np.array(mc_result["se_M"]) / mc_result["M0"]
    ax.errorbar(mc_result["checkpoints"], ratio, yerr=se_ratio, marker="o", capsize=3)
    ax.axhline(1.0, color="r", linestyle="--", label=r"1 ($M_0$ benchmark)")
    ax.set_title(r"$E[M_t]/M_0$: deflated cum-dividend stock price (Eq. Section 3.3)")
    ax.set_xlabel("t (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_bubble.png", dpi=150)
    plt.close(fig)

    metrics = {
        "parameters": {
            "gamma": cfg.GAMMA, "sigma_D": cfg.SIGMA_D, "mu_D": cfg.MU_D, "beta": cfg.BETA,
            "D0": cfg.D0, "Y0": cfg.Y0, "varrho": cfg.RHO, "h_power": cfg.H_POWER,
            "T_horizon": cfg.T_HORIZON, "n_steps": n_steps, "n_paths": n_paths, "seed": cfg.MC_SEED,
            "eta_offsets": list(cfg.ETA_OFFSETS),
        },
        "drift_check": drift_check,
        "monte_carlo": mc_result,
        "verdict": {
            "only_eta_equals_kappa_annihilates_drift": drift_check["only_eta_equals_kappa_annihilates_drift"],
            "E_M_strictly_below_M0": strictly_below_M0,
            "lower_bound_M_geq_cT_L1_holds": bound_holds,
            "sigma_S_bounded_away_from_zero": bool(np.min(result["sigma_S"]) > 0.0),
        },
    }
    with open(RESULTS_DIR / "exp3_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Saved results/exp3_bubble.png and results/exp3_metrics.json")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
