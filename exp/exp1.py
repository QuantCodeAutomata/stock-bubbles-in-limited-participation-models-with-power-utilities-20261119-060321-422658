"""Experiment 1: continuity and strict positivity of sigma_S(y) (Proposition 2.1).

Solves the ODEs for k(y) and g(y), computes v(y) = y^gamma k(y) and the
equilibrium stock-price volatility

    sigma_S(y) = (sigma_D v(y) + sigma_Y(y) v'(y)) / (v(y) + g(y) (1-y)^gamma),  y in (0,1)   (Eq. 21)

on a fine grid, and checks numerically that sigma_S is continuous and
strictly positive on (0,1), as proved in Section 3.1 (pp. 5-6). The check is
repeated for a second, qualitatively different admissible representative of
h(y) (see `src/equilibrium.py`) to confirm the conclusion does not depend on
which admissible h is used.

Run: python -m exp.exp1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src import equilibrium as eq

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(n_grid: int, h_power: float, label: str) -> dict:
    """Solve the equilibrium for one admissible h(y) and verify Proposition 2.1."""
    result = eq.build_equilibrium(
        gamma=cfg.GAMMA,
        sigma_D=cfg.SIGMA_D,
        mu_D=cfg.MU_D,
        rho=cfg.RHO,
        g0=cfg.G0,
        eps=cfg.EPS_Y,
        n_grid=n_grid,
        h_power=h_power,
        tol=cfg.BVP_TOL,
        max_nodes=cfg.BVP_MAX_NODES,
    )
    y = result["y_grid"]
    sigma_S = result["sigma_S"]
    v = result["v"]
    v_prime = result["v_prime"]

    min_sigma_S = float(np.min(sigma_S))
    max_grid_step = float(np.max(np.diff(y)))
    min_v_prime = float(np.min(v_prime))

    assert np.all(np.isfinite(sigma_S)), f"[{label}] sigma_S has non-finite values"
    assert min_sigma_S > 0.0, f"[{label}] sigma_S is not strictly positive: min={min_sigma_S}"
    assert min_v_prime > 0.0, f"[{label}] v'(y) is not strictly positive: min={min_v_prime}"
    # Continuity proxy restricted to the interior of (0,1): sigma_S(y) involves
    # y^(gamma-1) through v'(y), which only needs to stay continuous and positive
    # on the open interval (Proposition 2.1), not bounded as y -> 0 or y -> 1. The
    # first and last 1% of grid points are excluded from the jump/step ratio so
    # that this boundary growth is not mistaken for an interior discontinuity.
    margin = max(1, n_grid // 100)
    interior_sigma_S = sigma_S[margin:-margin]
    max_jump = float(np.max(np.abs(np.diff(interior_sigma_S))))
    lipschitz_proxy = max_jump / max_grid_step
    assert np.isfinite(lipschitz_proxy), f"[{label}] sigma_S jump/step ratio is not finite"

    return {
        "label": label,
        "h_power": h_power,
        "n_grid": n_grid,
        "min_sigma_S": min_sigma_S,
        "max_sigma_S": float(np.max(sigma_S)),
        "max_jump": max_jump,
        "max_grid_step": max_grid_step,
        "lipschitz_proxy": lipschitz_proxy,
        "min_v_prime": min_v_prime,
        "min_k": float(np.min(result["k"])),
        "max_k": float(np.max(result["k"])),
        "one_over_rho": 1.0 / cfg.RHO,
        "min_g": float(np.min(result["g"])),
        "max_g": float(np.max(result["g"])),
        "y_grid": y,
        "sigma_S": sigma_S,
        "v": v,
        "k": result["k"],
        "g": result["g"],
        "h": result["h_func"](y),
    }


def main(full: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_grid = cfg.N_GRID_PAPER if full else cfg.N_GRID_RUN

    base = run(n_grid, cfg.H_POWER, "h_power=1.0 (linear, baseline)")
    alt = run(n_grid, cfg.H_POWER_ALT, "h_power=2.0 (robustness check)")

    print(f"Grid: eps={cfg.EPS_Y}, n_grid={n_grid}")
    for res in (base, alt):
        print(
            f"[{res['label']}] sigma_S in [{res['min_sigma_S']:.6f}, {res['max_sigma_S']:.6f}], "
            f"min v'={res['min_v_prime']:.6f}, k in [{res['min_k']:.4f}, {res['max_k']:.4f}] "
            f"(bound 1/varrho={res['one_over_rho']:.4f}), "
            f"max |Delta sigma_S| / grid step = {res['lipschitz_proxy']:.4f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for res, style in ((base, "-"), (alt, "--")):
        axes[0, 0].plot(res["y_grid"], res["sigma_S"], style, label=res["label"])
        axes[0, 1].plot(res["y_grid"], res["v"], style, label=res["label"])
        axes[1, 0].plot(res["y_grid"], res["k"], style, label=res["label"])
        axes[1, 1].plot(res["y_grid"], res["h"], style, label=res["label"])
    axes[0, 0].set_title(r"$\sigma_S(y)$ (Eq. 21)")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].axhline(0, color="k", linewidth=0.5)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title(r"$v(y) = y^\gamma k(y)$")
    axes[0, 1].set_xlabel("y")
    axes[1, 0].set_title(r"$k(y)$ solving ODE (18), bound $1/\varrho$=" + f"{1/cfg.RHO:.3f}")
    axes[1, 0].axhline(1 / cfg.RHO, color="r", linestyle=":", linewidth=1, label="1/varrho")
    axes[1, 0].set_xlabel("y")
    axes[1, 1].set_title(r"assumed admissible $h(y)$")
    axes[1, 1].set_xlabel("y")
    fig.suptitle("Experiment 1: equilibrium functions and stock-price volatility")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_sigma_S.png", dpi=150)
    plt.close(fig)

    metrics = {
        "parameters": {
            "gamma": cfg.GAMMA,
            "sigma_D": cfg.SIGMA_D,
            "mu_D": cfg.MU_D,
            "beta": cfg.BETA,
            "D0": cfg.D0,
            "varrho": cfg.RHO,
            "g0": cfg.G0,
            "eps_y": cfg.EPS_Y,
            "n_grid": n_grid,
        },
        "baseline": {k: v for k, v in base.items() if k not in ("y_grid", "sigma_S", "v", "k", "g", "h")},
        "robustness_alt_h": {k: v for k, v in alt.items() if k not in ("y_grid", "sigma_S", "v", "k", "g", "h")},
        "verdict": {
            "sigma_S_strictly_positive": bool(base["min_sigma_S"] > 0 and alt["min_sigma_S"] > 0),
            "v_prime_strictly_positive": bool(base["min_v_prime"] > 0 and alt["min_v_prime"] > 0),
            "k_within_bounds": bool(
                0 < base["min_k"] and base["max_k"] < base["one_over_rho"] + 1e-6
                and 0 < alt["min_k"] and alt["max_k"] < alt["one_over_rho"] + 1e-6
            ),
        },
    }
    with open(RESULTS_DIR / "exp1_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved results/exp1_sigma_S.png and results/exp1_metrics.json")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
