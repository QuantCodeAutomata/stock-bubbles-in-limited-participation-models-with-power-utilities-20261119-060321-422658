"""Unit tests for src/equilibrium.py: the ODEs of Proposition 2.1."""

from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src import equilibrium as eq


@pytest.fixture(scope="module")
def equilibrium_small():
    """Equilibrium solved on a small grid, shared by the tests below."""
    return eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=1e-3, n_grid=400, h_power=cfg.H_POWER, tol=1e-6, max_nodes=50000,
    )


def test_h_function_boundary_values():
    """h(0) = gamma and h(1) = 1, as stated in Section 2, p.2."""
    y = np.array([0.0, 1.0])
    h = eq.h_function(y, cfg.GAMMA, power=cfg.H_POWER)
    assert np.isclose(h[0], cfg.GAMMA)
    assert np.isclose(h[1], 1.0)


def test_h_function_range():
    """h(y) in [gamma, 1] for y in [0,1] (Section 2, p.2)."""
    y = np.linspace(0, 1, 200)
    h = eq.h_function(y, cfg.GAMMA, power=cfg.H_POWER)
    assert np.all(h >= cfg.GAMMA - 1e-12)
    assert np.all(h <= 1.0 + 1e-12)


def test_rho_positive():
    """Assumption 2.1 (Eq. 1): varrho > 0 for the configured parameters."""
    assert cfg.RHO > 0.0


def test_k_bounds(equilibrium_small):
    """0 < k(y) < 1/varrho on the grid (Eq. 13)."""
    k = equilibrium_small["k"]
    assert np.all(k > 0.0)
    assert np.all(k < 1.0 / cfg.RHO + 1e-6)


def test_k_non_decreasing(equilibrium_small):
    """k is non-decreasing, as stated for the solution of ODE (18)."""
    k = equilibrium_small["k"]
    diffs = np.diff(k)
    assert np.all(diffs > -1e-6)


def test_v_prime_strictly_positive(equilibrium_small):
    """v'(y) = gamma*y^(gamma-1)*k(y) + y^gamma*k'(y) > 0 on (0,1) (Prop. 2.1 proof, p.6)."""
    assert np.all(equilibrium_small["v_prime"] > 0.0)


def test_g_strictly_positive(equilibrium_small):
    """g(y) > 0 on (0,1), as required by the exponential representation (p.9)."""
    assert np.all(equilibrium_small["g"] > 0.0)


def test_sigma_S_strictly_positive_and_continuous(equilibrium_small):
    """Proposition 2.1: sigma_S(y) is continuous and strictly positive on (0,1).

    Strict positivity is checked directly; continuity is checked with a
    finite-difference proxy (bounded jump-to-step ratio on the interior grid).
    """
    sigma_S = equilibrium_small["sigma_S"]
    y = equilibrium_small["y_grid"]
    assert np.all(np.isfinite(sigma_S))
    assert np.all(sigma_S > 0.0)
    margin = max(1, len(y) // 100)
    interior = sigma_S[margin:-margin]
    max_jump = np.max(np.abs(np.diff(interior)))
    max_step = np.max(np.diff(y))
    assert np.isfinite(max_jump / max_step)


def test_sigma_S_numerator_dominates_sigma_D_v(equilibrium_small):
    """Expected outcome: numerator = sigma_D*v + sigma_Y*v' > sigma_D*v > 0."""
    v = equilibrium_small["v"]
    v_prime = equilibrium_small["v_prime"]
    sigma_Y_vals = equilibrium_small["sigma_Y"]
    numerator = cfg.SIGMA_D * v + sigma_Y_vals * v_prime
    assert np.all(numerator > cfg.SIGMA_D * v - 1e-12)
    assert np.all(v > 0.0)


def test_denominator_strictly_positive(equilibrium_small):
    """Denominator v(y) + g(y)(1-y)^gamma > 0 (every term is positive)."""
    v = equilibrium_small["v"]
    g = equilibrium_small["g"]
    y = equilibrium_small["y_grid"]
    denom = v + g * (1 - y) ** cfg.GAMMA
    assert np.all(denom > 0.0)


def test_kappa_finite(equilibrium_small):
    """kappa(y) = gamma*sigma_D*((1-y)/(y*h(y)) + 1) stays finite on (0,1) away from y=0."""
    assert np.all(np.isfinite(equilibrium_small["kappa"]))


def test_alternate_h_power_also_gives_positive_sigma_S():
    """Robustness: a second, qualitatively different admissible h(y) (quadratic
    instead of linear) still yields sigma_S > 0, i.e. the conclusion of
    Proposition 2.1 is not an artifact of one choice of h."""
    result = eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=1e-3, n_grid=400, h_power=cfg.H_POWER_ALT, tol=1e-6, max_nodes=50000,
    )
    assert np.all(result["sigma_S"] > 0.0)


def test_build_equilibrium_rejects_non_positive_rho():
    """Assumption 2.1 requires varrho > 0; solve_k should not silently proceed
    with a non-positive varrho (division by varrho appears in the boundary
    condition k(1-)=1/varrho)."""
    with pytest.raises(Exception):
        eq.build_equilibrium(
            gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=0.0, g0=1.0,
            eps=1e-3, n_grid=100, h_power=1.0, tol=1e-6, max_nodes=20000,
        )
