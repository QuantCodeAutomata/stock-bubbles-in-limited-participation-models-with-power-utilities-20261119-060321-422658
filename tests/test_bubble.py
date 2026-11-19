"""Unit tests for src/bubble.py: Theorem 2.2's bubble-property building blocks."""

from __future__ import annotations

import numpy as np
import pytest

from src import bubble
from src import config as cfg
from src import equilibrium as eq


@pytest.fixture(scope="module")
def equilibrium_small():
    return eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=1e-3, n_grid=400, h_power=cfg.H_POWER, tol=1e-6, max_nodes=50000,
    )


def test_drift_vanishes_only_at_eta_equals_kappa(equilibrium_small):
    """Section 3.3, p.8: sigma_S(y)(kappa(y)-eta) = 0 for all y iff eta = kappa(y)."""
    sigma_S = equilibrium_small["sigma_S"]
    kappa = equilibrium_small["kappa"]

    drift_at_kappa = bubble.deflated_price_drift(sigma_S, kappa, kappa)
    assert np.allclose(drift_at_kappa, 0.0, atol=1e-10)

    drift_offset = bubble.deflated_price_drift(sigma_S, kappa, kappa + 0.1)
    assert np.all(np.abs(drift_offset) > 1e-8)


def test_stock_price_positive(equilibrium_small):
    """S_t = D_t v(Y_t) + D_t g(Y_t)(1-Y_t)^gamma > 0 (Eq. 20)."""
    v_interp = equilibrium_small["v_interp"]
    g_interp = equilibrium_small["g_interp"]
    Y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    D = np.full_like(Y, 2.0)
    S = bubble.stock_price(D, Y, v_interp, g_interp, cfg.GAMMA)
    assert np.all(S > 0.0)


def test_m_process_decomposition():
    """M_t = Z1_t*S_t + int_0^t Z1_s D_s ds is the sum of its two components."""
    Z1 = np.array([1.0, 0.9, 0.8])
    S = np.array([10.0, 11.0, 12.0])
    int_Z1_D = np.array([0.0, 0.5, 1.2])
    M = bubble.M_process(Z1, S, int_Z1_D)
    assert np.allclose(M, Z1 * S + int_Z1_D)


def test_c_T_bound_positive_and_decreasing_in_T():
    """c_T = X2bar_0 * exp(-T/g(0)) is positive and decreases as T grows
    (Section 3.3, p.8), since delta_2(y) <= 1/g(0) bounds the exponent."""
    c_small_T = bubble.c_T_bound(1.0, T=1.0, g0=cfg.G0)
    c_large_T = bubble.c_T_bound(1.0, T=10.0, g0=cfg.G0)
    assert c_small_T > 0.0
    assert c_large_T > 0.0
    assert c_large_T < c_small_T


def test_x2_bar_lower_bound_nonincreasing():
    """X-bar_{2,t} = X-bar_{2,0}*exp(-int_0^t delta_2(Y_s)ds) is non-increasing
    because delta_2 > 0, so the cumulative integral is non-decreasing."""
    x2_bar_0 = 5.0
    int_delta2 = np.array([0.0, 0.1, 0.3, 0.6])
    x2_bar = bubble.x2_bar_lower_bound(x2_bar_0, int_delta2)
    assert np.all(np.diff(x2_bar) <= 0.0)
    assert np.isclose(x2_bar[0], x2_bar_0)
