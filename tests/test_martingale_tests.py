"""Unit tests for src/martingale_tests.py: the Delbaen-Shirakawa integral test
(Eqs. 26-28) used in the proof of Lemma 3.1."""

from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src import equilibrium as eq
from src import martingale_tests as mt


def h_func(y):
    return eq.h_function(y, cfg.GAMMA, power=cfg.H_POWER)


def test_y_of_lambda_is_inverse_map():
    """y(lambda) = 1/(1+lambda^(1/gamma)) in (0,1) and y(0)=1, y(infty)=0 (Eq. 26)."""
    lam = np.array([0.0, 1.0, 1e6])
    y = mt.y_of_lambda(lam, cfg.GAMMA)
    assert np.isclose(y[0], 1.0)
    assert 0.0 < y[1] < 1.0
    assert y[2] < 1e-3


def test_a_function_positive():
    """a(lambda) > 0 for lambda > 0, since it is a diffusion coefficient (Eq. 27)."""
    lam = np.geomspace(1e-3, 1e8, 100)
    a = mt.a_function(lam, cfg.GAMMA, cfg.SIGMA_D, h_func)
    assert np.all(a > 0.0)


def test_asymptotic_a_over_lambda_converges_to_sigma_D():
    """a(lambda)/lambda^(1+1/gamma) -> sigma_D as lambda -> infinity (Section 3.2, p.8)."""
    lam = np.geomspace(1e3, 1e9, 20)
    ratio = mt.asymptotic_a_over_lambda(cfg.GAMMA, cfg.SIGMA_D, h_func, lam)
    assert np.isclose(ratio[-1], cfg.SIGMA_D, rtol=1e-3)
    # the ratio should approach sigma_D monotonically from this tail region
    assert abs(ratio[-1] - cfg.SIGMA_D) < abs(ratio[0] - cfg.SIGMA_D)


def test_integrand_tail_exponent_matches_derivation():
    """Exponent -1-2/gamma of the integrand's power-law tail (p.8)."""
    exponent = mt.integrand_tail_exponent(cfg.GAMMA)
    assert np.isclose(exponent, -1.0 - 2.0 / cfg.GAMMA)
    # for gamma in (0,1), -1-2/gamma < -3, well below -1: integrable at infinity
    assert exponent < -1.0


def test_integral_test_is_finite():
    """Eq. (28): int^infty lambda/a(lambda)^2 dlambda must be finite, certifying
    that Lambda is a strict local martingale under Q_2 (Lemma 3.1)."""
    value, err = mt.integral_test(cfg.GAMMA, cfg.SIGMA_D, h_func, cfg.LAMBDA_CUTOFF, cfg.LAMBDA_UPPER)
    assert np.isfinite(value)
    assert value >= 0.0


@pytest.mark.parametrize("gamma", [0.3, 0.5, 0.8])
def test_integral_test_finite_across_gamma(gamma):
    """The finiteness of the integral test should hold for any gamma in (0,1),
    since the tail exponent -1-2/gamma < -1 for every gamma in (0,1)."""
    def h_g(y):
        return eq.h_function(y, gamma, power=1.0)
    value, _ = mt.integral_test(gamma, cfg.SIGMA_D, h_g, 1.0e6, np.inf)
    assert np.isfinite(value)
