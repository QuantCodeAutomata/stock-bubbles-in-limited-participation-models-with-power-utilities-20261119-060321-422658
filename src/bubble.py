"""Bubble property of the equilibrium stock price (Theorem 2.2, Section 3.3).

Provides:
 - the finite-variation drift of the deflated cum-dividend price process for
   a generic deflator eta (used to show only eta = kappa(Y_t) annihilates it);
 - the processes M_t and L_1,t needed to show that even eta = kappa(Y_t) only
   yields a strict local martingale, not a true martingale;
 - the lower-bound construction (c_T, X-bar_2) used in the proof.
"""

from __future__ import annotations

import numpy as np


def deflated_price_drift(sigma_S_vals: np.ndarray, kappa_vals: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Finite-variation drift coefficient sigma_S(y) * (kappa(y) - eta) of
    d(Z^eta_t S_t + int_0^t Z^eta_s D_s ds) (Section 3.3, p. 8). It vanishes
    identically in y iff eta = kappa(y) because sigma_S(y) > 0 everywhere
    (Proposition 2.1)."""
    return sigma_S_vals * (kappa_vals - eta)


def stock_price(D: np.ndarray, Y: np.ndarray, v_interp, g_interp, gamma: float) -> np.ndarray:
    """S_t = D_t v(Y_t) + D_t g(Y_t) (1-Y_t)^gamma, Eq. (20)."""
    return D * (v_interp(Y) + g_interp(Y) * (1.0 - Y) ** gamma)


def normalized_L1(S0: np.ndarray, Z1: np.ndarray, Z1_0: float) -> np.ndarray:
    """L_1,t = S^(0)_t * Z1_t / Z1_0, Eq. (25)."""
    return S0 * Z1 / Z1_0


def M_process(Z1: np.ndarray, S: np.ndarray, int_Z1_D: np.ndarray) -> np.ndarray:
    """M_t = Z1_t S_t + int_0^t Z1_s D_s ds (Section 3.3, p. 8)."""
    return Z1 * S + int_Z1_D


def x2_bar_lower_bound(x2_bar_0: float, int_delta2: np.ndarray) -> np.ndarray:
    """X-bar_{2,t} = X-bar_{2,0} * exp(-int_0^t delta_2(Y_s) ds) (Section 3.3, p. 8)."""
    return x2_bar_0 * np.exp(-int_delta2)


def c_T_bound(x2_bar_0: float, T: float, g0: float) -> float:
    """c_T := X-bar_{2,0} * exp(-T/g(0)), the uniform lower bound on [0,T]
    obtained from delta_2(y) <= 1/g(0) (Section 3.3, p. 8)."""
    return x2_bar_0 * np.exp(-T / g0)
