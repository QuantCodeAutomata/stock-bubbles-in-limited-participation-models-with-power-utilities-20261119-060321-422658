"""Delbaen-Shirakawa (2002) integral test for the ratio process Lambda.

Implements Eqs. (26)-(28) of "Stock bubbles in limited participation models
with power utilities" (Section 3.2, pp. 7-8), used in the proof of Lemma 3.1
to show that Lambda_t = ((1-Y_t)/Y_t)^gamma is a strict local martingale
under Q_2.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad

HFunc = Callable[[np.ndarray], np.ndarray]


def y_of_lambda(lam: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse map y(lambda) = 1 / (1 + lambda^(1/gamma)), Eq. (26)."""
    lam = np.asarray(lam, dtype=float)
    return 1.0 / (1.0 + lam ** (1.0 / gamma))


def a_function(lam: np.ndarray, gamma: float, sigma_D: float, h_func: HFunc) -> np.ndarray:
    """Diffusion coefficient a(lambda) of the autonomous SDE (27) for Lambda."""
    lam = np.asarray(lam, dtype=float)
    y = y_of_lambda(lam, gamma)
    return lam * gamma * sigma_D / (y * h_func(y))


def integral_test(
    gamma: float,
    sigma_D: float,
    h_func: HFunc,
    lower_cutoff: float,
    upper: float = np.inf,
):
    """Numerically evaluate I = int_{lower_cutoff}^{upper} lambda / a(lambda)^2 dlambda.

    Delbaen and Shirakawa (2002), Theorem 1.6: N_t = Lambda_t/Lambda_0 is a
    (true) martingale under Q_2 iff int^infty lambda/a(lambda)^2 dlambda = infinity
    (Eq. 28). A finite value of I therefore certifies that N (and hence
    Lambda) is a strict local martingale.
    """

    def integrand(lam: float) -> float:
        a = a_function(lam, gamma, sigma_D, h_func)
        return lam / a ** 2

    value, err = quad(integrand, lower_cutoff, upper, limit=200)
    return value, err


def asymptotic_a_over_lambda(gamma: float, sigma_D: float, h_func: HFunc, lam: np.ndarray) -> np.ndarray:
    """a(lambda) / lambda^(1+1/gamma), which the paper shows converges to
    sigma_D as lambda -> infinity (Section 3.2, p. 8), because
    y(lambda)*lambda^(1/gamma) -> 1 and h(y(lambda)) -> h(0) = gamma."""
    lam = np.asarray(lam, dtype=float)
    a = a_function(lam, gamma, sigma_D, h_func)
    return a / lam ** (1.0 + 1.0 / gamma)


def integrand_tail_exponent(gamma: float) -> float:
    """Exponent of the lambda^(-1-2/gamma) asymptotic decay of the integrand
    in (28), derived on p. 8 from a(lambda) ~ sigma_D * lambda^(1+1/gamma)."""
    return -1.0 - 2.0 / gamma
