"""Equilibrium ODEs of the limited-participation economy.

Implements the state-process coefficients, the Dirichlet problem (18) for
k(y), the function g(y), and the stock-price volatility sigma_S(y) used in
Proposition 2.1 (Section 2, pp. 2-6) and in the proof of Theorem 2.2
(Section 3.3, p. 9) of "Stock bubbles in limited participation models with
power utilities".

The function h(y) that drives all of these coefficients comes from Theorem
3.6 in Guasoni, Larsen and Leoni (2025). This paper states only its boundary
behaviour: h is continuously differentiable on [0,1], valued in [gamma,1],
with h(0)=gamma and h(1)=1, and explicitly notes "the specific ODE for h is
not used here" (Section 2, p. 2). Since that ODE is not available, `h_function`
below is one admissible representative satisfying exactly the three stated
properties; `H_POWER_ALT` in `config.py` gives a second, qualitatively
different representative used as a robustness check in Experiment 1.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.interpolate import interp1d

HFunc = Callable[[np.ndarray], np.ndarray]


def h_function(y: np.ndarray, gamma: float, power: float = 1.0) -> np.ndarray:
    """Assumed admissible representative of h (see module docstring).

    h(y) = gamma + (1-gamma) * y**power satisfies h(0)=gamma, h(1)=1 and is
    valued in [gamma, 1] for y in [0,1] and any power > 0.
    """
    y = np.asarray(y, dtype=float)
    return gamma + (1.0 - gamma) * np.clip(y, 0.0, 1.0) ** power


def h_prime(y: np.ndarray, gamma: float, power: float = 1.0) -> np.ndarray:
    """Derivative of the assumed h(y) above."""
    y = np.clip(np.asarray(y, dtype=float), 1e-300, 1.0)
    return (1.0 - gamma) * power * y ** (power - 1.0)


def mu_Y(y: np.ndarray, gamma: float, sigma_D: float, h_func: HFunc) -> np.ndarray:
    """Drift of the state process Y under P, Eq. (2)."""
    y = np.asarray(y, dtype=float)
    h = h_func(y)
    return (
        sigma_D ** 2 * (1 - y) * (1 + gamma + 2 * gamma * y * h - 2 * y * (1 + gamma))
        / (2 * y * h ** 2)
    )


def sigma_Y(y: np.ndarray, sigma_D: float, h_func: HFunc) -> np.ndarray:
    """Volatility of the state process Y, Eq. (2)."""
    y = np.asarray(y, dtype=float)
    return sigma_D * (1 - y) / h_func(y)


def mu_Y_Q(y: np.ndarray, gamma: float, sigma_D: float, h_func: HFunc) -> np.ndarray:
    """Drift of Y under the measure Q used to define k, Eq. (12)."""
    return mu_Y(y, gamma, sigma_D, h_func) + (1 - gamma) * sigma_D * sigma_Y(y, sigma_D, h_func)


def kappa(y: np.ndarray, gamma: float, sigma_D: float, h_func: HFunc) -> np.ndarray:
    """Market price of risk for trader 1, Eq. (5)."""
    y = np.asarray(y, dtype=float)
    h = h_func(y)
    return gamma * sigma_D * ((1 - y) / (y * h) + 1.0)


def r_rate(y: np.ndarray, gamma: float, sigma_D: float, mu_D: float, rho: float, h_func: HFunc) -> np.ndarray:
    """Equilibrium interest rate, Eq. (5)."""
    y = np.asarray(y, dtype=float)
    h = h_func(y)
    return rho + mu_D - gamma * sigma_D ** 2 - gamma * (1 + gamma) * sigma_D ** 2 * (1 - y) / (2 * y * h ** 2)


def nu(y: np.ndarray, gamma: float, sigma_D: float, h_func: HFunc) -> np.ndarray:
    """Market price of risk entering trader 2's deflator, Eq. (9)."""
    y = np.asarray(y, dtype=float)
    h = h_func(y)
    return -gamma * sigma_D * (1 - h) / h


def solve_k(
    gamma: float,
    sigma_D: float,
    rho: float,
    h_func: HFunc,
    eps: float,
    n_grid: int,
    tol: float = 1e-8,
    max_nodes: int = 100000,
):
    """Solve the Dirichlet problem (18) for k(y) on [eps, 1-eps].

    ODE (18): (1/2) sigma_Y(y)^2 k''(y) + mu_Y^Q(y) k'(y) - rho k(y) = -y^(1-gamma).

    Boundary conditions. The paper states 0 < k(y) < 1/rho, k non-decreasing,
    and k in C([0,1]) with k(0+), k(1-) existing (Section 3.1, Step 2/3, p.4),
    but does not give their numeric values. Two facts about the ODE's
    coefficients pin them down:

    * As y -> 1, sigma_Y(y) = sigma_D(1-y)/h(y) -> 0 and mu_Y^Q(y) -> 0 (because
      mu_Y(y) has a factor (1-y) and sigma_Y(1)=0), so (18) degenerates to
      -rho*k(1) = -1, i.e. k(1-) = 1/rho exactly (the upper bound is attained
      in the limit).
    * As y -> 0, mu_Y(y) ~ C/y with C = sigma_D^2(1+gamma)/(2*gamma^2) > 0
      (since h(0)=gamma), so mu_Y^Q(y) also diverges like C/y. For k'(y) to
      stay bounded while (C/y)*k'(y) remains finite in (18), regularity at
      this singular boundary forces k'(0+) = 0.

    These two conditions (k'(eps) = 0, k(1-eps) = 1/rho) are used as the
    boundary conditions of the `solve_bvp` problem below.

    Returns the grid, k(y), k'(y) and the `solve_bvp` solution object.
    """

    def odefun(y: np.ndarray, state: np.ndarray) -> np.ndarray:
        k, kp = state
        h = h_func(y)
        sig_y = sigma_D * (1 - y) / h
        mu_yq = mu_Y_Q(y, gamma, sigma_D, h_func)
        kpp = (rho * k - y ** (1 - gamma) - mu_yq * kp) / (0.5 * sig_y ** 2)
        return np.vstack([kp, kpp])

    def bc(state_a: np.ndarray, state_b: np.ndarray) -> np.ndarray:
        return np.array([state_a[1] - 0.0, state_b[0] - 1.0 / rho])

    y_mesh = np.linspace(eps, 1 - eps, 400)
    guess_k = (1.0 / rho) * (y_mesh - eps) / (1 - 2 * eps)
    guess_kp = np.full_like(y_mesh, (1.0 / rho) / (1 - 2 * eps))
    y_guess = np.vstack([guess_k, guess_kp])

    sol = solve_bvp(odefun, bc, y_mesh, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    if not sol.success:
        raise RuntimeError(f"solve_k: solve_bvp did not converge ({sol.message})")

    y_grid = np.linspace(eps, 1 - eps, n_grid)
    k_vals, kp_vals = sol.sol(y_grid)
    assert np.all(k_vals > 0.0), "k(y) must be strictly positive (Eq. 13)"
    assert np.all(k_vals < 1.0 / rho + 1e-6), "k(y) must stay below 1/varrho (Eq. 13)"
    return y_grid, k_vals, kp_vals, sol


def v_and_v_prime(y_grid: np.ndarray, gamma: float, k_vals: np.ndarray, kp_vals: np.ndarray):
    """v(y) = y^gamma k(y) and v'(y) = gamma y^(gamma-1) k(y) + y^gamma k'(y)."""
    v = y_grid ** gamma * k_vals
    v_prime = gamma * y_grid ** (gamma - 1.0) * k_vals + y_grid ** gamma * kp_vals
    return v, v_prime


def solve_g(y_grid: np.ndarray, gamma: float, g0: float, h_func: HFunc):
    """g(y) from the exponential representation (Section 3.3, p. 9):

    delta_2(y) := (1-y)^(1-gamma) / g(y) = (1/g(0)) * exp( int_0^y (h(q)-1)/(1-q) dq )
    => g(y) = (1-y)^(1-gamma) * g(0) * exp( -int_0^y (h(q)-1)/(1-q) dq ).

    Returns g(y) on y_grid and the cumulative integral used for delta_2(y).
    """
    integrand = (h_func(y_grid) - 1.0) / (1.0 - y_grid)
    cum_integral = cumulative_trapezoid(integrand, y_grid, initial=0.0)
    g_vals = (1.0 - y_grid) ** (1.0 - gamma) * g0 * np.exp(-cum_integral)
    assert np.all(g_vals > 0.0), "g(y) must be strictly positive"
    return g_vals, cum_integral


def delta2(g0: float, cum_integral: np.ndarray) -> np.ndarray:
    """delta_2(y) = (1/g(0)) * exp(cum_integral(y)), Section 3.3, p. 9."""
    return np.exp(cum_integral) / g0


def sigma_S(v: np.ndarray, v_prime: np.ndarray, sigma_Y_vals: np.ndarray, g_vals: np.ndarray,
            y_grid: np.ndarray, gamma: float, sigma_D: float) -> np.ndarray:
    """Equilibrium stock-price volatility, Eq. (21)."""
    numerator = sigma_D * v + sigma_Y_vals * v_prime
    denominator = v + g_vals * (1 - y_grid) ** gamma
    return numerator / denominator


def build_equilibrium(
    gamma: float,
    sigma_D: float,
    mu_D: float,
    rho: float,
    g0: float,
    eps: float,
    n_grid: int,
    h_power: float = 1.0,
    tol: float = 1e-8,
    max_nodes: int = 100000,
) -> dict:
    """Solve the full equilibrium (h, k, v, g, sigma_S, kappa, r, nu, delta_2)
    on a grid in (0,1) and return everything plus interpolants over Y.
    """
    def h_func(y):
        return h_function(y, gamma, power=h_power)

    y_grid, k_vals, kp_vals, bvp_sol = solve_k(gamma, sigma_D, rho, h_func, eps, n_grid, tol, max_nodes)
    v_vals, v_prime_vals = v_and_v_prime(y_grid, gamma, k_vals, kp_vals)
    g_vals, cum_integral = solve_g(y_grid, gamma, g0, h_func)
    delta2_vals = delta2(g0, cum_integral)
    sigma_Y_vals = sigma_Y(y_grid, sigma_D, h_func)
    sigma_S_vals = sigma_S(v_vals, v_prime_vals, sigma_Y_vals, g_vals, y_grid, gamma, sigma_D)
    kappa_vals = kappa(y_grid, gamma, sigma_D, h_func)
    r_vals = r_rate(y_grid, gamma, sigma_D, mu_D, rho, h_func)
    nu_vals = nu(y_grid, gamma, sigma_D, h_func)

    interp_kwargs = dict(kind="cubic", bounds_error=False, fill_value="extrapolate")
    return {
        "h_func": h_func,
        "y_grid": y_grid,
        "k": k_vals,
        "k_prime": kp_vals,
        "v": v_vals,
        "v_prime": v_prime_vals,
        "g": g_vals,
        "delta2": delta2_vals,
        "sigma_Y": sigma_Y_vals,
        "sigma_S": sigma_S_vals,
        "kappa": kappa_vals,
        "r": r_vals,
        "nu": nu_vals,
        "bvp_sol": bvp_sol,
        "v_interp": interp1d(y_grid, v_vals, **interp_kwargs),
        "g_interp": interp1d(y_grid, g_vals, **interp_kwargs),
        "sigma_S_interp": interp1d(y_grid, sigma_S_vals, **interp_kwargs),
        "kappa_interp": interp1d(y_grid, kappa_vals, **interp_kwargs),
        "r_interp": interp1d(y_grid, r_vals, **interp_kwargs),
        "delta2_interp": interp1d(y_grid, delta2_vals, **interp_kwargs),
    }
