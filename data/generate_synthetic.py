"""Monte Carlo path generation for the state process Y and the dividend D.

No external market data is used anywhere in this project: Section 2 of
"Stock bubbles in limited participation models with power utilities"
specifies the economy entirely through model parameters (gamma, sigma_D,
mu_D, beta, D_0) and the deterministic functions of Theorem 3.6 in Guasoni,
Larsen and Leoni (2025). Experiments 2 and 3 verify martingale properties by
Monte Carlo simulation of the state-price deflators and the stock price
along simulated paths of (Y_t, D_t); this module produces those paths with
Euler-Maruyama discretization and accumulates the running integrals that
Experiments 2 and 3 need, without storing full trajectories in memory.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

HFunc = Callable[[np.ndarray], np.ndarray]


def simulate_P_paths(
    Y0: float,
    D0: float,
    gamma: float,
    sigma_D: float,
    mu_D: float,
    beta: float,
    h_func: HFunc,
    r_func: Callable[[np.ndarray], np.ndarray],
    delta2_func: Callable[[np.ndarray], np.ndarray],
    T: float,
    n_steps: int,
    n_paths: int,
    checkpoints: Sequence[float],
    seed: int,
    boundary_eps: float = 1e-6,
) -> tuple[dict, float]:
    """Euler-Maruyama simulation of (Y_t, D_t) under the physical measure P.

    Y follows Eq. (2): dY_t = mu_Y(Y_t)dt + sigma_Y(Y_t)dB_t.
    D follows the exogenous dividend GBM: dD_t = mu_D D_t dt + sigma_D D_t dB_t
    (Section 2, p. 2), driven by the SAME Brownian motion B as Y.

    Along the way it accumulates, without ever storing a full (n_steps x
    n_paths) array:
      * log S^(0)_t = int_0^t r(Y_s) ds (Eq. 3);
      * int_0^t Z1_s D_s ds, the running integral in M_t (Section 3.3, p. 8);
      * int_0^t delta_2(Y_s) ds, used for the X-bar_2 lower bound (p. 8).

    ``boundary_eps`` is a purely numerical safeguard, not a model parameter:
    the Dirichlet diffusion for Y never reaches the endpoints 0 and 1 (Eq. 2
    has finite, non-degenerate coefficients there), but a discrete-time Euler
    step can overshoot past them. Clipping to [boundary_eps, 1-boundary_eps]
    keeps the path inside the state space without materially affecting the
    econometrics of the interior process.

    Returns (checkpoint_data, Z1_0) where checkpoint_data maps each requested
    checkpoint time to a dict of length-n_paths arrays: Y, D, S0, Z1,
    int_Z1_D, int_delta2.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    eps = boundary_eps

    Y = np.full(n_paths, Y0, dtype=float)
    D = np.full(n_paths, D0, dtype=float)
    log_S0 = np.zeros(n_paths)
    int_Z1_D = np.zeros(n_paths)
    int_delta2 = np.zeros(n_paths)

    Z1_0 = np.exp(-beta * 0.0) * (D0 * Y0) ** (-gamma)

    checkpoints_sorted = sorted(checkpoints)
    results: dict[float, dict[str, np.ndarray]] = {}
    cp_idx = 0

    for i in range(1, n_steps + 1):
        y = np.clip(Y, eps, 1 - eps)
        h = h_func(y)
        mu_y = (
            sigma_D ** 2 * (1 - y) * (1 + gamma + 2 * gamma * y * h - 2 * y * (1 + gamma))
            / (2 * y * h ** 2)
        )
        sig_y = sigma_D * (1 - y) / h
        dB = rng.standard_normal(n_paths) * np.sqrt(dt)

        Y = np.clip(Y + mu_y * dt + sig_y * dB, eps, 1 - eps)
        D = D * np.exp((mu_D - 0.5 * sigma_D ** 2) * dt + sigma_D * dB)

        r_val = r_func(y)
        log_S0 += r_val * dt
        int_delta2 += delta2_func(y) * dt

        t = i * dt
        y_now = np.clip(Y, eps, 1 - eps)
        Z1 = np.exp(-beta * t) * (D * y_now) ** (-gamma)
        int_Z1_D += Z1 * D * dt

        if cp_idx < len(checkpoints_sorted) and np.isclose(t, checkpoints_sorted[cp_idx], atol=dt / 2):
            results[checkpoints_sorted[cp_idx]] = {
                "Y": Y.copy(),
                "D": D.copy(),
                "S0": np.exp(log_S0),
                "Z1": Z1.copy(),
                "int_Z1_D": int_Z1_D.copy(),
                "int_delta2": int_delta2.copy(),
            }
            cp_idx += 1

    return results, Z1_0


def simulate_Q2_lambda_paths(
    Y0: float,
    gamma: float,
    sigma_D: float,
    h_func: HFunc,
    nu_func: Callable[[np.ndarray], np.ndarray],
    T: float,
    n_steps: int,
    n_paths: int,
    checkpoints: Sequence[float],
    seed: int,
    boundary_eps: float = 1e-6,
) -> dict:
    """Simulate Y under Q_2 (drift adjusted by -sigma_Y*nu via Eq. 23) and
    return Lambda_t/Lambda_0 at the requested checkpoints.

    Under P, dY_t = mu_Y(Y_t)dt + sigma_Y(Y_t)dB_t. Eq. (23) defines the
    Q_2-Brownian motion W_{2,t} = B_t + int_0^t nu(Y_s)ds, so dB_t = dW_{2,t}
    - nu(Y_t)dt, giving the Q_2-dynamics
        dY_t = (mu_Y(Y_t) - sigma_Y(Y_t) nu(Y_t)) dt + sigma_Y(Y_t) dW_{2,t}.

    ``boundary_eps`` is the same numerical safeguard against Euler overshoot
    documented in :func:`simulate_P_paths`; Lambda = ((1-Y)/Y)^gamma diverges
    at Y=0, so a tighter clip bounds the worst-case discretization artifact.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    eps = boundary_eps

    Y = np.full(n_paths, Y0, dtype=float)
    Lambda_0 = ((1.0 - Y0) / Y0) ** gamma

    checkpoints_sorted = sorted(checkpoints)
    results: dict[float, np.ndarray] = {}
    cp_idx = 0

    for i in range(1, n_steps + 1):
        y = np.clip(Y, eps, 1 - eps)
        h = h_func(y)
        mu_y = (
            sigma_D ** 2 * (1 - y) * (1 + gamma + 2 * gamma * y * h - 2 * y * (1 + gamma))
            / (2 * y * h ** 2)
        )
        sig_y = sigma_D * (1 - y) / h
        mu_y_q2 = mu_y - sig_y * nu_func(y)
        dW2 = rng.standard_normal(n_paths) * np.sqrt(dt)

        Y = np.clip(Y + mu_y_q2 * dt + sig_y * dW2, eps, 1 - eps)

        t = i * dt
        if cp_idx < len(checkpoints_sorted) and np.isclose(t, checkpoints_sorted[cp_idx], atol=dt / 2):
            y_now = np.clip(Y, eps, 1 - eps)
            lam = ((1.0 - y_now) / y_now) ** gamma
            results[checkpoints_sorted[cp_idx]] = lam / Lambda_0
            cp_idx += 1

    return results
