"""Unit tests for data/generate_synthetic.py: Euler-Maruyama path generation
used by Experiments 2 and 3. Uses small synthetic grids only, no market data."""

from __future__ import annotations

import numpy as np

from data import generate_synthetic as gs
from src import config as cfg
from src import equilibrium as eq


def _h_func(y):
    return eq.h_function(y, cfg.GAMMA, power=cfg.H_POWER)


def test_simulate_P_paths_checkpoints_and_positivity():
    """Simulated D stays positive (geometric Brownian motion) and Y stays in
    (0,1) (the Dirichlet-diffusion state space, Eq. 2)."""
    result = eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=1e-3, n_grid=200, h_power=cfg.H_POWER, tol=1e-6, max_nodes=50000,
    )
    checkpoints = (0.1, 0.5, 1.0)
    paths, Z1_0 = gs.simulate_P_paths(
        Y0=0.5, D0=1.0, gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, beta=cfg.BETA,
        h_func=_h_func, r_func=result["r_interp"], delta2_func=result["delta2_interp"],
        T=1.0, n_steps=200, n_paths=500, checkpoints=checkpoints, seed=42,
    )
    assert Z1_0 > 0.0
    assert set(paths.keys()) == set(checkpoints)
    for t in checkpoints:
        data = paths[t]
        assert np.all(data["Y"] > 0.0) and np.all(data["Y"] < 1.0)
        assert np.all(data["D"] > 0.0)
        assert np.all(data["S0"] > 0.0)
        assert np.all(data["Z1"] > 0.0)
        assert np.all(data["int_Z1_D"] >= 0.0)
        assert np.all(data["int_delta2"] >= 0.0)
    # S^(0) is the exponentiated accumulated interest: it should grow with t
    # on average whenever the equilibrium rate r(y) is positive on average.
    assert np.mean(paths[1.0]["S0"]) != np.mean(paths[0.1]["S0"])


def test_simulate_P_paths_no_lookahead_in_checkpoint_ordering():
    """Checkpoints must be produced in increasing time order and each one
    must use only information accumulated up to that time (no look-ahead):
    the running integral int_Z1_D is non-decreasing across checkpoints
    because its integrand Z1_s*D_s is positive."""
    result = eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=1e-3, n_grid=200, h_power=cfg.H_POWER, tol=1e-6, max_nodes=50000,
    )
    checkpoints = (0.2, 0.6, 1.0)
    paths, _ = gs.simulate_P_paths(
        Y0=0.5, D0=1.0, gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, beta=cfg.BETA,
        h_func=_h_func, r_func=result["r_interp"], delta2_func=result["delta2_interp"],
        T=1.0, n_steps=200, n_paths=300, checkpoints=checkpoints, seed=7,
    )
    int_means = [np.mean(paths[t]["int_Z1_D"]) for t in checkpoints]
    assert int_means[0] <= int_means[1] <= int_means[2]
    int_delta2_means = [np.mean(paths[t]["int_delta2"]) for t in checkpoints]
    assert int_delta2_means[0] <= int_delta2_means[1] <= int_delta2_means[2]


def test_simulate_Q2_lambda_paths_ratio_positive():
    """Lambda_t/Lambda_0 = (((1-Y_t)/Y_t)/((1-Y_0)/Y_0))^gamma > 0 under Q_2."""
    result = eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=1e-3, n_grid=200, h_power=cfg.H_POWER, tol=1e-6, max_nodes=50000,
    )
    nu_func = lambda y: eq.nu(y, cfg.GAMMA, cfg.SIGMA_D, _h_func)
    checkpoints = (0.1, 0.5, 1.0)
    out = gs.simulate_Q2_lambda_paths(
        Y0=0.5, gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, h_func=_h_func, nu_func=nu_func,
        T=1.0, n_steps=200, n_paths=500, checkpoints=checkpoints, seed=11,
    )
    for t in checkpoints:
        assert np.all(out[t] > 0.0)
        assert np.all(np.isfinite(out[t]))


def test_single_step_reproducible_with_same_seed():
    """Same seed must reproduce identical paths (deterministic Monte Carlo)."""
    result = eq.build_equilibrium(
        gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, rho=cfg.RHO, g0=cfg.G0,
        eps=1e-3, n_grid=200, h_power=cfg.H_POWER, tol=1e-6, max_nodes=50000,
    )
    kwargs = dict(
        Y0=0.5, D0=1.0, gamma=cfg.GAMMA, sigma_D=cfg.SIGMA_D, mu_D=cfg.MU_D, beta=cfg.BETA,
        h_func=_h_func, r_func=result["r_interp"], delta2_func=result["delta2_interp"],
        T=0.5, n_steps=50, n_paths=100, checkpoints=(0.5,), seed=123,
    )
    out1, _ = gs.simulate_P_paths(**kwargs)
    out2, _ = gs.simulate_P_paths(**kwargs)
    assert np.allclose(out1[0.5]["Y"], out2[0.5]["Y"])
    assert np.allclose(out1[0.5]["D"], out2[0.5]["D"])
