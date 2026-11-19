"""Model parameters for the three experiments.

All parameters are module-level constants, grouped by the experiment that
introduces them. Every value carries a one-line note on its meaning, unit and
source (a paper section or equation, or "not specified in the paper").

The paper (Guasoni and Leoni, "Stock bubbles in limited participation models
with power utilities") studies a two-trader limited-participation economy
with power utilities, following the equilibrium constructed in Theorem 3.6 of
Guasoni, Larsen and Leoni (2025) ("GLL (2025)"). It gives no numerical values
for gamma, sigma_D, mu_D, beta, D_0, or the initial state Y_0: these are
chosen here only to satisfy Assumption 2.1 (Eq. 1), and every number that
depends on them is reported next to this configuration, not as a claim of the
paper.
"""

import numpy as np

# --------------------------------------------------------------------------
# Common economic parameters (Section 2, Assumption 2.1)
# --------------------------------------------------------------------------
GAMMA = 0.5     # common relative risk-aversion coefficient, gamma in (0,1); not specified numerically in the paper
SIGMA_D = 0.2   # dividend volatility (annualized), sigma_D > 0; not specified numerically in the paper
MU_D = 0.03     # dividend drift (annualized), mu_D in R; not specified numerically in the paper
BETA = 0.05     # subjective discount rate, beta > 0; not specified numerically in the paper
D0 = 1.0        # initial dividend level D_0 > 0; not specified numerically in the paper

# varrho := beta - (1-gamma)*mu_D + 0.5*gamma*(1-gamma)*sigma_D^2, Eq. (1); must be > 0 (Assumption 2.1)
RHO = BETA - (1.0 - GAMMA) * MU_D + 0.5 * GAMMA * (1.0 - GAMMA) * SIGMA_D ** 2
assert RHO > 0.0, "Assumption 2.1 requires varrho > 0; adjust GAMMA/SIGMA_D/MU_D/BETA"

Y0 = 0.5        # initial state value Y_0 in (0,1); the paper (footnote 2, p.2) says it is set by
                # the traders' initial bank-account endowments but gives no numerical value

# h(y) is the deterministic function of Theorem 3.6 in Guasoni, Larsen and Leoni (2025), valued
# in [gamma,1] with h(0)=gamma, h(1)=1 (Section 2, p.2). Its ODE "is not used" in this paper
# (Section 2, p.2), so it is not reproduced here; H_POWER selects one admissible representative
# h(y) = gamma + (1-gamma)*y**H_POWER satisfying exactly these three stated properties.
H_POWER = 1.0   # exponent of the assumed admissible h(y); not specified in the paper
H_POWER_ALT = 2.0  # alternate admissible exponent, used only for the Experiment 1 robustness check

# g(0) is the free positive constant in Theorem 3.6(1) of GLL (2025) used in the exponential
# representation of g (Section 3.3, p.9: delta_2(y) = (1-y)^(1-gamma)/g(y) = (1/g(0)) * exp(...)).
# Its value depends on trader 2's initial wealth, which the paper does not specify; it is set
# here equal to the upper bound 1/varrho that k(y) satisfies (Eq. 13), for comparability of scale.
G0 = 1.0 / RHO  # not specified in the paper

# --------------------------------------------------------------------------
# Experiment 1: equilibrium ODEs and sigma_S(y) (Proposition 2.1)
# --------------------------------------------------------------------------
EPS_Y = 1e-4          # distance of the numerical grid from the singular endpoints 0 and 1; not specified in the paper
N_GRID_PAPER = 10000  # number of evaluation points on [eps, 1-eps]; not specified in the paper
N_GRID_RUN = 2000      # number of evaluation points actually used (equivalent accuracy, lighter runtime)
BVP_TOL = 1e-6        # solve_bvp residual tolerance for the k(y) Dirichlet problem (ODE 18)
BVP_MAX_NODES = 200000  # solve_bvp maximum mesh size

# --------------------------------------------------------------------------
# Experiment 2: Delbaen-Shirakawa integral test and Monte Carlo (Lemma 3.1)
# --------------------------------------------------------------------------
LAMBDA_CUTOFF = 1.0e6   # lower cutoff of the integral test (28), lambda -> infinity regime
LAMBDA_UPPER = np.inf   # upper limit of the integral test (28)

T_HORIZON = 5.0         # Monte Carlo time horizon (years); not specified in the paper
CHECKPOINTS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)  # times at which E[L_1,t] and E[M_t] are evaluated

N_STEPS_PAPER = 2500    # Euler-Maruyama steps over T_HORIZON (dt = 2e-3); not specified in the paper
N_STEPS_RUN = 1000      # steps actually used (dt = 5e-3)

N_PATHS_PAPER = 100000  # Monte Carlo paths; not specified in the paper
N_PATHS_RUN = 20000     # paths actually used for the main run

MC_SEED = 20251119      # Monte Carlo seed for reproducibility; not specified in the paper

# Numerical safeguard against Euler-Maruyama overshoot past the natural boundaries
# 0 and 1 of Y (Eq. 2 has finite, non-degenerate coefficients there; the diffusion
# itself never reaches them). Not a model parameter, not specified in the paper.
MC_BOUNDARY_EPS = 1e-6

# --------------------------------------------------------------------------
# Experiment 3: bubble property (Theorem 2.2)
# --------------------------------------------------------------------------
ETA_OFFSETS = (-0.5, -0.1, 0.0, 0.1, 0.5)  # eta - kappa(y) offsets used to verify that only eta=kappa(Y_t)
                                            # annihilates the finite-variation drift (Section 3.3)
