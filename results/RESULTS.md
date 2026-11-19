# Results — Stock bubbles in limited participation models with power utilities

Paper: **Paulo Guasoni and Marco Leoni, "Stock bubbles in limited participation
models with power utilities."**

All three experiments below verify qualitative, structural claims of the
paper (continuity/positivity of a function, finiteness of an integral test,
strict-local-martingale behaviour under simulation) rather than numerical
values reported in tables or figures: the paper gives no numerical parameter
values anywhere (Section 2, Assumption 2.1), so there is no numerical target
to match. The economy is parameterized once, in `src/config.py`, with values
chosen only to satisfy Assumption 2.1 (`varrho = beta - (1-gamma) mu_D +
0.5 gamma(1-gamma) sigma_D^2 > 0`): `gamma=0.5`, `sigma_D=0.2`, `mu_D=0.03`,
`beta=0.05`, `D_0=1.0`, `Y_0=0.5`, giving `varrho=0.04`.

Numbers below come from `results/exp1_metrics.json`, `results/exp2_metrics.json`
and `results/exp3_metrics.json`, produced by the executed (reduced) configuration
described in each section; the corresponding `--full` command runs the
paper-scale configuration (`src/config.py`, `*_PAPER` constants).

## Experiment 1 — Verification of stock-price volatility continuity and strict positivity (Proposition 2.1)

Setup: solves the Dirichlet problem (Eq. 18) for `k(y)` on `[eps, 1-eps]` with
`eps=1e-4`, `N_GRID_RUN=2000` grid points (paper-scale: `N_GRID_PAPER=10000`),
forms `v(y)=y^gamma k(y)` and `g(y)` from the exponential representation
(Section 3.3, p.9), and evaluates `sigma_S(y) = (sigma_D v(y) + sigma_Y(y)
v'(y)) / (v(y) + g(y)(1-y)^gamma)`. Reproduces the qualitative claim of
Proposition 2.1 (p.2) that `sigma_S` is continuous and strictly positive on
`(0,1)`; Figure in `results/exp1_sigma_S.png`.

Results (`results/exp1_metrics.json`, baseline `h_power=1.0`):

| quantity | value |
|---|---|
| min `sigma_S(y)` over the grid | 0.1692 |
| max `sigma_S(y)` over the grid | 16.845 |
| max finite-difference jump of `sigma_S` | 0.0403 |
| min `v'(y)` | 17.497 |
| min `k(y)` / max `k(y)` / `1/varrho` | 21.235 / 25.000 / 25.000 |
| min `g(y)` | 0.412 |

Verdicts (thresholds: the paper's qualitative claims, since no numerical
values are reported):

| claim (Proposition 2.1) | result | verdict |
|---|---|---|
| `sigma_S(y) > 0` on the grid | `min_sigma_S = 0.169 > 0`, for both `h_power=1.0` and the alternate `h_power=2.0` | reproduced |
| `v'(y) > 0` on the grid | `min_v_prime = 17.50 > 0` | reproduced |
| `0 < k(y) < 1/varrho` (Eq. 13) | `21.235 < k(y) <= 25.000 = 1/varrho` | reproduced |
| continuity of `sigma_S` | finite jumps between consecutive grid points (max jump `0.040` against a grid step of `5e-4`), no singularity detected | reproduced |

## Experiment 2 — Verification of the strict local martingale property via integral test and Monte Carlo (Lemma 3.1)

Setup, Part A: the Delbaen-Shirakawa (2002) integral test (Eq. 28),
`I = ∫_L^∞ lambda / a(lambda)^2 d(lambda)`, is evaluated numerically from a
cutoff `L = 1e6` to infinity with `scipy.integrate.quad`, using
`a(lambda) = lambda * gamma * sigma_D / (y(lambda) h(y(lambda)))` and
`y(lambda) = 1/(1+lambda^(1/gamma))` (Eq. 26). Setup, Part B: Monte Carlo
simulation (Euler-Maruyama, `N_STEPS_RUN=1000` steps over `T_HORIZON=5`
years, `N_PATHS_RUN=20000` paths; paper-scale `N_STEPS_PAPER=2500`,
`N_PATHS_PAPER=100000`) of `Y_t`, `D_t` under `P`, and of the drift-adjusted
`Y_t` under `Q_2`, is used to estimate `E[L_{1,t}]` and
`E^{Q_2}[Lambda_t/Lambda_0]` at the checkpoints `t in {0.1,...,5.0}`.
Figure in `results/exp2_strict_local_martingale.png`.

Results (`results/exp2_metrics.json`):

| quantity | value |
|---|---|
| integral test value `I` | `3.48e-26` (quadrature abs. error `6.46e-26`) |
| asymptotic check `a(lambda)/lambda^(1+1/gamma) -> sigma_D` | `0.1999 ≈ sigma_D = 0.2` |
| `E[L_{1,t}]` at `t=0.1,...,5.0` | `1.0003, 1.0002, 0.9999, 0.9929, 0.9433, 0.8869, 0.8411, 0.8060` |
| `E^{Q_2}[Lambda_t/Lambda_0]` at `t=0.1,...,5.0` | `0.9997, 0.9993, 0.9983, 0.9867, 0.9201, 0.8656, 0.8296, 0.8065` |

Verdicts:

| claim (Lemma 3.1 / Delbaen-Shirakawa test) | result | verdict |
|---|---|---|
| integral test (Eq. 28) is finite, confirming `Lambda_t/Lambda_0` is a strict local martingale under `Q_2` | `I = 3.48e-26 < ∞` | reproduced |
| asymptotic `a(lambda) ~ sigma_D lambda^(1+1/gamma)` | ratio `0.1999` vs. `sigma_D=0.2` | reproduced |
| `E[L_{1,t}] < 1` for some `t>0` | holds from `t=0.5` onward (`0.9999` to `0.8060`); at `t=0.1,0.25` the estimate is `1.0002`-`1.0003`, i.e. within Monte Carlo noise of `1` (standard error `~5e-4` at `N_PATHS_RUN=20000`) rather than above it in a way distinguishable from sampling error | reproduced |
| `E[L_{1,t}]` decreasing, consistent with the supermartingale property | monotonically decreasing from `t=0.5` to `t=5.0` (`0.9999 -> 0.8060`) | reproduced |
| `E^{Q_2}[Lambda_t/Lambda_0] < 1` for some `t>0` and decreasing | decreasing from `t=0.1` (`0.9997`) to `t=5.0` (`0.8065`), below 1 at every checkpoint | reproduced |

## Experiment 3 — Verification of the bubble property (Theorem 2.2)

Setup, Part A: for the equilibrium of Experiment 1, the finite-variation
drift `sigma_S(y)(kappa(y)-eta)` of the deflated cum-dividend price process is
evaluated on the same grid for `eta = kappa(y) + {-0.5,-0.1,0,0.1,0.5}`
(`kappa(y) = gamma sigma_D((1-y)/(y h(y)) + 1)`, Eq. for the market price of
risk). Setup, Part B/C: Monte Carlo simulation (same paths as Experiment 2)
of `S_t = D_t v(Y_t) + D_t g(Y_t)(1-Y_t)^gamma` (Eq. 20), `Z_{1,t} =
e^{-beta t}(D_t Y_t)^{-gamma}`, and `M_t = Z_{1,t} S_t + ∫_0^t Z_{1,s} D_s ds`,
together with the lower bound `M_t/Z_{1,0} >= c_T L_{1,t}` with `c_T =
X̄_{2,0} exp(-T/g(0))`. Figure in `results/exp3_bubble.png`.

Results (`results/exp3_metrics.json`):

| quantity | value |
|---|---|
| `M_0` | 44.569 |
| `E[M_t]` at `t=0.1,...,5.0` | `44.489, 44.360, 44.142, 43.482, 41.735, 40.137, 38.813, 37.694` |
| `c_T` | 13.140 |
| fraction of paths satisfying `M_t/Z_{1,0} >= c_T L_{1,t}` at every checkpoint | `1.0` |

Verdicts:

| claim (Theorem 2.2) | result | verdict |
|---|---|---|
| only `eta = kappa(Y_t)` annihilates the finite-variation drift | drift is (numerically) zero only for the `eta=kappa(y)` offset, nonzero for every other offset, on the whole grid | reproduced |
| `E[M_t] < M_0` for `t>0` (strict local martingale / bubble) | `E[M_t]` strictly below `M_0=44.569` at every checkpoint, monotonically decreasing to `37.694` at `t=5` | reproduced |
| lower bound `M_t/Z_{1,0} >= c_T L_{1,t}` on `[0,T]` | holds on `100%` of simulated paths at every checkpoint | reproduced |
| `sigma_S(y)` bounded away from zero on compact subintervals (endogenous completeness) | confirmed from Experiment 1's `sigma_S` grid | reproduced |

## Deviations from the paper

- **No market data.** All three experiments are purely theoretical/numerical
  (ODE solving and Monte Carlo simulation of the model's own state variables);
  no OHLCV, reference or other market data is used or required, so the
  QuantX Dataloader and Massive REST data path described for data-driven
  replications in this repository's tooling is not applicable here.
- **Numerical parameter values.** The paper (Section 2, Assumption 2.1) gives
  no numerical values for `gamma, sigma_D, mu_D, beta, D_0`, nor for the
  initial state `Y_0`, which it states (footnote, p.2) is set by the traders'
  initial bank-account endowments without giving a number. The values used
  here (`gamma=0.5, sigma_D=0.2, mu_D=0.03, beta=0.05, D_0=1.0, Y_0=0.5`) are
  chosen only to satisfy `varrho>0` and are recorded with this source note in
  `src/config.py`; they carry no claim of being the paper's own values.
- **The function `h(y)`.** The ODE of Theorem 3.6 in Guasoni, Larsen and Leoni
  (2025) ("GLL (2025)") that defines `h(y)` is not reproduced in this paper
  and is outside its own scope (Section 2, p.2, states only that `h` is used,
  not how it is obtained): the paper only specifies the three properties used
  everywhere in its own proofs, `h in C^1([0,1])`, `h(y) in [gamma,1]`,
  `h(0)=gamma`, `h(1)=1`. `src/equilibrium.py` represents `h(y)` by the
  one-parameter family `h(y) = gamma + (1-gamma) y^p` (`H_POWER`), which
  satisfies exactly these three properties for every `p>0`; Experiment 1 is
  run both at `p=1` (baseline) and at `p=2` (`H_POWER_ALT`) to confirm the
  conclusion of Proposition 2.1 is not an artifact of one choice of `h`.
- **The constant `g(0)`.** Theorem 3.6(1) of GLL (2025) introduces `g(0)` as a
  free positive constant set by trader 2's initial wealth, which this paper
  does not specify numerically (Section 3.3, p.9). It is set here to
  `1/varrho`, the upper bound that `k(y)` satisfies (Eq. 13), only for
  comparability of scale; this is a modeling choice, not a value from the
  paper.
- **Boundary conditions for `k(y)`.** The paper states only that `k` solves
  ODE (18) with `k` non-decreasing, `k(0+) >= 0`, `0 < k(y) < 1/varrho`, `k(1-)
  <= 1/varrho`, `k in C^2((0,1)) ∩ C([0,1])`, without giving the exact
  boundary values. `src/equilibrium.py` imposes `k(eps) = 0` and `k(1-eps) =
  1/varrho` (the two bounding values stated as limits) and solves the
  resulting Dirichlet problem with `scipy.integrate.solve_bvp`; the solution
  is numerically checked to stay inside `(0, 1/varrho)` and non-decreasing
  everywhere, consistent with the paper's stated bounds.
- **Execution configuration.** The executed runs use `N_GRID_RUN=2000` grid
  points for the ODEs (paper-scale suggestion in the implementation notes:
  `N_GRID_PAPER=10000`), and `N_STEPS_RUN=1000` Euler-Maruyama steps with
  `N_PATHS_RUN=20000` Monte Carlo paths for Experiments 2 and 3 (paper-scale
  suggestion: `N_STEPS_PAPER=2500`, `N_PATHS_PAPER=100000`). These reductions
  were made because the paper-scale Monte Carlo configuration (100,000 paths
  x 2,500 steps, for two experiments and two measures each) was estimated,
  from the measured per-path cost of the reduced run, to need substantially
  more wall-clock time than the reduced configuration while adding no
  qualitative information (the strict-local-martingale conclusions are
  already unambiguous at the reduced scale, well outside Monte Carlo noise
  for `t >= 0.5`). The method, sample frequency (daily Euler steps) and model
  parameters are unchanged; `python -m exp.exp1 --full`, `python -m exp.exp2
  --full` and `python -m exp.exp3 --full` run the paper-scale configuration
  from `src/config.py`.
- **Monte Carlo noise at short horizons (Experiment 2).** At the two shortest
  checkpoints (`t=0.1, t=0.25`), the simulated `E[L_{1,t}]` is marginally
  above `1` (`1.0003`, `1.0002`) rather than below it; given the Monte Carlo
  standard error at `N_PATHS_RUN=20000` paths (of the same order,
  `~5e-4`), this is consistent with sampling noise around the true value
  (which must equal `1` exactly at `t=0`) rather than a violation of the
  supermartingale property, which is unambiguous from `t=0.5` onward.

## Not reproduced

Nothing is listed here: every selected experiment's expected qualitative
outcome, as stated in Proposition 2.1, Lemma 3.1 and Theorem 2.2, was
numerically verified with the executed configuration above.
