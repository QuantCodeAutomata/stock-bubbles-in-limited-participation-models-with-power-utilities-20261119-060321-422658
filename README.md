# Stock bubbles in limited participation models with power utilities

Replication of selected numerical experiments from:

**Paulo Guasoni and Marco Leoni, "Stock bubbles in limited participation models with power utilities."**

The paper studies a continuous-time, two-trader exchange economy with power
utilities in which one trader is unrestricted and one is restricted to a
limited menu of assets (a "limited participation" model). It builds on the
equilibrium constructed in Theorem 3.6 of Guasoni, Larsen and Leoni (2025),
"Risk sharing and endogenous volatility" ("GLL (2025)"), and shows that the
equilibrium stock price is a bubble: its state-price deflated cum-dividend
value is a strict local martingale rather than a true martingale, even though
the market faced by the unrestricted trader is endogenously complete.

This repository is a purely numerical/theoretical replication: no market data
is used anywhere, because every selected experiment is a verification of a
proposition, lemma or theorem through ordinary-differential-equation solving
and Monte Carlo simulation of the model's own state variables.

## Experiments reproduced

1. **Experiment 1 — Proposition 2.1** (`exp/exp1.py`): solves the Dirichlet
   problem (18) for `k(y)`, builds `v(y) = y^gamma k(y)`, `g(y)` and the
   equilibrium stock-price volatility `sigma_S(y)`, and verifies that it is
   continuous and strictly positive on `(0,1)`.
2. **Experiment 2 — Lemma 3.1** (`exp/exp2.py`): applies the Delbaen-Shirakawa
   (2002) integral test (Eq. 28) to the ratio process `Lambda_t` under the
   measure `Q_2`, and uses Monte Carlo simulation of the state process `Y`
   and the dividend `D` to estimate `E[L_{1,t}]` and `E^{Q_2}[Lambda_t/Lambda_0]`
   over a range of horizons, verifying the strict local (super-)martingale
   property.
3. **Experiment 3 — Theorem 2.2** (`exp/exp3.py`): verifies that only the
   deflator `eta = kappa(Y_t)` annihilates the finite-variation drift of the
   deflated cum-dividend stock price, and uses Monte Carlo simulation to show
   that the resulting process `M_t` has `E[M_t] < M_0` for `t > 0`, together
   with the lower bound `M_t/Z_{1,0} >= c_T L_{1,t}` used in the paper's proof
   of the bubble property.

## Repository layout

```
src/            equilibrium ODEs, Delbaen-Shirakawa test, bubble/drift analysis, config
data/           Monte Carlo path generator for the state process Y and dividend D
exp/            one script per experiment, runnable as `python -m exp.<name>`
tests/          pytest unit tests of src/ components
results/        metrics (JSON), figures (PNG) and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required for any of the three experiments.

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1          # reduced grid/paths (default, see src/config.py)
python -m exp.exp2
python -m exp.exp3
python -m exp.exp1 --full   # paper-scale grid/paths (N_GRID_PAPER, N_PATHS_PAPER, ...)
python -m exp.exp2 --full
python -m exp.exp3 --full
```

Each script writes its metrics and figures directly to `results/` (no
per-experiment subdirectories), overwriting any earlier run.

## `reproduce.sh`

```bash
./reproduce.sh
```

runs the test suite and all three experiments in their default (reduced)
configuration, exactly as in the commands above.

## Results

See `results/RESULTS.md` for the numbers produced by the saved runs, their
comparison with the paper's qualitative claims, and the "Deviations from the
paper" / "Not reproduced" sections.
