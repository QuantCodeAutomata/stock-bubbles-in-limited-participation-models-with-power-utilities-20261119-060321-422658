# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A numerical replication of three results from Paulo Guasoni and Marco Leoni,
"Stock bubbles in limited participation models with power utilities":
Proposition 2.1 (continuity and strict positivity of the equilibrium stock
volatility `sigma_S(y)`), Lemma 3.1 (strict local martingale property of the
density ratio `Lambda_t` and of the normalized state-price density `L_1`),
and Theorem 2.2 (the equilibrium stock price is a bubble). All three
experiments are purely numerical/theoretical: they solve the model's ODEs and
simulate its state process; no market data is used anywhere in this
repository.

## Layout

```
src/            flat modules: equilibrium ODEs (equilibrium.py), the
                Delbaen-Shirakawa integral test (martingale_tests.py), the
                bubble/drift analysis (bubble.py), and config.py (all
                parameters as module-level constants, with a source note each)
data/           generate_synthetic.py: Euler-Maruyama simulation of the state
                process Y and the dividend D (the only "data" these
                experiments use, since no market data applies)
exp/            exp1.py, exp2.py, exp3.py — one script per experiment,
                run as `python -m exp.exp1` etc.; each takes an optional
                `--full` flag selecting the paper-scale configuration
tests/          pytest unit tests of src/ and data/ components
results/        exp*_metrics.json, exp*.png, and RESULTS.md (overwritten by
                each rerun of the experiment scripts)
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (no market data is used anywhere, so there is no `market_data` marker
in this repository and no API keys are required):

```bash
python -m pytest
```

Experiments (reduced configuration by default; `--full` selects the
paper-scale configuration from `src/config.py`):

```bash
python -m exp.exp1 [--full]
python -m exp.exp2 [--full]
python -m exp.exp3 [--full]
```

`./reproduce.sh` runs the tests and all three experiments in their default
configuration.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` constants, grouped by experiment, each with a one-line comment
  on its meaning, unit and source (a paper section/equation, or "not
  specified in the paper"). No parameter is hardcoded elsewhere.
- Library modules are flat files directly under `src/`; experiment scripts
  live only in `exp/`; all pytest files live only in `tests/`.
- `data/generate_synthetic.py` is the only place that simulates the state
  process `Y` and the dividend `D`; `src/` and `exp/` call it rather than
  duplicating the Euler-Maruyama scheme.
- Run experiments from the repository root as `python -m exp.<name>`, and
  tests as `python -m pytest`; no module edits `sys.path`.
- No API keys are needed by this repository, so there is no `.env` or
  `.env.example` file.
- After rerunning the experiment scripts, every file under `results/` is
  overwritten; the numbers and verdicts written by hand into
  `results/RESULTS.md` must then be checked against the new files and updated
  if they changed.
- The exact ODE that defines `h(y)` in Theorem 3.6 of Guasoni, Larsen and
  Leoni (2025) is outside the scope of this paper, which uses only its three
  stated properties (`h(0)=gamma`, `h(1)=1`, `h(y) in [gamma,1]`). This
  repository represents `h(y)` by the one-parameter family
  `h(y) = gamma + (1-gamma) y^p` satisfying exactly those properties; see
  `results/RESULTS.md`, "Deviations from the paper", for the full list of
  interpretations of details the paper leaves open.
