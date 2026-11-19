#!/usr/bin/env bash
# Runs the test suite and all three experiments in their default configuration.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pytest

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3

echo "Done. See results/RESULTS.md for the comparison with the paper."
