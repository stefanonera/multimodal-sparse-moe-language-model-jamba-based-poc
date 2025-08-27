#!/usr/bin/env bash
set -e
source .venv/bin/activate

python -m src.training.train --config src/training/config.yaml --preset base
LAST=$(ls -td checkpoints/poc/*/ | head -n 1)

# Ablations to show routing shifts
python -m src.training.train --config src/training/config.yaml --preset ablation_small
python -m src.training.train --config src/training/config.yaml --preset ablation_wide

# Evaluate the base run
python -m src.evaluation.eval_poc --run_dir "$LAST"