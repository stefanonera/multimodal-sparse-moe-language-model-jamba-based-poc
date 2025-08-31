#!/usr/bin/env bash
set -e
source .venv/bin/activate

# Parse command line arguments
SAVE_CHECKPOINTS=""
if [[ "$1" == "--save-checkpoints" ]]; then
    SAVE_CHECKPOINTS="--save-checkpoints"
    echo "Checkpoint saving enabled"
else
    echo "Checkpoint saving disabled (use --save-checkpoints to enable)"
fi

python -m src.training.train --config src/training/config.yaml --preset base $SAVE_CHECKPOINTS
LAST=$(ls -td checkpoints/poc/*/ | head -n 1)

# Ablations to show routing shifts
python -m src.training.train --config src/training/config.yaml --preset ablation_small $SAVE_CHECKPOINTS
python -m src.training.train --config src/training/config.yaml --preset ablation_wide $SAVE_CHECKPOINTS

# Evaluate the base run
python -m src.evaluation.eval_poc --run_dir "$LAST"