#!/usr/bin/env bash
set -euo pipefail

CFG_PATH="src/training/config.yaml"
PYTHON=${PYTHON:-python}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT_DIR/checkpoints"
: > "$ROOT_DIR/checkpoints/run_index.txt"
SUMMARY_CSV="$ROOT_DIR/checkpoints/summary_runs.csv"
echo "tag,task,layers,experts,capacity,run_dir,train_loss_last,val_loss_last" > "$SUMMARY_CSV"

# helpers 

edit_cfg() {
  # edit_cfg <in_cfg> <out_cfg> <task> <outdir> <layers> <experts> <capacity>
  local IN="$1"; local OUT="$2"; local TASK="$3"; local ODIR="$4"
  local LAYERS="$5"; local EXP="$6"; local CAP="$7"
  "$PYTHON" - "$IN" "$OUT" "$TASK" "$ODIR" "$LAYERS" "$EXP" "$CAP" <<PY
import sys, yaml
inp, outp = sys.argv[1], sys.argv[2]
task, outdir, layers, experts, cap = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
with open(inp, 'r') as f:
    cfg = yaml.safe_load(f)

# Force task + output dir and *explicitly* set MoE knobs so presets don't surprise us.
cfg.setdefault('base', {})['task'] = task
cfg['base']['output_dir'] = outdir
cfg.setdefault('moe', {})['num_layers'] = layers
cfg['moe']['num_experts'] = experts
cfg['moe']['capacity_factor'] = cap

with open(outp, 'w') as f:
    yaml.safe_dump(cfg, f)
PY
}

get_video_root() {
  "$PYTHON" - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_PATH"))
print(cfg["data"]["video"].get("root", "data/mini_ucf"))
PY
}

latest_run_dir() {
  # latest_run_dir <outdir>
  local ODIR="$1"
  ls -td "$ODIR"/*/ | head -n 1
}

summarize_run() {
  # summarize_run <tag> <task> <layers> <experts> <capacity> <run_dir>
  local TAG="$1" TASK="$2" LAYERS="$3" EXP="$4" CAP="$5" RUN="$6"
  local TRAIN_CSV="$RUN/logs/train_loss.csv"
  local VAL_CSV="$RUN/logs/val_loss.csv"

  # pull last values safely
  local TRAIN_LAST=""; local VAL_LAST=""
  if [[ -f "$TRAIN_CSV" && -s "$TRAIN_CSV" ]]; then
    TRAIN_LAST="$(tail -n 1 "$TRAIN_CSV" | awk -F, '{if(NF>=3) print $3; else print ""}')"
  fi
  if [[ -f "$VAL_CSV" && -s "$VAL_CSV" ]]; then
    VAL_LAST="$(tail -n 1 "$VAL_CSV" | awk -F, '{if(NF>=2) print $2; else print ""}')"
  fi

  echo "$TAG -> $RUN" | tee -a "$ROOT_DIR/checkpoints/run_index.txt" >/dev/null
  printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$TAG" "$TASK" "$LAYERS" "$EXP" "$CAP" "$RUN" "$TRAIN_LAST" "$VAL_LAST" >> "$SUMMARY_CSV"
}

train_and_eval() {
  # train_and_eval <task> <outdir> <layers> <experts> <capacity> <label>
  local TASK="$1" ODIR="$2" LAYERS="$3" EXP="$4" CAP="$5" LABEL="$6"

  local TMP_CFG; TMP_CFG="$(mktemp)"
  edit_cfg "$CFG_PATH" "$TMP_CFG" "$TASK" "$ODIR/$LABEL" "$LAYERS" "$EXP" "$CAP"

echo "→ Training [$TASK/$LABEL] ..."
    $PYTHON -m src.training.train --config "$TMP_CFG" --preset base

    local RUN_DIR; RUN_DIR="$(latest_run_dir "$ODIR/$LABEL")"
    echo "→ Evaluating [$TASK/$LABEL] at: $RUN_DIR"
    $PYTHON -m src.evaluation.eval_poc --run_dir "$RUN_DIR"

    summarize_run "$TASK/$LABEL" "$TASK" "$LAYERS" "$EXP" "$CAP" "$RUN_DIR"
    rm -f "$TMP_CFG"
}

# scenario matrix 

declare -a SCENARIOS=(
  "image_text_cifar10 checkpoints/poc/cifar10"
  "text_agnews       checkpoints/poc/ag_news"
)

# add video task only if the folder exists
VID_ROOT="$(get_video_root)"
if [[ -d "$VID_ROOT/train" && -d "$VID_ROOT/val" ]]; then
  SCENARIOS+=("video_folder     checkpoints/poc/video")
else
  echo "Skipping video_folder: $VID_ROOT/{train,val} not found."
fi

for S in "${SCENARIOS[@]}"; do
  TASK="$(echo "$S" | awk '{print $1}')"
  ODIR="$(echo "$S" | awk '{print $2}')"

  # base
  train_and_eval "$TASK" "$ODIR" 1 4 1.25 "base"

  # small
  train_and_eval "$TASK" "$ODIR" 1 2 1.25 "ablation_small"

  # wide
  train_and_eval "$TASK" "$ODIR" 2 8 1.25 "ablation_wide"

  # low capacity to stress routing
  train_and_eval "$TASK" "$ODIR" 1 4 1.00 "cap_low"
done

echo "-------------------------------------------------------------------"
echo "Completed. Index at checkpoints/run_index.txt"
echo "Summary CSV at $SUMMARY_CSV"
