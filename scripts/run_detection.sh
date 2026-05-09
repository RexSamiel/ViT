#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_detection.sh
#
# Runs detection experiments and merges results into the database.
#
# Usage (from repo root):
#   bash scripts/run_detection.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Python interpreter
PYTHON="${PYTHON:-.venv/bin/python}"

# Single output file where all run results are appended
OUTPUT_FILE="results/detection_results/detection_measurements/runs.json"

# Folder containing the database JSONs and merge.py
DB_DIR="results/detection_results"

# ── Which experiment blocks to run (true/false) ───────────────────────────────
RUN_BASELINE=false
RUN_DETECTION=false
RUN_ZERO=false
RUN_CORRECTION=false
RUN_INPUT_DETECTION=true

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

WARMUP=10
REPEATS=100
BATCH_SIZE=100
MAX_BATCHES=1
FAULTS=1
FAULT_SEED=1

MODELS=(
  vit_tiny
  deit_tiny
  swin_tiny
)

# Bit mode label → --bit_range value (empty = unrestricted, no flag passed)
declare -A BIT_MODES
BIT_MODES=(
  [unrestricted]=""
  [without_bit30]="0,31^30"
  [without_mantissa]="23,31"
)

METHODS=(
  #checksum
  checkone
)

# Input bit range label → --input_bit_range value
declare -A INPUT_BIT_MODES
INPUT_BIT_MODES=(
  [bit30_only]="30,30"
)

# ═════════════════════════════════════════════════════════════════════════════

# Resolve to absolute paths before any cd
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABS_OUTPUT="${REPO_ROOT}/${OUTPUT_FILE}"
ABS_DB_DIR="${REPO_ROOT}/${DB_DIR}"
ABS_MERGE="${ABS_DB_DIR}/merge.py"
ABS_PYTHON="${REPO_ROOT}/${PYTHON}"

mkdir -p "$(dirname "$ABS_OUTPUT")"

run_and_merge() {
  local experiment="$1"
  local model_arg="$2" # -m
  local model_val="$3" # vit_tiny etc.
  shift 3

  echo ""
  echo "─────────────────────────────────────────────────────────"
  echo "  python -m cli $model_arg $model_val --output \"$OUTPUT_FILE\" $*"
  echo "─────────────────────────────────────────────────────────"
  "$ABS_PYTHON" -m cli "$model_arg" "$model_val" --output "$ABS_OUTPUT" "$@"
  "$ABS_PYTHON" "$ABS_MERGE" "$ABS_OUTPUT" --experiment "$experiment"
}

cd "${REPO_ROOT}/src"

# ═════════════════════════════════════════════════════════════════════════════
# BASELINE  (no detection, fault injection only for timing/accuracy)
# ═════════════════════════════════════════════════════════════════════════════

if $RUN_BASELINE; then
  echo "═══════════════════════════════════════"
  echo "  BASELINE"
  echo "═══════════════════════════════════════"
  for MODEL in "${MODELS[@]}"; do
    for BIT_LABEL in "${!BIT_MODES[@]}"; do
      BIT_ARG="${BIT_MODES[$BIT_LABEL]}"
      FI_ARGS="-r $REPEATS --max_batches $MAX_BATCHES --batch_size $BATCH_SIZE -w $WARMUP fi --faults $FAULTS --fault_seed $FAULT_SEED"
      [ -n "$BIT_ARG" ] && FI_ARGS="$FI_ARGS --bit_range $BIT_ARG"
      run_and_merge "baseline" -m "$MODEL" $FI_ARGS
    done
  done
fi

# ═════════════════════════════════════════════════════════════════════════════
# DETECTION  (fault injection + detection, no correction)
# ═════════════════════════════════════════════════════════════════════════════

if $RUN_DETECTION; then
  echo "═══════════════════════════════════════"
  echo "  DETECTION"
  echo "═══════════════════════════════════════"
  for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for BIT_LABEL in "${!BIT_MODES[@]}"; do
        BIT_ARG="${BIT_MODES[$BIT_LABEL]}"
        FI_ARGS="-r $REPEATS --max_batches $MAX_BATCHES --batch_size $BATCH_SIZE -w $WARMUP fi --faults $FAULTS --fault_seed $FAULT_SEED"
        [ -n "$BIT_ARG" ] && FI_ARGS="$FI_ARGS --bit_range $BIT_ARG"
        run_and_merge "detection" \
          -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all
      done
    done
  done
fi

# ═════════════════════════════════════════════════════════════════════════════
# ZERO  (fault injection + detection + correction=zero)
# ═════════════════════════════════════════════════════════════════════════════

if $RUN_ZERO; then
  echo "═══════════════════════════════════════"
  echo "  ZERO CORRECTION"
  echo "═══════════════════════════════════════"
  for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for BIT_LABEL in "${!BIT_MODES[@]}"; do
        BIT_ARG="${BIT_MODES[$BIT_LABEL]}"
        FI_ARGS="-r $REPEATS --max_batches $MAX_BATCHES --batch_size $BATCH_SIZE -w $WARMUP fi --faults $FAULTS --fault_seed $FAULT_SEED"
        [ -n "$BIT_ARG" ] && FI_ARGS="$FI_ARGS --bit_range $BIT_ARG"
        run_and_merge "zero" \
          -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all --correction zero
      done
    done
  done
fi

# ═════════════════════════════════════════════════════════════════════════════
# CORRECTION  (fault injection + detection + correction=correct)
# ═════════════════════════════════════════════════════════════════════════════

if $RUN_CORRECTION; then
  echo "═══════════════════════════════════════"
  echo "  ARITHMETIC CORRECTION"
  echo "═══════════════════════════════════════"
  for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for BIT_LABEL in "${!BIT_MODES[@]}"; do
        BIT_ARG="${BIT_MODES[$BIT_LABEL]}"
        FI_ARGS="-r $REPEATS --max_batches $MAX_BATCHES --batch_size $BATCH_SIZE -w $WARMUP fi --faults $FAULTS --fault_seed $FAULT_SEED"
        [ -n "$BIT_ARG" ] && FI_ARGS="$FI_ARGS --bit_range $BIT_ARG"
        run_and_merge "correction" \
          -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all --correction correct
      done
    done
  done
fi

# ═════════════════════════════════════════════════════════════════════════════
# INPUT DETECTION  (input activation fault injection + checkone detection)
# ═════════════════════════════════════════════════════════════════════════════

if $RUN_INPUT_DETECTION; then
  echo "═══════════════════════════════════════"
  echo "  INPUT FAULT DETECTION"
  echo "═══════════════════════════════════════"
  for MODEL in "${MODELS[@]}"; do
    for INPUT_BIT_LABEL in "${!INPUT_BIT_MODES[@]}"; do
      INPUT_BIT_ARG="${INPUT_BIT_MODES[$INPUT_BIT_LABEL]}"
      FI_ARGS="-r $REPEATS --max_batches $MAX_BATCHES --batch_size $BATCH_SIZE -w $WARMUP fi --faults 0 --input_faults 1 --input_bit_range $INPUT_BIT_ARG"
      run_and_merge "input_detection" \
        -m "$MODEL" $FI_ARGS hr --method checkone --detect all
    done
  done
fi

echo ""
echo "All done. Results merged into ${ABS_DB_DIR}/*.json"
