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
# FAULT INJECTION
# Select which fault types to inject. Both false = no fault injection (clean /
# overhead runs). Both can be true — the script runs each section separately.
# ═════════════════════════════════════════════════════════════════════════════

INJECT_WEIGHT_FAULTS=false # bit flips in model weights
INJECT_INPUT_FAULTS=false  # bit flips in input activations

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# Which experiment types to run. Applied to every enabled fault type above.
# ═════════════════════════════════════════════════════════════════════════════

RUN_BASELINE=ture  # no detection — measures accuracy / timing with faults
RUN_DETECTION=true # detection active, no correction
RUN_ZERO=true      # detection active, zero-out correction
# RUN_CORRECTION=false  # detection active, arithmetic correction (rarely used)

# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_FILE="results/detection_results/detection_measurements/runs.json"
DB_DIR="results/detection_results"

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

METHODS=(
  checkone
  checksum
)

# Weight fault bit range label → --bit_range value
declare -A BIT_MODES
BIT_MODES=(
  [unrestricted]=""
  [without_bit30]="0,31^30"
  [without_mantissa]="23,31"
)

# Input fault bit range label → --input_bit_range value
declare -A INPUT_BIT_MODES
INPUT_BIT_MODES=(
  [bit30_only]="30,30"
  [unrestricted]=""
  [without_bit30]="0,31^30"
  [without_mantissa]="23,31"
)

# ═════════════════════════════════════════════════════════════════════════════

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABS_OUTPUT="${REPO_ROOT}/${OUTPUT_FILE}"
ABS_DB_DIR="${REPO_ROOT}/${DB_DIR}"
ABS_MERGE="${ABS_DB_DIR}/merge.py"
ABS_PYTHON="${REPO_ROOT}/${PYTHON}"

mkdir -p "$(dirname "$ABS_OUTPUT")"

run_and_merge() {
  local experiment="$1"
  local model_arg="$2"
  local model_val="$3"
  shift 3

  echo ""
  echo "─────────────────────────────────────────────────────────"
  echo "  python -m cli $model_arg $model_val --output \"$OUTPUT_FILE\" $*"
  echo "─────────────────────────────────────────────────────────"
  "$ABS_PYTHON" -m cli "$model_arg" "$model_val" --output "$ABS_OUTPUT" "$@"
  "$ABS_PYTHON" "$ABS_MERGE" "$ABS_OUTPUT" --experiment "$experiment"
}

cd "${REPO_ROOT}/src"

COMMON="-r $REPEATS --max_batches $MAX_BATCHES --batch_size $BATCH_SIZE -w $WARMUP"

# ═════════════════════════════════════════════════════════════════════════════
# WEIGHT / NO-FAULT EXPERIMENTS
# Runs when INJECT_WEIGHT_FAULTS=true (weight faults) or both flags are false
# (clean runs). Loops over BIT_MODES when injecting, single unrestricted pass
# when not (bit range is irrelevant without fault injection).
# ═════════════════════════════════════════════════════════════════════════════

if $INJECT_WEIGHT_FAULTS; then
  ACTIVE_BIT_LABELS=("${!BIT_MODES[@]}")
  FI_CMD="fi --faults $FAULTS --fault_seed $FAULT_SEED"
  echo "═══════════════════════════════════════"
  echo "  WEIGHT FAULT EXPERIMENTS"
  echo "═══════════════════════════════════════"
else
  ACTIVE_BIT_LABELS=(unrestricted)
  FI_CMD="fi --faults 0"
  echo "═══════════════════════════════════════"
  echo "  CLEAN / OVERHEAD EXPERIMENTS"
  echo "═══════════════════════════════════════"
fi

for MODEL in "${MODELS[@]}"; do
  for BIT_LABEL in "${ACTIVE_BIT_LABELS[@]}"; do
    BIT_ARG="${BIT_MODES[$BIT_LABEL]:-}"
    FI_ARGS="$COMMON $FI_CMD"
    [ -n "$BIT_ARG" ] && FI_ARGS="$FI_ARGS --bit_range $BIT_ARG"

    if $RUN_BASELINE; then
      run_and_merge "baseline" -m "$MODEL" $FI_ARGS
    fi

    for METHOD in "${METHODS[@]}"; do
      if $RUN_DETECTION; then
        run_and_merge "detection" \
          -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all
      fi
      if $RUN_ZERO; then
        run_and_merge "zero" \
          -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all --correction zero
      fi
      # if $RUN_CORRECTION; then
      #   run_and_merge "correction" \
      #     -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all --correction correct
      # fi
    done
  done
done

# ═════════════════════════════════════════════════════════════════════════════
# INPUT FAULT EXPERIMENTS
# Only runs when INJECT_INPUT_FAULTS=true. Loops over INPUT_BIT_MODES.
# baseline → baseline.json  |  detection → detection.json  |  zero → zero.json
# All stored with fault_type="input" so they never overwrite weight/clean runs.
# ═════════════════════════════════════════════════════════════════════════════

if $INJECT_INPUT_FAULTS; then
  echo "═══════════════════════════════════════"
  echo "  INPUT FAULT EXPERIMENTS"
  echo "═══════════════════════════════════════"
  for MODEL in "${MODELS[@]}"; do
    for INPUT_BIT_LABEL in "${!INPUT_BIT_MODES[@]}"; do
      INPUT_BIT_ARG="${INPUT_BIT_MODES[$INPUT_BIT_LABEL]}"
      FI_ARGS="$COMMON fi --faults 0 --input_faults 1"
      [ -n "$INPUT_BIT_ARG" ] && FI_ARGS="$FI_ARGS --input_bit_range $INPUT_BIT_ARG"

      if $RUN_BASELINE; then
        run_and_merge "baseline" -m "$MODEL" $FI_ARGS
      fi

      for METHOD in "${METHODS[@]}"; do
        if $RUN_DETECTION; then
          run_and_merge "detection" \
            -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all
        fi
        if $RUN_ZERO; then
          run_and_merge "zero" \
            -m "$MODEL" $FI_ARGS hr --method "$METHOD" --detect all --correction zero
        fi
      done
    done
  done
fi

echo ""
echo "All done. Results merged into ${ABS_DB_DIR}/*.json"
