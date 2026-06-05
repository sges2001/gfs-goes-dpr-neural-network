#!/bin/bash
# run_experiments.sh
#
# Launches a batch of training runs (05_main_train.py) for a given loss family,
# iterating over GFS input variables and experiment numbers.
#
# Usage:
#   bash run_experiments.sh
#
# Output logs are written to OUTPUT_DIR/<exp_number>.out

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

LOSS_NAME="MSE"
OUTPUT_DIR="output/models"

VARS=(   "pwat" "cape" "cin"  "prate" "gh850" "gh500" "gh300"
         "t850" "t500" "t300" "u850"  "u500"  "u300"
         "v850" "v500" "v300" )

EXP_NUMBERS=( 1  2  3  4  5  6  7
               8  9 10 11 12 13
              14 15 16 )

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if [ "${#VARS[@]}" -ne "${#EXP_NUMBERS[@]}" ]; then
  echo "ERROR: VARS and EXP_NUMBERS must have the same length."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

for idx in "${!VARS[@]}"; do
  var="${VARS[$idx]}"
  exp="${EXP_NUMBERS[$idx]}"
  log="${OUTPUT_DIR}/${exp}.out"

  echo "Launching: loss=${LOSS_NAME}  exp=${exp}  var=${var}  log=${log}"
  python -u 05_main_train.py "$LOSS_NAME" "$exp" "$var" > "$log" 2>&1
done

echo "All experiments finished."