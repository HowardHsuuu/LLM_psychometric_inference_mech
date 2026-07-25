#!/usr/bin/env bash
# Run the pending extension/control experiments in a resumable order.
#
# Default stages:
#   1. factorial_geometry: Big Five factorial profiles -> activation geometry
#   2. prompt_behavior: no-scale-name/chat-template x argmax/expected-value
#   3. prompt_geometry: no-scale-name/chat-template geometry robustness
#   4. finalization: statistics summary, release bundle rebuild, bundle verify

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS=(qwen14b_instruct llama8b_instruct)
DEVICE="cuda"
SOURCE_SCALE="big_five"
FACTORIAL_MODE="factorial"
LAYERS=()
TARGET_ITEMS_PER_SUBSCALE=1
N_RANDOM=100
MAX_PROFILES=""
MAX_SUBJECTS=""
N_REPLICATIONS=3
N_PROBES=20
N_PERMUTATIONS=1000
PROMPT_VARIANTS=(no_scale_name chat_template)
READOUTS=(argmax expected_value)
LOG_DIR="${ROOT}/outputs/logs"

RUN_FACTORIAL_GEOMETRY=1
RUN_PROMPT_BEHAVIOR=1
RUN_PROMPT_GEOMETRY=1
RUN_FINALIZE=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_extension_controls.sh [options]

Common options:
  --only STAGES                  Comma-separated stages to run:
                                 factorial_geometry,prompt_behavior,prompt_geometry,finalize
  --skip-factorial-geometry
  --skip-prompt-behavior
  --skip-prompt-geometry
  --skip-finalize
  --dry-run                      Print commands without running them.

Runtime/model options:
  --python PATH                  Python executable. Default: python3 or $PYTHON_BIN.
  --models "A B"                 Ordered model list. Default: "qwen14b_instruct llama8b_instruct".
  --model NAME                   Single-model alias for --models NAME.
  --device DEVICE                HF device_map argument. Default: cuda.
  --log-dir PATH                 Log directory. Default: outputs/logs.

Factorial/profile geometry options:
  --source-scale NAME            Default: big_five.
  --factorial-mode MODE          factorial, random, or collapsed. Default: factorial.
  --layers "24 28"               Space-separated layers. Default: each model's configured default layer.
  --target-items-per-subscale N  Default: 1.
  --max-profiles N               Optional smoke-test cap.
  --n-random N                   Random profiles if --factorial-mode random. Default: 100.
  --n-permutations N             Mantel permutations for analysis. Default: 1000.

Prompt behavior / prompt geometry options:
  --max-subjects N               Optional smoke-test cap for prompt behavior.
  --n-replications N             Prompt geometry replications. Default: 3.
  --n-probes N                   Prompt geometry probes. Default: 20.
  --prompt-variants "a b"        Default: "no_scale_name chat_template".
  --readouts "argmax expected_value"

Examples:
  scripts/run_extension_controls.sh
  scripts/run_extension_controls.sh --only factorial_geometry --max-profiles 6 --dry-run
  scripts/run_extension_controls.sh --model qwen14b_instruct
  scripts/run_extension_controls.sh --only prompt_behavior --max-subjects 20
EOF
}

enable_only() {
  RUN_FACTORIAL_GEOMETRY=0
  RUN_PROMPT_BEHAVIOR=0
  RUN_PROMPT_GEOMETRY=0
  RUN_FINALIZE=0

  local spec="$1"
  local old_ifs="${IFS}"
  IFS=','
  read -r -a stages <<< "${spec}"
  IFS="${old_ifs}"

  local stage
  for stage in "${stages[@]}"; do
    case "${stage}" in
      factorial_geometry|profile_geometry)
        RUN_FACTORIAL_GEOMETRY=1
        ;;
      prompt_behavior|behavior)
        RUN_PROMPT_BEHAVIOR=1
        ;;
      prompt_geometry|geometry)
        RUN_PROMPT_GEOMETRY=1
        ;;
      finalize|finalization)
        RUN_FINALIZE=1
        ;;
      all)
        RUN_FACTORIAL_GEOMETRY=1
        RUN_PROMPT_BEHAVIOR=1
        RUN_PROMPT_GEOMETRY=1
        RUN_FINALIZE=1
        ;;
      *)
        echo "Unknown stage for --only: ${stage}" >&2
        exit 2
        ;;
    esac
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --only)
      enable_only "$2"
      shift 2
      ;;
    --skip-factorial-geometry)
      RUN_FACTORIAL_GEOMETRY=0
      shift
      ;;
    --skip-prompt-behavior)
      RUN_PROMPT_BEHAVIOR=0
      shift
      ;;
    --skip-prompt-geometry)
      RUN_PROMPT_GEOMETRY=0
      shift
      ;;
    --skip-finalize)
      RUN_FINALIZE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --model)
      MODELS=("$2")
      shift 2
      ;;
    --models)
      read -r -a MODELS <<< "$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --source-scale)
      SOURCE_SCALE="$2"
      shift 2
      ;;
    --factorial-mode)
      FACTORIAL_MODE="$2"
      shift 2
      ;;
    --layers)
      read -r -a LAYERS <<< "$2"
      shift 2
      ;;
    --target-items-per-subscale)
      TARGET_ITEMS_PER_SUBSCALE="$2"
      shift 2
      ;;
    --max-profiles)
      MAX_PROFILES="$2"
      shift 2
      ;;
    --max-subjects)
      MAX_SUBJECTS="$2"
      shift 2
      ;;
    --n-random)
      N_RANDOM="$2"
      shift 2
      ;;
    --n-replications)
      N_REPLICATIONS="$2"
      shift 2
      ;;
    --n-probes)
      N_PROBES="$2"
      shift 2
      ;;
    --n-permutations)
      N_PERMUTATIONS="$2"
      shift 2
      ;;
    --prompt-variants)
      read -r -a PROMPT_VARIANTS <<< "$2"
      shift 2
      ;;
    --readouts)
      read -r -a READOUTS <<< "$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/psychometric_mpl_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/psychometric_xdg_cache}"

mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${TIMESTAMP}_${name}.log"

  echo
  echo "========================================================================"
  echo "STEP: ${name}"
  echo "LOG:  ${log_file}"
  echo "========================================================================"
  print_command "$@"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  "$@" 2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  if [[ "${status}" -ne 0 ]]; then
    echo "Step failed (${name}) with exit status ${status}" >&2
    exit "${status}"
  fi
}

echo "Repository: ${ROOT}"
echo "Python:     ${PYTHON_BIN}"
echo "Models:     ${MODELS[*]}"
echo "Device:     ${DEVICE}"
if [[ "${#LAYERS[@]}" -gt 0 ]]; then
  echo "Layers:     ${LAYERS[*]}"
else
  echo "Layers:     model defaults"
fi
echo "Log dir:    ${LOG_DIR}"

finalize_outputs() {
  local label="$1"
  if [[ "${RUN_FINALIZE}" != "1" ]]; then
    return 0
  fi
  run_step "${label}_summarize_statistics" "${PYTHON_BIN}" scripts/analysis/summarize_statistics.py
  run_step "${label}_build_release_bundle" "${PYTHON_BIN}" scripts/artifact/build_release_bundle.py --replace
  run_step "${label}_verify_release_bundle" "${PYTHON_BIN}" scripts/artifact/verify_release_bundle.py --bundle release_artifacts
}

for MODEL in "${MODELS[@]}"; do
  echo
  echo "########################################################################"
  echo "MODEL: ${MODEL}"
  echo "########################################################################"

  if [[ "${RUN_FACTORIAL_GEOMETRY}" == "1" ]]; then
    factorial_cmd=(
      "${PYTHON_BIN}" -m psychometric_inference.mechanisms.factorial_geometry
      --model "${MODEL}"
      --source_scale "${SOURCE_SCALE}"
      --mode "${FACTORIAL_MODE}"
      --target_items_per_subscale "${TARGET_ITEMS_PER_SUBSCALE}"
      --n_random "${N_RANDOM}"
      --n_permutations "${N_PERMUTATIONS}"
      --device "${DEVICE}"
      --skip_existing
    )
    if [[ "${#LAYERS[@]}" -gt 0 ]]; then
      factorial_cmd+=(--layers "${LAYERS[@]}")
    fi
    if [[ -n "${MAX_PROFILES}" ]]; then
      factorial_cmd+=(--max_profiles "${MAX_PROFILES}")
    fi
    run_step "${MODEL}_factorial_geometry" "${factorial_cmd[@]}"
  fi

  if [[ "${RUN_PROMPT_BEHAVIOR}" == "1" ]]; then
    prompt_behavior_cmd=(
      "${PYTHON_BIN}" scripts/behavior/run_prompt_sensitivity.py
      --models "${MODEL}"
      --variants "${PROMPT_VARIANTS[@]}"
      --readouts "${READOUTS[@]}"
      --device "${DEVICE}"
      --skip_existing
    )
    if [[ -n "${MAX_SUBJECTS}" ]]; then
      prompt_behavior_cmd+=(--max_subjects "${MAX_SUBJECTS}")
    fi
    run_step "${MODEL}_prompt_behavior" "${prompt_behavior_cmd[@]}"
  fi

  if [[ "${RUN_PROMPT_GEOMETRY}" == "1" ]]; then
    prompt_geometry_cmd=(
      "${PYTHON_BIN}" -m psychometric_inference.mechanisms.prompt_geometry
      --models "${MODEL}"
      --variants "${PROMPT_VARIANTS[@]}"
      --n_replications "${N_REPLICATIONS}"
      --n_probes "${N_PROBES}"
      --skip_existing
    )
    run_step "${MODEL}_prompt_geometry" "${prompt_geometry_cmd[@]}"
  fi

  # Persist summaries and the public artifact after each model. This means the
  # first model's results are saved even if a later model fails or is stopped.
  finalize_outputs "${MODEL}"
done

echo
echo "All requested stages completed."
