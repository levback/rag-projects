#!/usr/bin/env bash
# =============================================================================
# scripts/run_examples.sh
#
# Run all 10 RAG-Projects examples in demo mode and collect outputs.
# No API keys required — every example runs with its built-in _MockLLM.
#
# Usage:
#   bash scripts/run_examples.sh            # run all 10
#   bash scripts/run_examples.sh 01 03 07   # run specific examples by number
#   bash scripts/run_examples.sh --live     # examples 06/07 use real web (needs internet)
#
# Outputs saved to:  examples/output/NN_*_output.json
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Colour helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

pass()  { echo -e "${GREEN}  ✓${RESET} $*"; }
fail()  { echo -e "${RED}  ✗${RESET} $*"; }
info()  { echo -e "${BLUE}  →${RESET} $*"; }
title() { echo -e "\n${BOLD}$*${RESET}"; }

# ── Activate venv ──────────────────────────────────────────────────────────
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  PYTHON="python"
else
  PYTHON="python3"
fi

# ── Parse arguments ────────────────────────────────────────────────────────
LIVE=false
SELECTED=()

for arg in "$@"; do
  case "$arg" in
    --live) LIVE=true ;;
    [0-9][0-9]) SELECTED+=("$arg") ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# All examples in order
ALL_EXAMPLES=(01 02 03 04 05 06 07 08 09 10)
RUN_LIST=("${SELECTED[@]:-${ALL_EXAMPLES[@]}}")

mkdir -p "$ROOT/examples/output"

PASS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS

title "RAG-Projects — Example Runner"
echo "  Python: $($PYTHON --version 2>&1)"
echo "  Output: $ROOT/examples/output/"
echo "  Live web mode: $LIVE"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Map number → script filename
# ═══════════════════════════════════════════════════════════════════════════
get_script() {
  local num="$1"
  local match
  match=$(find "$ROOT/examples" -maxdepth 1 -name "${num}_*.py" | head -1)
  echo "$match"
}

# ═══════════════════════════════════════════════════════════════════════════
# Run each example
# ═══════════════════════════════════════════════════════════════════════════
for num in "${RUN_LIST[@]}"; do
  script=$(get_script "$num")

  if [[ -z "$script" ]]; then
    warn "  No script found for example $num — skipping"
    continue
  fi

  name=$(basename "$script" .py)
  output_file="$ROOT/examples/output/${name}_output.json"

  echo -e "${BOLD}[$num/10] $(basename "$script")${RESET}"

  # Pass --live to examples 06 and 07 if requested
  EXTRA_ARGS=""
  if $LIVE && [[ "$num" == "06" || "$num" == "07" ]]; then
    EXTRA_ARGS="--live"
  fi

  START=$(date +%s%3N)
  if $PYTHON "$script" $EXTRA_ARGS 2>&1 | \
       grep -E "^\[|  Q:|  A:|Synthesis:|Output saved|→" | \
       sed 's/^/    /'; then
    END=$(date +%s%3N)
    ELAPSED=$(( END - START ))

    if [[ -f "$output_file" ]]; then
      SIZE=$(wc -c < "$output_file")
      pass "Done in ${ELAPSED}ms — output: $(basename "$output_file") (${SIZE} bytes)"
      RESULTS[$num]="PASS"
      (( PASS_COUNT++ )) || true
    else
      fail "Script ran but output file not found: $output_file"
      RESULTS[$num]="NO_OUTPUT"
      (( FAIL_COUNT++ )) || true
    fi
  else
    END=$(date +%s%3N)
    fail "Script exited with error after $(( END - START ))ms"
    RESULTS[$num]="FAIL"
    (( FAIL_COUNT++ )) || true
  fi
  echo ""
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
title "Summary"
echo "  Passed: $PASS_COUNT / $(( PASS_COUNT + FAIL_COUNT ))"
echo ""

for num in "${RUN_LIST[@]}"; do
  status="${RESULTS[$num]:-SKIP}"
  script=$(get_script "$num")
  name=$(basename "${script:--}" .py 2>/dev/null || echo "example_$num")
  case "$status" in
    PASS)      echo -e "  ${GREEN}✓${RESET} $name" ;;
    FAIL)      echo -e "  ${RED}✗${RESET} $name" ;;
    NO_OUTPUT) echo -e "  ${YELLOW}⚠${RESET} $name (no output file)" ;;
    SKIP)      echo -e "  ${YELLOW}-${RESET} $name (skipped)" ;;
  esac
done

echo ""
echo "  Output files:"
for f in "$ROOT/examples/output/"*_output.json; do
  [[ -f "$f" ]] && echo "    $(basename "$f")"
done

echo ""
if [[ $FAIL_COUNT -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}  All examples completed successfully.${RESET}"
  exit 0
else
  echo -e "${RED}${BOLD}  $FAIL_COUNT example(s) failed.${RESET}"
  exit 1
fi
