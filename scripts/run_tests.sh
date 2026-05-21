#!/usr/bin/env bash
# run_tests.sh — Run the test suite with coverage reporting.
# Usage:
#   ./scripts/run_tests.sh                   # unit tests only, 95% coverage gate
#   ./scripts/run_tests.sh --all             # unit + integration
#   ./scripts/run_tests.sh --no-coverage     # skip coverage measurement
#   ./scripts/run_tests.sh -v                # verbose output
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# ── Activate venv if present ──────────────────────────────────────────────────
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

cd "${PROJECT_ROOT}"

# ── Parse arguments ───────────────────────────────────────────────────────────
RUN_UNIT=true
RUN_INTEGRATION=false
COVERAGE=true
VERBOSE=false
FAIL_UNDER=95

for arg in "$@"; do
    case "${arg}" in
        --integration) RUN_INTEGRATION=true ;;
        --all)         RUN_INTEGRATION=true ;;
        --no-coverage) COVERAGE=false ;;
        --verbose|-v)  VERBOSE=true ;;
        --fail-under=*) FAIL_UNDER="${arg#*=}" ;;
        *) echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ── Build pytest command ───────────────────────────────────────────────────────
PYTEST_ARGS=("pytest")

if [[ "${VERBOSE}" == "true" ]]; then
    PYTEST_ARGS+=("-v")
fi

if [[ "${COVERAGE}" == "true" ]]; then
    PYTEST_ARGS+=(
        "--cov=src"
        "--cov-report=term-missing"
        "--cov-report=html:htmlcov"
        "--cov-report=xml:coverage.xml"
        "--cov-fail-under=${FAIL_UNDER}"
    )
fi

# Select test paths
TEST_PATHS=()
if [[ "${RUN_UNIT}" == "true" ]]; then
    TEST_PATHS+=("tests/unit")
fi
if [[ "${RUN_INTEGRATION}" == "true" ]]; then
    TEST_PATHS+=("tests/integration")
fi

PYTEST_ARGS+=("${TEST_PATHS[@]}")

echo "==> Running: ${PYTEST_ARGS[*]}"
"${PYTEST_ARGS[@]}"

if [[ "${COVERAGE}" == "true" ]]; then
    echo ""
    echo "==> HTML coverage report written to: ${PROJECT_ROOT}/htmlcov/index.html"
fi


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# ── Activate venv if present ──────────────────────────────────────────────────
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

cd "${PROJECT_ROOT}"

# ── Parse arguments ───────────────────────────────────────────────────────────
RUN_UNIT=true
RUN_INTEGRATION=false
COVERAGE=true
VERBOSE=false

for arg in "$@"; do
    case "${arg}" in
        --integration) RUN_INTEGRATION=true ;;
        --all)         RUN_INTEGRATION=true ;;
        --no-coverage) COVERAGE=false ;;
        --verbose|-v)  VERBOSE=true ;;
        *) echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ── Build pytest command ───────────────────────────────────────────────────────
PYTEST_ARGS=("pytest")

if [[ "${VERBOSE}" == "true" ]]; then
    PYTEST_ARGS+=("-v")
fi

if [[ "${COVERAGE}" == "true" ]]; then
    PYTEST_ARGS+=(
        "--cov=src"
        "--cov-report=term-missing"
        "--cov-report=html:htmlcov"
        "--cov-fail-under=80"
    )
fi

# Select test paths
TEST_PATHS=()
if [[ "${RUN_UNIT}" == "true" ]]; then
    TEST_PATHS+=("tests/unit")
fi
if [[ "${RUN_INTEGRATION}" == "true" ]]; then
    TEST_PATHS+=("tests/integration")
fi

PYTEST_ARGS+=("${TEST_PATHS[@]}")

echo "==> Running: ${PYTEST_ARGS[*]}"
"${PYTEST_ARGS[@]}"
