#!/usr/bin/env bash
# setup_env.sh — Bootstrap the Python virtual environment and install dependencies.
# Usage:
#   ./scripts/setup_env.sh           # create / refresh venv
#   ./scripts/setup_env.sh --clean   # delete existing venv and recreate from scratch
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

# ── Handle --clean flag ────────────────────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "==> Removing existing virtual environment at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

echo "==> Setting up environment in ${VENV_DIR}"

# ── Python version check ───────────────────────────────────────────────────────
PYTHON_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)
if [[ "${PYTHON_MAJOR}" -lt 3 || ( "${PYTHON_MAJOR}" -eq 3 && "${PYTHON_MINOR}" -lt 10 ) ]]; then
    echo "ERROR: Python 3.10+ is required (found ${PYTHON_VERSION})." >&2
    exit 1
fi
echo "    Python version: ${PYTHON_VERSION} — OK"

# ── Create venv ────────────────────────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
    echo "    Virtual environment created."
else
    echo "    Virtual environment already exists — skipping creation."
fi

# ── Activate ───────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── Upgrade pip / build tools ──────────────────────────────────────────────────
pip install --quiet --upgrade pip setuptools wheel

# ── Install dependencies ───────────────────────────────────────────────────────
echo "==> Installing Python dependencies …"
pip install --quiet -r "${PROJECT_ROOT}/requirements.txt"

# ── Download NLTK data needed for sentence tokenisation ───────────────────────
echo "==> Downloading NLTK punkt tokeniser data …"
"${VENV_DIR}/bin/python" - <<'PYEOF'
import nltk, os
nltk_data_dir = os.path.expanduser("~/nltk_data")
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(pkg, quiet=True, download_dir=nltk_data_dir)
    except Exception as exc:
        print(f"Warning: could not download {pkg}: {exc}")
PYEOF

# ── Create runtime directories ─────────────────────────────────────────────────
mkdir -p \
    "${PROJECT_ROOT}/logs" \
    "${PROJECT_ROOT}/data/cache" \
    "${PROJECT_ROOT}/data/embeddings" \
    "${PROJECT_ROOT}/data/vectordb" \
    "${PROJECT_ROOT}/data/output" \
    "${PROJECT_ROOT}/examples/output"

# ── AWS credential hint ────────────────────────────────────────────────────────
echo ""
echo "==> AWS Bedrock credentials (required when running with --provider llm):"
echo "    Option A — environment variables:"
echo "       export AWS_ACCESS_KEY_ID=..."
echo "       export AWS_SECRET_ACCESS_KEY=..."
echo "       export AWS_DEFAULT_REGION=us-east-1"
echo "    Option B — named AWS CLI profile:"
echo "       aws configure --profile my-profile"
echo "       then run: python main.py --provider llm --aws-profile my-profile"
echo ""
echo "==> Setup complete. Activate the environment with:"
echo "    source .venv/bin/activate"


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

echo "==> Setting up environment in ${VENV_DIR}"

# ── Create venv ────────────────────────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
    echo "    Virtual environment created."
else
    echo "    Virtual environment already exists — skipping creation."
fi

# ── Activate ───────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── Upgrade pip ────────────────────────────────────────────────────────────────
pip install --quiet --upgrade pip

# ── Install dependencies ───────────────────────────────────────────────────────
pip install --quiet -r "${PROJECT_ROOT}/requirements.txt"

# ── Create runtime directories ─────────────────────────────────────────────────
mkdir -p \
    "${PROJECT_ROOT}/logs" \
    "${PROJECT_ROOT}/data/cache" \
    "${PROJECT_ROOT}/data/embeddings" \
    "${PROJECT_ROOT}/data/vectordb"

# ── Validate required env vars ─────────────────────────────────────────────────
REQUIRED_VARS=()
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        MISSING_VARS+=("${var}")
    fi
done

if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: The following environment variables are not set:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - ${var}"
    done
    echo "  Copy .env.example to .env and fill in the values."
fi

echo ""
echo "==> Setup complete. Activate the environment with:"
echo "    source .venv/bin/activate"
