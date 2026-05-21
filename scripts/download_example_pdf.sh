#!/usr/bin/env bash
# scripts/download_example_pdf.sh
#
# Downloads a publicly available Google PDF to examples/pdfs/ and optionally
# runs the Document Analysis pipeline on it.
#
# Usage:
#   ./scripts/download_example_pdf.sh                        # download only
#   ./scripts/download_example_pdf.sh --analyze              # download + HuggingFace analysis
#   ./scripts/download_example_pdf.sh --analyze --provider llm --aws-profile my-profile
#
# The script downloads the Google Site Reliability Engineering (SRE) book
# chapter (publicly available at sre.google) as a representative real-world
# multi-page PDF.  Any other public PDF URL can be substituted via PDF_URL.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDF_DIR="${PROJECT_ROOT}/examples/pdfs"
OUTPUT_DIR="${PROJECT_ROOT}/examples/output"
VENV_DIR="${PROJECT_ROOT}/.venv"

# ── Default PDF — Google SRE book, Chapter 1 (preface, 10 pages, public) ──────
# Source: https://sre.google/sre-book/preface/  (downloadable sample PDF)
# We use a Google-published white paper on Site Reliability Engineering that
# is freely available without authentication.
PDF_FILENAME="google_sre_intro.pdf"
PDF_URL="https://sre.google/static/default-roles-and-responsibilities-of-an-sre.pdf"

# Fallback to a well-known public research PDF (Google Brain / DeepMind paper)
FALLBACK_URL="https://arxiv.org/pdf/1706.03762"   # "Attention Is All You Need" — ArXiv
FALLBACK_FILENAME="attention_is_all_you_need.pdf"

# ── Parse flags ────────────────────────────────────────────────────────────────
ANALYZE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --analyze) ANALYZE=true; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Create directories ─────────────────────────────────────────────────────────
mkdir -p "${PDF_DIR}" "${OUTPUT_DIR}"

# ── Download ───────────────────────────────────────────────────────────────────
TARGET="${PDF_DIR}/${PDF_FILENAME}"

if [[ -f "${TARGET}" ]]; then
    echo "==> PDF already present: ${TARGET}"
else
    echo "==> Downloading PDF from ${PDF_URL} …"
    if command -v curl &>/dev/null; then
        HTTP_STATUS=$(curl -L --silent --output "${TARGET}" \
            --write-out "%{http_code}" \
            --connect-timeout 30 \
            --max-time 120 \
            "${PDF_URL}") || true

        if [[ "${HTTP_STATUS}" != "200" ]] || [[ ! -s "${TARGET}" ]]; then
            echo "    Primary URL failed (HTTP ${HTTP_STATUS:-?}). Trying fallback …"
            rm -f "${TARGET}"
            TARGET="${PDF_DIR}/${FALLBACK_FILENAME}"
            curl -L --silent --output "${TARGET}" \
                --connect-timeout 30 \
                --max-time 120 \
                "${FALLBACK_URL}" || {
                    echo "ERROR: Could not download PDF. Check your internet connection." >&2
                    exit 1
                }
        fi
    elif command -v wget &>/dev/null; then
        wget --quiet --timeout=120 -O "${TARGET}" "${PDF_URL}" || {
            echo "    Primary URL failed. Trying fallback …"
            rm -f "${TARGET}"
            TARGET="${PDF_DIR}/${FALLBACK_FILENAME}"
            wget --quiet --timeout=120 -O "${TARGET}" "${FALLBACK_URL}" || {
                echo "ERROR: Could not download PDF. Check your internet connection." >&2
                exit 1
            }
        }
    else
        echo "ERROR: Neither curl nor wget is available." >&2
        exit 1
    fi

    # Verify the file looks like a PDF
    if ! head -c 4 "${TARGET}" | grep -q '%PDF'; then
        echo "ERROR: Downloaded file does not appear to be a valid PDF." >&2
        echo "       File saved at: ${TARGET}" >&2
        exit 1
    fi

    FILE_SIZE=$(wc -c < "${TARGET}")
    echo "==> Downloaded: ${TARGET} (${FILE_SIZE} bytes)"
fi

echo ""
echo "PDF ready: ${TARGET}"

# ── Optionally run the analysis ────────────────────────────────────────────────
if [[ "${ANALYZE}" == "true" ]]; then
    echo ""
    echo "==> Running Document Analysis pipeline …"

    # Activate venv if available
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    fi

    cd "${PROJECT_ROOT}"
    python examples/analyze_pdf.py \
        --pdf "${TARGET}" \
        --output-dir "${OUTPUT_DIR}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

echo ""
echo "==> Done."
echo "    PDF location : ${TARGET}"
echo "    Output dir   : ${OUTPUT_DIR}"
if [[ "${ANALYZE}" == "true" ]]; then
    echo ""
    echo "    To re-run the analysis later:"
    echo "      python examples/analyze_pdf.py --pdf ${TARGET} --provider llm"
fi
