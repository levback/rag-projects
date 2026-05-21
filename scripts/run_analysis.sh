#!/usr/bin/env bash
# scripts/run_analysis.sh
#
# Run the Document Analysis pipeline against a PDF and store all outputs
# (JSON result + human-readable report + run log) under examples/output/.
#
# USAGE
#   bash scripts/run_analysis.sh [OPTIONS]
#
# OPTIONS
#   --provider      huggingface|llm        Inference backend (default: huggingface)
#   --llm-provider  bedrock|openai|anthropic|local
#                                          LLM backend when --provider=llm (default: bedrock)
#   --llm-model     MODEL_ID               Override the default model for the chosen provider
#   --aws-region    REGION                 AWS region for Bedrock (default: us-east-1)
#   --aws-profile   PROFILE                Named AWS CLI profile for Bedrock
#   --pdf           PATH                   PDF to analyse (default: examples/pdfs/attention_is_all_you_need.pdf)
#   --word-limit    N                      Max words per passage (default: 200)
#   --output-dir    DIR                    Output directory (default: examples/output)
#   --help                                 Show this help message
#
# PROVIDER CREDENTIALS
#   huggingface : no credentials needed — runs entirely locally
#   bedrock     : AWS credentials via env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
#                 AWS_DEFAULT_REGION) OR a named profile (--aws-profile) OR IAM role
#   openai      : OPENAI_API_KEY env var
#   anthropic   : ANTHROPIC_API_KEY env var
#   local       : no credentials needed — runs a local model via Transformers
#
# EXAMPLES
#   # HuggingFace (no API key needed)
#   bash scripts/run_analysis.sh
#
#   # Amazon Bedrock (default model: claude-3-5-sonnet)
#   bash scripts/run_analysis.sh --provider llm --llm-provider bedrock --aws-profile my-profile
#
#   # Amazon Bedrock with a specific model
#   bash scripts/run_analysis.sh --provider llm --llm-provider bedrock \
#       --llm-model amazon.nova-pro-v1:0 --aws-region us-east-1
#
#   # OpenAI
#   OPENAI_API_KEY=sk-... bash scripts/run_analysis.sh --provider llm --llm-provider openai
#
#   # Anthropic (Claude API direct)
#   ANTHROPIC_API_KEY=sk-... bash scripts/run_analysis.sh --provider llm --llm-provider anthropic

set -euo pipefail

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
PROVIDER="huggingface"
LLM_PROVIDER="bedrock"
LLM_MODEL=""
AWS_REGION=""
AWS_PROFILE=""
PDF_PATH="examples/pdfs/attention_is_all_you_need.pdf"
WORD_LIMIT=200
OUTPUT_DIR="examples/output"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)     PROVIDER="$2";     shift 2 ;;
    --llm-provider) LLM_PROVIDER="$2"; shift 2 ;;
    --llm-model)    LLM_MODEL="$2";    shift 2 ;;
    --aws-region)   AWS_REGION="$2";   shift 2 ;;
    --aws-profile)  AWS_PROFILE="$2";  shift 2 ;;
    --pdf)          PDF_PATH="$2";     shift 2 ;;
    --word-limit)   WORD_LIMIT="$2";   shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
    --help|-h)
      awk '/^# USAGE/{found=1} found{if(/^[^#]/ && !/^#/){exit} sub(/^# ?/,""); print}' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1  (run with --help for usage)" >&2; exit 1 ;;
  esac
done

# ── Validate provider values ──────────────────────────────────────────────────
if [[ "${PROVIDER}" != "huggingface" && "${PROVIDER}" != "llm" ]]; then
  echo "ERROR: --provider must be 'huggingface' or 'llm' (got '${PROVIDER}')" >&2
  exit 1
fi

if [[ "${PROVIDER}" == "llm" ]]; then
  case "${LLM_PROVIDER}" in
    bedrock|openai|anthropic|local) ;;
    *) echo "ERROR: --llm-provider must be bedrock|openai|anthropic|local (got '${LLM_PROVIDER}')" >&2
       exit 1 ;;
  esac
fi

# ── Validate environment ──────────────────────────────────────────────────────
if [[ ! -f ".venv/bin/python" ]]; then
  echo "ERROR: virtual environment not found at .venv/" >&2
  echo "Run:  bash scripts/setup_env.sh" >&2
  exit 1
fi

if [[ ! -f "${PDF_PATH}" ]]; then
  echo "ERROR: PDF not found at ${PDF_PATH}" >&2
  echo "Run:  bash scripts/download_example_pdf.sh" >&2
  exit 1
fi

# ── Credential checks (warn early, don't block — boto3/SDK provide their own errors) ──
if [[ "${PROVIDER}" == "llm" ]]; then
  case "${LLM_PROVIDER}" in
    openai)
      if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "WARNING: OPENAI_API_KEY is not set. Export it before running:" >&2
        echo "  export OPENAI_API_KEY=sk-..." >&2
      fi
      ;;
    anthropic)
      if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "WARNING: ANTHROPIC_API_KEY is not set. Export it before running:" >&2
        echo "  export ANTHROPIC_API_KEY=sk-ant-..." >&2
      fi
      ;;
    bedrock)
      # Accept any of: explicit profile flag, env vars, or ambient IAM role
      if [[ -z "${AWS_PROFILE}" && -z "${AWS_ACCESS_KEY_ID:-}" && -z "${AWS_PROFILE:-}" ]]; then
        echo "INFO: No explicit AWS credentials found — relying on the boto3 default chain" >&2
        echo "      (env vars → ~/.aws/credentials → IAM role)." >&2
      fi
      ;;
  esac
fi

# ── Prepare output directory ──────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

PDF_STEM="$(basename "${PDF_PATH}" .pdf)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/${PDF_STEM}_run_${TIMESTAMP}.log"
REPORT_FILE="${OUTPUT_DIR}/${PDF_STEM}_report_${TIMESTAMP}.txt"

# ── Build Python command ──────────────────────────────────────────────────────
PY_ARGS=(
  examples/analyze_pdf.py
  --pdf         "${PDF_PATH}"
  --provider    "${PROVIDER}"
  --word-limit  "${WORD_LIMIT}"
  --output-dir  "${OUTPUT_DIR}"
)

if [[ "${PROVIDER}" == "llm" ]]; then
  PY_ARGS+=(--llm-provider "${LLM_PROVIDER}")
  [[ -n "${LLM_MODEL}"   ]] && PY_ARGS+=(--llm-model    "${LLM_MODEL}")
  [[ -n "${AWS_REGION}"  ]] && PY_ARGS+=(--aws-region   "${AWS_REGION}")
  [[ -n "${AWS_PROFILE}" ]] && PY_ARGS+=(--aws-profile  "${AWS_PROFILE}")
fi

# ── Print run summary ─────────────────────────────────────────────────────────
BACKEND_DESC="${PROVIDER}"
if [[ "${PROVIDER}" == "llm" ]]; then
  BACKEND_DESC="llm / ${LLM_PROVIDER}"
  [[ -n "${LLM_MODEL}" ]] && BACKEND_DESC+=" (${LLM_MODEL})"
fi

echo "============================================================"
echo "  Document Analysis Pipeline"
echo "  PDF:      ${PDF_PATH}"
echo "  Backend:  ${BACKEND_DESC}"
echo "  Output:   ${OUTPUT_DIR}/"
echo "============================================================"
echo ""

# ── Run analysis ──────────────────────────────────────────────────────────────
# stdout → tee to terminal + report file
# stderr → tee to terminal + log file
# The Python script saves its own JSON to examples/output/ via --output-dir

{
  .venv/bin/python "${PY_ARGS[@]}" \
    2> >(tee "${LOG_FILE}" >&2)
} | tee "${REPORT_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "  Outputs written to ${OUTPUT_DIR}/:"
echo "    JSON result : ${PDF_STEM}_analysis.json"
echo "    Text report : $(basename "${REPORT_FILE}")"
echo "    Run log     : $(basename "${LOG_FILE}")"
echo "============================================================"

exit "${EXIT_CODE}"
