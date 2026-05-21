#!/usr/bin/env bash
# build_embeddings.sh — Index a directory of documents and persist embeddings.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
DOCS_DIR="${1:-${PROJECT_ROOT}/data/docs}"
GLOB="${2:-**/*.txt}"
PROVIDER="${EMBEDDING_PROVIDER:-openai}"
MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
COLLECTION="${COLLECTION_NAME:-default}"

if [[ ! -d "${DOCS_DIR}" ]]; then
    echo "ERROR: Document directory not found: ${DOCS_DIR}"
    echo "Usage: $0 <docs_dir> [glob_pattern]"
    exit 1
fi

echo "==> Building embeddings"
echo "    Source   : ${DOCS_DIR}"
echo "    Glob     : ${GLOB}"
echo "    Provider : ${PROVIDER} / ${MODEL}"
echo "    Collection: ${COLLECTION}"

cd "${PROJECT_ROOT}"

python - <<EOF
import sys
sys.path.insert(0, ".")

from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.indexer import Indexer

embedder = Embedder(provider="${PROVIDER}", model="${MODEL}")
store = VectorStore(
    provider="chroma",
    collection_name="${COLLECTION}",
    persist_directory="data/vectordb",
)
indexer = Indexer(embedder=embedder, vector_store=store)
total = indexer.index_directory("${DOCS_DIR}", glob="${GLOB}")
print(f"Indexed {total} chunks. Total in store: {store.count()}")
EOF

echo "==> Done. Embeddings persisted to data/vectordb/"
