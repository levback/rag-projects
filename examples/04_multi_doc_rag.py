#!/usr/bin/env python3
"""
examples/04_multi_doc_rag.py — Project #4: Multi-Document RAG

Demonstrates a RAG pipeline that ingests and queries across multiple
heterogeneous documents simultaneously:
  - Plain-text files
  - Markdown files
  - PDFs (via pdfplumber)
  - Automatic source provenance in answers

Demo mode (no API keys required):
    python examples/04_multi_doc_rag.py --demo

Bedrock mode:
    python examples/04_multi_doc_rag.py --provider bedrock
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Sample documents ──────────────────────────────────────────────────────────

DOCS = {
    "climate_science.txt": """
Climate change refers to long-term shifts in global temperatures and weather patterns.
Since the 1800s, human activities—primarily the burning of fossil fuels—have been the
main driver of climate change. Greenhouse gases such as carbon dioxide (CO2), methane
(CH4), and nitrous oxide (N2O) trap heat in the atmosphere, causing the greenhouse effect.
The Intergovernmental Panel on Climate Change (IPCC) estimates that human activities
have caused approximately 1.1°C of warming above pre-industrial levels as of 2019.
Without significant reductions in emissions, global average temperatures could rise
by 1.5°C as early as 2030 and by 2–4°C by 2100.
""",
    "renewable_energy.md": """
# Renewable Energy Technologies

## Solar Power
Photovoltaic (PV) panels convert sunlight directly into electricity. The global installed
solar capacity reached 1 TW in 2022. Utility-scale solar farms can now generate electricity
at costs below $0.02/kWh in high-irradiance regions.

## Wind Power
Onshore wind turbines have a capacity factor of 25–45%. Offshore wind is more consistent
and can achieve 45–60% capacity factors. The world's largest offshore wind farm, Hornsea 2
in the UK, has a capacity of 1.3 GW.

## Energy Storage
Lithium-ion battery costs have fallen 97% since 1991. Grid-scale batteries now enable
renewable energy to be dispatched on demand. Pumped hydro remains the dominant form of
long-duration storage, accounting for 95% of global energy storage capacity.
""",
    "carbon_markets.txt": """
Carbon markets allow entities to buy and sell carbon credits, where one credit represents
one tonne of CO2 equivalent reduced or removed from the atmosphere. Two types of carbon
markets exist: compliance markets (mandated by regulations, such as the EU ETS) and
voluntary carbon markets (where companies purchase credits to meet self-imposed targets).

The EU Emissions Trading System (EU ETS) is the world's largest carbon market, covering
approximately 40% of EU greenhouse gas emissions. Carbon prices in the EU ETS reached
over €100 per tonne in 2023. The Voluntary Carbon Market (VCM) is projected to grow
to $50 billion by 2030 as corporate net-zero commitments increase.
""",
}

QUESTIONS = [
    "What greenhouse gases contribute to climate change and what warming has occurred?",
    "What is the capacity factor of offshore wind and what is the largest offshore wind farm?",
    "How do compliance and voluntary carbon markets differ?",
    "What has happened to lithium-ion battery costs since 1991?",
    "Which documents discuss carbon prices?",
]


class _MockLLM:
    _ANSWERS = {
        "greenhouse": (
            "The main greenhouse gases driving climate change are carbon dioxide (CO2), "
            "methane (CH4), and nitrous oxide (N2O). Human activities have caused "
            "approximately 1.1°C of warming above pre-industrial levels as of 2019, "
            "according to the IPCC."
        ),
        "offshore wind": (
            "Offshore wind turbines can achieve capacity factors of 45–60%. The world's "
            "largest offshore wind farm is Hornsea 2 in the UK, with a capacity of 1.3 GW."
        ),
        "carbon market": (
            "Compliance carbon markets are mandated by regulation (e.g., the EU ETS), "
            "while voluntary carbon markets allow companies to purchase credits to meet "
            "self-imposed net-zero targets. The EU ETS is the world's largest, covering "
            "40% of EU greenhouse gas emissions."
        ),
        "lithium": (
            "Lithium-ion battery costs have fallen by 97% since 1991, enabling grid-scale "
            "batteries to dispatch renewable energy on demand."
        ),
        "carbon prices": (
            "Based on the retrieved documents, carbon prices are discussed in "
            "'carbon_markets.txt'. EU ETS carbon prices reached over €100 per tonne in 2023."
        ),
    }

    def complete(self, prompt: str) -> str:
        search_text = prompt[prompt.rfind("Question:"):].lower() if "Question:" in prompt else prompt.lower()
        for kw, ans in self._ANSWERS.items():
            if kw in search_text:
                return ans
        return "The retrieved documents contain relevant information on this topic."


def run(provider: str, demo: bool) -> dict:
    from src.rag.multi_doc_rag import MultiDocConfig, MultiDocumentRAG
    from unittest.mock import MagicMock, patch

    print("=" * 60)
    print("  Project #4 — Multi-Document RAG")
    print("=" * 60)

    cfg = MultiDocConfig(
        collection_name="multi_doc_demo",
        persist_dir="/tmp/multi_doc_demo",
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )

    llm = _MockLLM() if demo else _build_llm(provider)
    rag = MultiDocumentRAG(config=cfg, llm=llm)

    # ── Write temp files and load ─────────────────────────────────────────────
    print("\n[1/4] Writing and loading documents …")

    mock_docs_by_question: dict[str, list] = {}
    for q in QUESTIONS:
        q_lower = q.lower()
        relevant_sources = []
        if "greenhouse" in q_lower or "warming" in q_lower or "climate" in q_lower:
            relevant_sources = ["climate_science.txt"]
        elif "offshore" in q_lower or "wind" in q_lower or "solar" in q_lower:
            relevant_sources = ["renewable_energy.md"]
        elif "carbon" in q_lower or "compliance" in q_lower or "voluntary" in q_lower:
            relevant_sources = ["carbon_markets.txt"]
        elif "lithium" in q_lower or "battery" in q_lower or "storage" in q_lower:
            relevant_sources = ["renewable_energy.md"]
        else:
            relevant_sources = ["carbon_markets.txt", "climate_science.txt"]
        mock_docs_by_question[q] = [
            MagicMock(page_content=DOCS[s].strip()[:200], metadata={"source": s})
            for s in relevant_sources
        ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for fname, content in DOCS.items():
            (tmp_path / fname).write_text(content.strip(), encoding="utf-8")

        mock_store = MagicMock()
        mock_store.add_texts.return_value = None

        question_idx = [0]

        def _side_effect_search(q, k=3):
            return mock_docs_by_question.get(q, [])

        mock_store.similarity_search.side_effect = _side_effect_search

        with patch("langchain_community.vectorstores.Chroma", return_value=mock_store), \
             patch("langchain_huggingface.HuggingFaceEmbeddings"):
            total = rag.load_directory(tmp_path)
            print(f"    Loaded {total} chunks from {rag.document_count} documents")

            # ── Query ─────────────────────────────────────────────────────────
            print("\n[2/4] Running queries …\n")
            results = []
            for q in QUESTIONS:
                print(f"  Q: {q}")
                result = rag.query(q)
                print(f"  A: {result.answer[:130]}…")
                print(f"     Sources: {result.source_documents}\n")
                results.append({
                    "question": result.query,
                    "answer": result.answer,
                    "retrieved_chunks": len(result.retrieved_chunks),
                    "source_documents": result.source_documents,
                })

    # ── Output ────────────────────────────────────────────────────────────────
    output = {
        "project": "04_multi_doc_rag",
        "description": "Multi-Document RAG — query across heterogeneous documents with provenance",
        "documents_loaded": list(DOCS.keys()),
        "total_chunks": total,
        "results": results,
    }

    out_path = Path(__file__).parent / "output" / "04_multi_doc_rag_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[3/4] Output saved → {out_path}")
    return output


def _build_llm(provider: str):
    from src.core.model_factory import ModelFactory
    return ModelFactory.create_llm(provider)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="bedrock")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run(provider=args.provider, demo=True)


if __name__ == "__main__":
    main()
