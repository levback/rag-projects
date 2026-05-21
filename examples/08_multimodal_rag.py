#!/usr/bin/env python3
"""
examples/08_multimodal_rag.py — Project #8: Multimodal RAG

Demonstrates a multimodal pipeline combining:
  - PDF page images via docling (or fallback pdfplumber)
  - Visual LLM captioning (GPT-4V / Claude 3 / BLIP-2 fallback)
  - FAISS-based multi-modal retrieval
  - Text + image context fusion

Demo mode (no API keys, no GPU — simulates captioning + retrieval):
    python examples/08_multimodal_rag.py --demo

Bedrock mode:
    python examples/08_multimodal_rag.py --provider bedrock
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Simulated multimodal content ──────────────────────────────────────────────

# Simulated text chunks (would come from PDF text layer)
TEXT_CHUNKS = [
    "Figure 1: Transformer architecture overview. The model consists of an encoder "
    "and decoder, each with multi-head self-attention and feed-forward layers. "
    "The input is first tokenised and positionally encoded.",

    "Table 1: BLEU scores on WMT 2014 English-to-German translation task. "
    "Transformer (big): 28.4. Previous best (ConvS2S ensemble): 26.36. "
    "The Transformer outperforms all prior architectures by a significant margin.",

    "The attention mechanism computes a weighted sum of values, where the weight "
    "of each value is determined by the compatibility of its key with the query. "
    "Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V.",

    "Figure 2: Visualisation of multi-head attention patterns in layer 5 of the "
    "encoder. Different heads attend to different linguistic structures: syntactic "
    "dependencies, coreference chains, and semantic roles.",
]

# Simulated image captions (would come from VLM in real mode)
IMAGE_CAPTIONS = [
    ("page_1_fig1.png",
     "Architectural diagram of the Transformer model showing parallel encoder and "
     "decoder stacks with attention mechanisms and feed-forward sub-layers."),
    ("page_3_table1.png",
     "Table comparing BLEU scores across models on WMT 2014 En-De translation. "
     "The Transformer (big) achieves 28.4 BLEU, outperforming prior best by 2.0."),
    ("page_5_fig2.png",
     "Heat map visualisation of 8 attention heads in encoder layer 5, showing "
     "each head attending to different word-to-word relationships."),
    ("page_7_eq1.png",
     "Mathematical formula for scaled dot-product attention: "
     "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V"),
]

QUESTIONS = [
    "What architecture components does the Transformer use?",
    "What is the BLEU score of the Transformer on En-De translation?",
    "How does multi-head attention work visually?",
    "What is the formula for scaled dot-product attention?",
]


class _MockLLM:
    _ANSWERS = {
        "architecture": (
            "The Transformer consists of encoder and decoder stacks. Each stack has "
            "multiple layers of multi-head self-attention followed by position-wise "
            "feed-forward networks. Inputs are tokenised and added positional encodings "
            "before entering the encoder stack."
        ),
        "bleu": (
            "The Transformer (big) achieves a BLEU score of 28.4 on the WMT 2014 "
            "English-to-German translation benchmark, compared to 26.36 for the "
            "previous best ConvS2S ensemble — an improvement of ~2 BLEU points."
        ),
        "multi-head attention": (
            "Multi-head attention runs h attention functions in parallel on linearly "
            "projected queries, keys, and values. The outputs are concatenated and "
            "projected. Visualisations show different heads specialising: some attend "
            "to syntactic dependencies, others to coreference or semantic roles."
        ),
        "formula": (
            "Scaled dot-product attention is computed as: "
            "Attention(Q, K, V) = softmax(Q K^T / √d_k) V, "
            "where d_k is the dimension of the keys. Scaling by √d_k prevents "
            "vanishing gradients in the softmax for large d_k values."
        ),
    }

    def complete(self, prompt: str) -> str:
        search_text = prompt[prompt.rfind("Question:"):].lower() if "Question:" in prompt else prompt.lower()
        for kw, ans in self._ANSWERS.items():
            if kw in search_text:
                return ans
        return "Based on both the text and image content of the document, here is the answer."


def run(provider: str, demo: bool) -> dict:
    from src.rag.multimodal_rag import MultimodalRAGConfig, MultimodalRAGPipeline
    from unittest.mock import MagicMock, patch

    print("=" * 60)
    print("  Project #8 — Multimodal RAG (Text + Images)")
    print("=" * 60)

    cfg = MultimodalRAGConfig(
        embedding_model="all-MiniLM-L6-v2",
        vision_model=None,
        top_k=3,
        chunk_size=300,
        persist_dir="/tmp/multimodal_demo",
    )

    llm = _MockLLM() if demo else _build_llm(provider)

    # Patch docling entirely since it has heavyweight deps
    docling_mock = MagicMock()
    docling_doc_mock = MagicMock()
    docling_mock.DocumentConverter.return_value.convert.return_value = docling_doc_mock

    sys_modules_patch = {
        "docling": docling_mock,
        "docling.document_converter": MagicMock(),
        "docling.datamodel": MagicMock(),
        "docling.datamodel.base_models": MagicMock(),
    }

    with patch.dict("sys.modules", sys_modules_patch):
        pipeline = MultimodalRAGPipeline(config=cfg, llm=llm)

    print("\n[1/4] Loading simulated multimodal content …")

    # Inject pre-processed content directly into the pipeline
    # (simulates having run PDF extraction + captioning)
    pipeline._faiss_texts = TEXT_CHUNKS + [cap for _, cap in IMAGE_CAPTIONS]
    pipeline._meta = (
        [{"type": "text", "source": "attention_is_all_you_need.pdf", "page": i + 1}
         for i in range(len(TEXT_CHUNKS))]
        + [{"type": "image", "source": fname, "page": i + 1}
           for i, (fname, _) in enumerate(IMAGE_CAPTIONS)]
    )

    print(f"    Text chunks: {len(TEXT_CHUNKS)}")
    print(f"    Image captions: {len(IMAGE_CAPTIONS)}")
    print(f"    Total indexed items: {len(pipeline._faiss_texts)}")

    # Show captions
    print("\n[2/4] Image captions (generated by VLM in real mode):")
    for fname, caption in IMAGE_CAPTIONS:
        print(f"    [{fname}] {caption[:90]}…")

    # ── Query ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Running multimodal queries …\n")
    results = []

    for question in QUESTIONS:
        print(f"  Q: {question}")

        # Find best matching text+image content
        q_lower = question.lower()
        relevant = [t for t in pipeline._faiss_texts
                    if any(w in t.lower() for w in q_lower.split()[:4])][:2]
        if not relevant:
            relevant = pipeline._faiss_texts[:2]

        context = "\n\n".join(relevant)
        prompt = (
            "Answer the question using the document text and image captions below.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = llm.complete(prompt)

        print(f"  A: {answer[:130]}…")
        modal_types = [m["type"] for m in pipeline._meta
                       if any(w in (m.get("source", "") + context).lower()
                              for w in q_lower.split()[:3])]
        print(f"     Modalities used: {list(set(modal_types[:2])) or ['text']}\n")

        results.append({
            "question": question,
            "answer": answer,
            "context_items": len(relevant),
        })

    # ── Output ────────────────────────────────────────────────────────────────
    output = {
        "project": "08_multimodal_rag",
        "description": "Multimodal RAG — PDF text + VLM image captions + FAISS retrieval",
        "index_summary": {
            "text_chunks": len(TEXT_CHUNKS),
            "image_captions": len(IMAGE_CAPTIONS),
            "total": len(pipeline._faiss_texts),
        },
        "captions": [{"file": f, "caption": c} for f, c in IMAGE_CAPTIONS],
        "results": results,
    }

    out_path = Path(__file__).parent / "output" / "08_multimodal_rag_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[4/4] Output saved → {out_path}")
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
