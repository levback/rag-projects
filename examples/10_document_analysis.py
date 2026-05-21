#!/usr/bin/env python3
"""
examples/10_document_analysis.py — Project #10: Document Analysis Pipeline

Demonstrates the full document analysis pipeline:
  - PDF text extraction (via pdfplumber)
  - Section detection and chunking
  - Summarisation (T5 / BART / LLM)
  - Question generation
  - Question answering (QA)
  - Named entity extraction
  - Report generation (Markdown)

Demo mode (no API keys, uses pre-written analysis of a technical paper excerpt):
    python examples/10_document_analysis.py --demo

Full mode (requires actual PDF):
    python examples/10_document_analysis.py --pdf path/to/doc.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Sample document text (excerpt from "Attention Is All You Need") ───────────

SAMPLE_TEXT = """
Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional
neural networks that include an encoder and a decoder. The best performing models also
connect the encoder and decoder through an attention mechanism. We propose a new simple
network architecture, the Transformer, based solely on attention mechanisms, dispensing
with recurrence and convolutions entirely. Experiments on two machine translation tasks
show these models to be superior in quality while being more parallelizable and requiring
significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014
English-to-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model
establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5
days on eight GPUs, a small fraction of the training costs of the best models from the
literature.

1. Introduction
Recurrent neural networks, long short-term memory (LSTM) and gated recurrent neural
networks in particular, have been firmly established as state of the art approaches in
sequence modeling and transduction problems such as language modeling and machine
translation. Numerous efforts have since continued to push the boundaries of recurrent
language models and encoder-decoder architectures. Recurrent models typically factor
computation along the symbol positions of the input and output sequences. Aligning the
positions to steps in computation time, they generate a sequence of hidden states ht,
as a function of the previous hidden state ht-1 and the input for position t. This
inherently sequential nature precludes parallelization within training examples.

2. Background
The goal of reducing sequential computation also forms the foundation of the Extended
Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as
basic building block, computing hidden representations in parallel for all input and
output positions. In these models, the number of operations required to relate signals
from two arbitrary input or output positions grows in the distance between positions,
linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to
learn dependencies between distant positions. In the Transformer this is reduced to a
constant number of operations, albeit at the cost of reduced effective resolution due
to averaging attention-weighted positions.

3. Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure.
The encoder maps an input sequence of symbol representations to a sequence of continuous
representations z. Given z, the decoder then generates an output sequence of symbols one
element at a time. At each step the model is auto-regressive, consuming the previously
generated symbols as additional input when generating the next. The Transformer follows
this overall architecture using stacked self-attention and point-wise, fully connected
layers for both the encoder and decoder.
"""

DEMO_ANALYSIS = {
    "title": "Attention Is All You Need — Demo Analysis",
    "word_count": len(SAMPLE_TEXT.split()),
    "sections": ["Abstract", "1. Introduction", "2. Background", "3. Model Architecture"],
    "summary": (
        "This paper proposes the Transformer, a neural sequence transduction architecture "
        "based entirely on attention mechanisms, eliminating recurrence and convolutions. "
        "The Transformer achieves 28.4 BLEU on WMT 2014 English-German and 41.0 BLEU on "
        "English-French translation, outperforming all prior models while requiring "
        "significantly less training time due to full parallelisation."
    ),
    "key_entities": [
        {"text": "Transformer", "type": "ARCHITECTURE"},
        {"text": "WMT 2014", "type": "BENCHMARK"},
        {"text": "LSTM", "type": "ARCHITECTURE"},
        {"text": "BLEU", "type": "METRIC"},
        {"text": "ConvS2S", "type": "MODEL"},
        {"text": "ByteNet", "type": "MODEL"},
        {"text": "Extended Neural GPU", "type": "MODEL"},
    ],
    "generated_questions": [
        "What architecture does the Transformer replace?",
        "What BLEU score does the Transformer achieve on WMT 2014 English-German?",
        "Why is the Transformer more parallelisable than recurrent models?",
        "How does the Transformer relate signals from distant positions?",
    ],
    "qa_pairs": [
        {
            "question": "What architecture does the Transformer replace?",
            "answer": (
                "The Transformer replaces recurrent neural networks (RNNs), LSTMs, and "
                "gated recurrent units (GRUs), as well as convolutional sequence-to-sequence "
                "models like ConvS2S, relying entirely on attention mechanisms instead."
            ),
        },
        {
            "question": "What BLEU score does the Transformer achieve on WMT 2014 English-German?",
            "answer": (
                "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German translation, "
                "improving over the existing best results (including ensembles) by over 2 BLEU."
            ),
        },
        {
            "question": "Why is the Transformer more parallelisable than recurrent models?",
            "answer": (
                "Recurrent models must process tokens sequentially (h_t depends on h_{t-1}), "
                "preventing parallelisation within training examples. The Transformer's "
                "attention mechanism computes all position interactions simultaneously."
            ),
        },
        {
            "question": "How does the Transformer relate signals from distant positions?",
            "answer": (
                "The Transformer relates signals from any two positions in a constant number "
                "of operations via attention, unlike ConvS2S (linear growth in distance) or "
                "ByteNet (logarithmic growth), making long-range dependency learning easier."
            ),
        },
    ],
}


def run(pdf_path: str | None, demo: bool) -> dict:
    print("=" * 60)
    print("  Project #10 — Document Analysis Pipeline")
    print("=" * 60)

    if demo or not pdf_path:
        return _run_demo()
    else:
        return _run_full(pdf_path)


def _run_demo() -> dict:
    print("\n  Mode: DEMO (pre-computed analysis of 'Attention Is All You Need')\n")

    a = DEMO_ANALYSIS

    print(f"[1/6] Document loaded")
    print(f"    Title: {a['title']}")
    print(f"    Word count: {a['word_count']}")
    print(f"    Sections: {a['sections']}\n")

    print(f"[2/6] Summary:")
    print(f"    {a['summary'][:160]}…\n")

    print(f"[3/6] Named entities ({len(a['key_entities'])}):")
    for ent in a["key_entities"]:
        print(f"    [{ent['type']}] {ent['text']}")

    print(f"\n[4/6] Generated questions:")
    for q in a["generated_questions"]:
        print(f"    • {q}")

    print(f"\n[5/6] QA pairs:")
    for qa in a["qa_pairs"]:
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer'][:120]}…\n")

    output = {
        "project": "10_document_analysis",
        "description": "Document Analysis Pipeline — PDF extraction, summarisation, QG, QA, NER",
        "mode": "demo",
        "analysis": a,
        "sample_text_excerpt": SAMPLE_TEXT[:500].strip() + "…",
    }

    out_path = Path(__file__).parent / "output" / "10_document_analysis_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[6/6] Output saved → {out_path}")
    return output


def _run_full(pdf_path: str) -> dict:
    """Run the real pipeline on an actual PDF."""
    from src.processing.pdf_extractor import PDFExtractor
    from src.processing.text_processor import TextProcessor
    from src.processing.summarizer import Summarizer
    from src.processing.question_generator import QuestionGenerator
    from src.processing.qa_engine import QAEngine

    print(f"\n  Mode: FULL (real pipeline on {pdf_path})\n")

    pdf = PDFExtractor()
    doc = pdf.extract(pdf_path)
    processor = TextProcessor()
    sections = processor.split_sections(doc.text)

    summarizer = Summarizer()
    summary = summarizer.summarize(doc.text[:3000])

    qg = QuestionGenerator()
    questions = qg.generate(doc.text[:2000], n=4)

    qa = QAEngine()
    qa_pairs = []
    for q in questions:
        answer = qa.answer(q, doc.text[:3000])
        qa_pairs.append({"question": q, "answer": answer})

    output = {
        "project": "10_document_analysis",
        "description": "Document Analysis Pipeline — PDF extraction, summarisation, QG, QA",
        "mode": "full",
        "file": pdf_path,
        "word_count": len(doc.text.split()),
        "sections": [s.title for s in sections],
        "summary": summary,
        "qa_pairs": qa_pairs,
    }

    out_path = Path(__file__).parent / "output" / "10_document_analysis_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Output saved → {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="Path to a PDF file for full analysis")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run(pdf_path=args.pdf, demo=args.demo or not args.pdf)


if __name__ == "__main__":
    main()
