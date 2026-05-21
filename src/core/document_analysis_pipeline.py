"""DocumentAnalysisPipeline — orchestrates all 6 steps from extract to QA."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.inference.qa_engine import QAEngine, QAResult
from src.inference.question_generator import QuestionGenerator
from src.inference.summarizer import Summarizer, SummaryConfig
from src.processing.chunking import SentenceChunker, SentenceChunkingConfig
from src.processing.pdf_extractor import PDFExtractor
from src.processing.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Runtime configuration for the full document analysis pipeline.

    Set ``provider = "llm"`` and pass an LLM client to switch from local
    HuggingFace models to GPT / Claude for all generative steps.
    """

    # General
    provider: str = "huggingface"   # "huggingface" | "llm"
    llm_provider: str = "bedrock"   # which LLM backend when provider=="llm"
    llm_model: str | None = None    # override default model for the LLM backend
    output_dir: str = "data/output"
    cache_dir: str = "data/cache"

    # Summarisation
    summarization_model: str = "t5-small"
    summary_max_length: int = 150
    summary_min_length: int = 30

    # Chunking
    passage_word_limit: int = 200

    # Question generation
    qg_model: str = "valhalla/t5-base-qg-hl"
    min_questions_per_passage: int = 3

    # QA
    qa_model: str = "deepset/roberta-base-squad2"


@dataclass
class PassageAnalysis:
    """Results for a single passage."""

    passage_index: int
    passage: str
    questions: list[str] = field(default_factory=list)
    qa_pairs: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete output from a document analysis run."""

    source: str
    extracted_text: str
    text_preview: str
    summary: str
    num_passages: int
    passages: list[PassageAnalysis] = field(default_factory=list)
    all_qa_pairs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class DocumentAnalysisPipeline:
    """End-to-end document analysis: PDF → text → summary → Q&A.

    Pipeline steps:
    1. **Extract** — pull plain text from the PDF via :class:`PDFExtractor`.
    2. **Preview** — log the first 500 characters for a quick sanity check.
    3. **Preprocess** — clean the text (normalise whitespace, remove control chars).
    4. **Chunk** — split into sentence-based passages via :class:`SentenceChunker`.
    5. **Summarise** — produce a high-level summary via :class:`Summarizer`.
    6. **Generate questions** — auto-generate questions per passage.
    7. **Answer questions** — answer each question against its source passage.

    Example::

        pipeline = DocumentAnalysisPipeline(config=AnalysisConfig())
        result = pipeline.run("path/to/document.pdf")
        print(result.summary)
        pipeline.save_results(result)
    """

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        llm=None,
    ) -> None:
        self._cfg = config or AnalysisConfig()
        self._llm = llm

        # ── Wire up components ────────────────────────────────────────────────
        self._extractor = PDFExtractor(cache_dir=self._cfg.cache_dir)
        self._preprocessor = TextPreprocessor(
            remove_html=True,
            normalize_ws=True,
            remove_urls=False,
        )
        self._chunker = SentenceChunker(
            config=SentenceChunkingConfig(word_limit=self._cfg.passage_word_limit)
        )
        self._summarizer = Summarizer(
            config=SummaryConfig(
                provider=self._cfg.provider,
                hf_model=self._cfg.summarization_model,
                max_length=self._cfg.summary_max_length,
                min_length=self._cfg.summary_min_length,
            ),
            llm=self._llm,
        )
        self._qg = QuestionGenerator(
            provider=self._cfg.provider,
            hf_model=self._cfg.qg_model,
            llm=self._llm,
            min_questions=self._cfg.min_questions_per_passage,
        )
        self._qa = QAEngine(
            provider=self._cfg.provider,
            hf_model=self._cfg.qa_model,
            llm=self._llm,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, pdf_path: str | Path, preview_chars: int = 500) -> AnalysisResult:
        """Execute all pipeline steps and return an :class:`AnalysisResult`.

        Args:
            pdf_path: Path to the input PDF file.
            preview_chars: How many characters to include in the text preview.

        Returns:
            A fully populated :class:`AnalysisResult`.
        """
        path = Path(pdf_path)
        logger.info("=== Document Analysis Pipeline: %s ===", path.name)

        # ── Step 1: Extract ───────────────────────────────────────────────────
        logger.info("Step 1 — Extracting text from PDF…")
        raw_text = self._extractor.extract(path)
        logger.info("  Extracted %d characters.", len(raw_text))

        # ── Step 2: Preview ───────────────────────────────────────────────────
        preview = raw_text[:preview_chars]
        logger.info("Step 2 — Text preview:\n%s\n…", preview)

        # ── Step 3: Preprocess ────────────────────────────────────────────────
        logger.info("Step 3 — Preprocessing text…")
        clean_text = self._preprocessor.process(raw_text)
        logger.info("  Cleaned text: %d characters.", len(clean_text))

        # ── Step 4: Chunk into passages ───────────────────────────────────────
        logger.info("Step 4 — Splitting into passages (word_limit=%d)…", self._cfg.passage_word_limit)
        passages = self._chunker.split(clean_text)
        logger.info("  Created %d passages.", len(passages))

        # ── Step 5: Summarise ─────────────────────────────────────────────────
        logger.info("Step 5 — Summarising document via %s…", self._cfg.provider)
        summary = self._summarizer.summarize_passages(passages)
        logger.info("  Summary length: %d characters.", len(summary))

        # ── Step 6 + 7: Generate questions and answer them ────────────────────
        logger.info(
            "Steps 6–7 — Generating and answering questions (%d passages)…",
            len(passages),
        )
        passage_analyses: list[PassageAnalysis] = []
        all_qa_pairs: list[dict] = []

        for idx, passage in enumerate(passages):
            questions = self._qg.generate(passage)
            qa_results: list[QAResult] = self._qa.answer_batch(questions, passage)

            qa_pairs = [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "score": round(r.score, 4),
                }
                for r in qa_results
            ]
            all_qa_pairs.extend(qa_pairs)

            passage_analyses.append(
                PassageAnalysis(
                    passage_index=idx,
                    passage=passage,
                    questions=questions,
                    qa_pairs=qa_pairs,
                )
            )
            logger.debug(
                "  Passage %d: %d questions generated, %d answered.",
                idx,
                len(questions),
                len(qa_pairs),
            )

        logger.info("=== Pipeline complete. %d total Q&A pairs. ===", len(all_qa_pairs))

        return AnalysisResult(
            source=str(path),
            extracted_text=clean_text,
            text_preview=preview,
            summary=summary,
            num_passages=len(passages),
            passages=passage_analyses,
            all_qa_pairs=all_qa_pairs,
        )

    def save_results(
        self, result: AnalysisResult, output_dir: str | Path | None = None
    ) -> Path:
        """Serialise *result* to a JSON file in *output_dir*.

        Returns the path to the written file.
        """
        out_dir = Path(output_dir or self._cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(result.source).stem
        out_path = out_dir / f"{stem}_analysis.json"
        out_path.write_text(result.to_json(), encoding="utf-8")
        logger.info("Results saved to %s", out_path)
        return out_path

    def print_results(self, result: AnalysisResult) -> None:
        """Pretty-print the pipeline results to stdout."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Document: {result.source}")
        print(sep)

        print("\n--- TEXT PREVIEW ---")
        print(result.text_preview)
        print("…")

        print("\n--- SUMMARY ---")
        print(result.summary)

        print(f"\n--- Q&A ({len(result.all_qa_pairs)} pairs across {result.num_passages} passages) ---")
        for item in result.all_qa_pairs:
            print(f"\nQ: {item['question']}")
            print(f"A: {item['answer']}")
            if item.get("score"):
                print(f"   [confidence: {item['score']:.3f}]")
        print(f"\n{sep}\n")
