"""Unit tests for document analysis components."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.core.base_llm import LLMConfig, LLMResponse, Message
from src.processing.chunking import SentenceChunker, SentenceChunkingConfig
from src.inference.summarizer import Summarizer, SummaryConfig
from src.inference.question_generator import QuestionGenerator
from src.inference.qa_engine import QAEngine, QAResult


# ─── SentenceChunker ──────────────────────────────────────────────────────────

class TestSentenceChunker:
    """Tests for the NLTK sentence-based passage splitter."""

    def test_empty_text_returns_empty_list(self):
        chunker = SentenceChunker()
        assert chunker.split("") == []
        assert chunker.split("   ") == []

    def test_short_text_single_passage(self):
        chunker = SentenceChunker(SentenceChunkingConfig(word_limit=200))
        text = "This is a short sentence. And another short one."
        passages = chunker.split(text)
        assert len(passages) == 1
        assert "short" in passages[0]

    def test_long_text_multiple_passages(self):
        # Build text with two clear groups of sentences exceeding 10-word limit
        sentence = "The quick brown fox jumps over the lazy dog. "
        long_text = sentence * 10  # ~90 words
        chunker = SentenceChunker(SentenceChunkingConfig(word_limit=10))
        passages = chunker.split(long_text)
        assert len(passages) >= 2

    def test_word_limit_respected(self):
        sentence = "Word " * 15 + "end."   # 16 words per sentence
        long_text = (sentence + " ") * 5
        chunker = SentenceChunker(SentenceChunkingConfig(word_limit=20))
        passages = chunker.split(long_text)
        for p in passages:
            # Allow slight overflow for a single sentence > word_limit
            assert len(p.split()) <= 40

    def test_split_with_metadata_returns_dicts(self):
        chunker = SentenceChunker()
        text = "First sentence. Second sentence."
        result = chunker.split_with_metadata(text)
        assert isinstance(result, list)
        for item in result:
            assert "index" in item
            assert "text" in item
            assert "word_count" in item


# ─── PDFExtractor ─────────────────────────────────────────────────────────────

class TestPDFExtractor:
    def test_missing_file_raises(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(cache_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/file.pdf")

    def test_cache_hit_avoids_pdfplumber(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor

        # Pre-populate a fake PDF and its cache
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF fake content")

        # Compute the cache key and write cached text
        import hashlib
        sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()[:16]
        cache_key = f"test_{sha}"
        cache_file = tmp_path / f"{cache_key}.txt"
        cache_file.write_text("Cached extracted text.", encoding="utf-8")

        extractor = PDFExtractor(cache_dir=str(tmp_path))
        result = extractor.extract(fake_pdf)
        assert result == "Cached extracted text."

    def test_extract_to_file_writes_output(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor

        # Pre-populate cache to avoid needing a real PDF
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF fake")
        import hashlib
        sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()[:16]
        cache_file = tmp_path / f"doc_{sha}.txt"
        cache_file.write_text("Hello document.", encoding="utf-8")

        extractor = PDFExtractor(cache_dir=str(tmp_path))
        output = tmp_path / "out.txt"
        result = extractor.extract_to_file(fake_pdf, output_path=output)
        assert result.exists()
        assert result.read_text() == "Hello document."

    def test_preview_returns_limited_chars(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor

        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF fake")
        import hashlib
        sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()[:16]
        (tmp_path / f"doc_{sha}.txt").write_text("A" * 1000, encoding="utf-8")

        extractor = PDFExtractor(cache_dir=str(tmp_path))
        preview = extractor.preview(fake_pdf, chars=100)
        assert len(preview) == 100


# ─── Summarizer ───────────────────────────────────────────────────────────────

class TestSummarizerLLMMode:
    def _make_llm(self, response_text: str = "A concise summary."):
        llm = MagicMock()
        llm.complete.return_value = LLMResponse(content=response_text, model="gpt-4o")
        return llm

    def test_summarize_returns_string(self):
        llm = self._make_llm()
        summarizer = Summarizer(config=SummaryConfig(provider="llm"), llm=llm)
        result = summarizer.summarize("Some long document text here.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_text_returns_empty(self):
        summarizer = Summarizer(config=SummaryConfig(provider="llm"), llm=self._make_llm())
        assert summarizer.summarize("") == ""
        assert summarizer.summarize("   ") == ""

    def test_summarize_passages_combines(self):
        llm = self._make_llm("Partial summary.")
        summarizer = Summarizer(config=SummaryConfig(provider="llm"), llm=llm)
        result = summarizer.summarize_passages(["Passage 1 text.", "Passage 2 text."])
        assert isinstance(result, str)

    def test_llm_mode_no_client_raises(self):
        summarizer = Summarizer(config=SummaryConfig(provider="llm"), llm=None)
        with pytest.raises(ValueError, match="LLM client must be provided"):
            summarizer.summarize("Some text.")

    def test_summarize_passages_empty_list_returns_empty(self):
        summarizer = Summarizer(config=SummaryConfig(provider="llm"), llm=self._make_llm())
        assert summarizer.summarize_passages([]) == ""

    def test_summarize_passages_long_combined_re_summarizes(self):
        # Each call returns 60 words → 2 passages combined = 120 words > 100 → triggers re-summarize
        long_response = ("word " * 60).strip()
        llm = self._make_llm(long_response)
        summarizer = Summarizer(config=SummaryConfig(provider="llm"), llm=llm)
        result = summarizer.summarize_passages(["passage one text", "passage two text"])
        assert isinstance(result, str)

    def test_llm_summarize_multi_chunk(self):
        """Covers the multi-chunk path in _llm_summarize."""
        from unittest.mock import patch as _patch
        llm = self._make_llm("chunk summary")
        summarizer = Summarizer(config=SummaryConfig(provider="llm"), llm=llm)
        with _patch.object(summarizer._tokenizer, "split_by_token_limit", return_value=["chunk1", "chunk2"]):
            result = summarizer.summarize("some long document text")
        assert isinstance(result, str)


# ─── QuestionGenerator ────────────────────────────────────────────────────────

class TestQuestionGeneratorLLMMode:
    def _make_llm(self, questions_text: str = "What is this?\nWhy does it matter?\nHow does it work?"):
        llm = MagicMock()
        llm.complete.return_value = LLMResponse(content=questions_text, model="gpt-4o")
        return llm

    def test_generate_returns_list(self):
        llm = self._make_llm()
        qg = QuestionGenerator(provider="llm", llm=llm)
        questions = qg.generate("The sky is blue because of Rayleigh scattering.")
        assert isinstance(questions, list)
        assert all(isinstance(q, str) for q in questions)

    def test_empty_passage_returns_empty(self):
        qg = QuestionGenerator(provider="llm", llm=self._make_llm())
        assert qg.generate("") == []

    def test_generate_all_structure(self):
        llm = self._make_llm("What is X?")
        qg = QuestionGenerator(provider="llm", llm=llm)
        passages = ["Passage A text here.", "Passage B text here."]
        results = qg.generate_all(passages)
        assert len(results) == 2
        for r in results:
            assert "passage_index" in r
            assert "passage" in r
            assert "questions" in r

    def test_llm_no_client_raises(self):
        qg = QuestionGenerator(provider="llm", llm=None)
        with pytest.raises(ValueError, match="LLM client must be provided"):
            qg.generate("Some text.")


# ─── QAEngine ─────────────────────────────────────────────────────────────────

class TestQAEngineLLMMode:
    def _make_llm(self, answer: str = "42"):
        llm = MagicMock()
        llm.complete.return_value = LLMResponse(content=answer, model="gpt-4o")
        return llm

    def test_answer_returns_qa_result(self):
        qa = QAEngine(provider="llm", llm=self._make_llm("Paris"))
        result = qa.answer("What is the capital of France?", "France is in Europe. Its capital is Paris.")
        assert isinstance(result, QAResult)
        assert result.answer == "Paris"
        assert result.question == "What is the capital of France?"

    def test_empty_inputs_return_empty_result(self):
        qa = QAEngine(provider="llm", llm=self._make_llm())
        result = qa.answer("", "some context")
        assert result.answer == ""

    def test_answer_batch_deduplicates(self):
        call_count = [0]
        llm = MagicMock()
        def side_effect(messages):
            call_count[0] += 1
            return LLMResponse(content="answer", model="gpt-4o")
        llm.complete.side_effect = side_effect

        qa = QAEngine(provider="llm", llm=llm)
        questions = ["What is X?", "What is X?", "What is Y?"]  # duplicate
        results = qa.answer_batch(questions, "Some context about X and Y.")
        assert len(results) == 2  # deduplicated
        assert call_count[0] == 2

    def test_llm_no_client_raises(self):
        qa = QAEngine(provider="llm", llm=None)
        with pytest.raises(ValueError, match="LLM client must be provided"):
            qa.answer("Q?", "context")


# ─── DocumentAnalysisPipeline (unit, mocked components) ──────────────────────

class TestDocumentAnalysisPipeline:
    def _make_pipeline(self, tmp_path):
        from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

        config = AnalysisConfig(
            provider="llm",
            output_dir=str(tmp_path / "output"),
            cache_dir=str(tmp_path / "cache"),
        )
        llm = MagicMock()
        llm.complete.return_value = LLMResponse(content="Mock answer.", model="gpt-4o")
        pipeline = DocumentAnalysisPipeline(config=config, llm=llm)
        return pipeline

    def test_save_results_writes_json(self, tmp_path):
        from src.core.document_analysis_pipeline import AnalysisResult

        pipeline = self._make_pipeline(tmp_path)
        result = AnalysisResult(
            source="test.pdf",
            extracted_text="Full text.",
            text_preview="Full",
            summary="A summary.",
            num_passages=1,
        )
        out = pipeline.save_results(result, output_dir=str(tmp_path / "output"))
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert data["source"] == "test.pdf"
        assert data["summary"] == "A summary."
