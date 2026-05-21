"""Unit tests for Tokenizer, ResponseParser, InferenceEngine, and pipeline extensions."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.base_llm import LLMConfig, LLMResponse, Message
from src.inference.response_parser import ParsedResponse, ResponseParser
from src.processing.tokenizer import Tokenizer


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class TestTokenizerWithTiktoken:
    """Tests that run when tiktoken is importable (it's in requirements)."""

    def test_count_tokens_returns_int(self):
        tok = Tokenizer(model="gpt-4o")
        count = tok.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_string(self):
        tok = Tokenizer()
        count = tok.count_tokens("")
        # tiktoken returns 0 for empty; fallback returns max(1, 0//4)=1. Allow both.
        assert count >= 0

    def test_truncate_short_text_unchanged(self):
        tok = Tokenizer()
        text = "Hello"
        result = tok.truncate(text, max_tokens=500)
        assert result == text

    def test_truncate_long_text_reduced(self):
        tok = Tokenizer()
        long_text = "word " * 5000  # ~5000 words ≈ lots of tokens
        result = tok.truncate(long_text, max_tokens=10)
        assert len(result) < len(long_text)

    def test_fits_in_context_short_returns_true(self):
        tok = Tokenizer()
        assert tok.fits_in_context("hi", max_tokens=1000) is True

    def test_fits_in_context_long_returns_false(self):
        tok = Tokenizer()
        long_text = "word " * 10000
        assert tok.fits_in_context(long_text, max_tokens=5) is False

    def test_split_by_token_limit_returns_list(self):
        tok = Tokenizer()
        text = "word " * 100
        parts = tok.split_by_token_limit(text, max_tokens=20)
        assert isinstance(parts, list)
        assert len(parts) >= 1


class TestTokenizerFallback:
    """Tests the character-based fallback when tiktoken is unavailable."""

    def _make_fallback_tokenizer(self):
        tok = Tokenizer()
        tok._encoding = None  # force fallback mode
        return tok

    def test_count_tokens_fallback_returns_int(self):
        tok = self._make_fallback_tokenizer()
        count = tok.count_tokens("Hello World")
        assert count >= 1

    def test_truncate_fallback(self):
        tok = self._make_fallback_tokenizer()
        long_text = "a" * 100
        result = tok.truncate(long_text, max_tokens=5)
        # 5 tokens * 4 chars/token = 20 chars max
        assert len(result) <= 20

    def test_fits_in_context_fallback_true(self):
        tok = self._make_fallback_tokenizer()
        assert tok.fits_in_context("hi", max_tokens=1000) is True

    def test_fits_in_context_fallback_false(self):
        tok = self._make_fallback_tokenizer()
        long_text = "a" * 1000
        assert tok.fits_in_context(long_text, max_tokens=5) is False

    def test_split_by_token_limit_fallback(self):
        tok = self._make_fallback_tokenizer()
        text = "a" * 100
        parts = tok.split_by_token_limit(text, max_tokens=5)
        assert isinstance(parts, list)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 5 * 4  # max_tokens * _CHARS_PER_TOKEN


# ── ResponseParser ────────────────────────────────────────────────────────────

class TestResponseParserExtractJson:
    def test_extracts_fenced_json_object(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```'
        result = ResponseParser.extract_json(text)
        assert result == {"key": "value"}

    def test_extracts_fenced_json_without_language_hint(self):
        text = "Result:\n```\n[1, 2, 3]\n```"
        result = ResponseParser.extract_json(text)
        assert result == [1, 2, 3]

    def test_extracts_bare_json_object(self):
        text = 'Some text {"name": "Alice"} more text'
        result = ResponseParser.extract_json(text)
        assert result["name"] == "Alice"

    def test_raises_when_no_json(self):
        with pytest.raises(ValueError, match="No JSON"):
            ResponseParser.extract_json("No JSON here at all.")


class TestResponseParserTryExtractJson:
    def test_returns_parsed_response_on_success(self):
        text = '{"answer": 42}'
        result = ResponseParser.try_extract_json(text)
        assert isinstance(result, ParsedResponse)
        assert result.json_data == {"answer": 42}
        assert result.parse_error is None

    def test_returns_parsed_response_on_failure(self):
        result = ResponseParser.try_extract_json("no json here")
        assert result.json_data is None
        assert result.parse_error is not None


class TestResponseParserTextCleaning:
    def test_strip_thinking_tags(self):
        text = "Before <thinking>internal thoughts</thinking> After"
        result = ResponseParser.strip_thinking_tags(text)
        assert "internal thoughts" not in result
        assert "Before" in result
        assert "After" in result

    def test_strip_think_tags(self):
        text = "<think>reasoning</think>Answer: 42"
        result = ResponseParser.strip_thinking_tags(text)
        assert "reasoning" not in result
        assert "42" in result

    def test_strip_markdown_bold(self):
        text = "This is **bold** text."
        result = ResponseParser.strip_markdown_formatting(text)
        assert "**" not in result
        assert "bold" in result

    def test_strip_markdown_headers(self):
        text = "## Section Title\nContent here."
        result = ResponseParser.strip_markdown_formatting(text)
        assert "##" not in result
        assert "Section Title" in result

    def test_strip_markdown_code_block(self):
        text = "Some text\n```\ncode here\n```\nMore text"
        result = ResponseParser.strip_markdown_formatting(text)
        assert "```" not in result

    def test_strip_markdown_italic(self):
        text = "This is *italic* text."
        result = ResponseParser.strip_markdown_formatting(text)
        assert "*italic*" not in result

    def test_strip_markdown_inline_code(self):
        text = "Use `print()` function."
        result = ResponseParser.strip_markdown_formatting(text)
        assert "`" not in result

    def test_extract_answer_with_prefix(self):
        text = "Reasoning: because it is. Answer: Paris"
        result = ResponseParser.extract_answer(text, answer_prefix="Answer:")
        assert result == "Paris"

    def test_extract_answer_missing_prefix_returns_whole(self):
        text = "Just a plain response."
        result = ResponseParser.extract_answer(text, answer_prefix="Answer:")
        assert result == "Just a plain response."

    def test_normalize_whitespace(self):
        text = "too   many   spaces"
        result = ResponseParser.normalize_whitespace(text)
        assert result == "too many spaces"

    def test_clean_pipeline(self):
        text = "<think>reasoning</think>  some  answer  "
        result = ResponseParser.clean(text)
        assert "reasoning" not in result
        assert "some answer" in result

    def test_clean_no_strip_thinking(self):
        text = "  normal   text  "
        result = ResponseParser.clean(text, strip_thinking=False)
        assert result == "normal text"


# ── InferenceEngine ───────────────────────────────────────────────────────────

class TestInferenceEngine:
    def _make_engine(self, with_retriever: bool = True):
        from src.inference.inference_engine import InferenceEngine, InferenceRequest
        from src.rag.retriever import RetrievalConfig, Retriever
        from src.rag.vector_store import Document, SearchResult

        mock_llm = MagicMock()
        mock_llm.complete.return_value = LLMResponse(
            content="The answer is 42.", model="gpt-4o"
        )
        mock_llm.config = LLMConfig(model="gpt-4o")

        if with_retriever:
            mock_retriever = MagicMock()
            doc = Document(id="d1", text="Context chunk.", metadata={"source": "file.txt"})
            mock_result = MagicMock()
            mock_result.document = doc
            mock_retriever.retrieve.return_value = [mock_result]
            engine = InferenceEngine(llm=mock_llm, retriever=mock_retriever)
        else:
            engine = InferenceEngine(llm=mock_llm, retriever=None)

        return engine, mock_llm

    def test_run_with_rag_returns_inference_result(self):
        from src.inference.inference_engine import InferenceRequest

        engine, mock_llm = self._make_engine(with_retriever=True)
        request = InferenceRequest(query="What is the answer?")
        result = engine.run(request)

        assert result.answer == "The answer is 42."
        assert result.model == "gpt-4o"
        assert isinstance(result.sources, list)

    def test_run_without_retriever_skips_rag(self):
        from src.inference.inference_engine import InferenceRequest

        engine, mock_llm = self._make_engine(with_retriever=False)
        request = InferenceRequest(query="Direct question", use_rag=False)
        result = engine.run(request)

        assert result.answer == "The answer is 42."
        assert result.retrieved_chunks == []

    def test_run_use_rag_false_with_retriever_skips_retrieval(self):
        from src.inference.inference_engine import InferenceRequest

        engine, mock_llm = self._make_engine(with_retriever=True)
        request = InferenceRequest(query="test", use_rag=False)
        engine.run(request)
        engine._retriever.retrieve.assert_not_called()

    def test_run_with_system_prompt(self):
        from src.inference.inference_engine import InferenceRequest

        engine, mock_llm = self._make_engine(with_retriever=False)
        request = InferenceRequest(
            query="test", use_rag=False, system_prompt="You are an expert."
        )
        engine.run(request)
        call_messages = mock_llm.complete.call_args[0][0]
        roles = [m.role for m in call_messages]
        assert "system" in roles

    def test_run_extra_context_included_in_prompt(self):
        from src.inference.inference_engine import InferenceRequest

        engine, mock_llm = self._make_engine(with_retriever=False)
        request = InferenceRequest(
            query="test", use_rag=False, extra_context="Extra context here."
        )
        engine.run(request)
        call_messages = mock_llm.complete.call_args[0][0]
        # Extra context should appear somewhere in the user message
        user_msg = next(m for m in call_messages if m.role == "user")
        assert "Extra context here." in user_msg.content


# ── AnalysisResult helpers ────────────────────────────────────────────────────

class TestAnalysisResult:
    def _make_result(self):
        from src.core.document_analysis_pipeline import AnalysisResult, PassageAnalysis

        passage = PassageAnalysis(
            passage_index=0,
            passage="Some passage text.",
            questions=["What is this?"],
            qa_pairs=[{"question": "What is this?", "answer": "A test.", "score": 0.95}],
        )
        return AnalysisResult(
            source="doc.pdf",
            extracted_text="Full text here.",
            text_preview="Full text",
            summary="A brief summary.",
            num_passages=1,
            passages=[passage],
            all_qa_pairs=[{"question": "What?", "answer": "It.", "score": 0.9}],
        )

    def test_to_dict_contains_expected_keys(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["source"] == "doc.pdf"
        assert d["summary"] == "A brief summary."
        assert d["num_passages"] == 1
        assert len(d["passages"]) == 1

    def test_to_json_is_valid_json(self):
        result = self._make_result()
        text = result.to_json()
        parsed = json.loads(text)
        assert parsed["source"] == "doc.pdf"

    def test_to_json_indentation(self):
        result = self._make_result()
        text = result.to_json(indent=4)
        assert "\n    " in text  # 4-space indented


class TestDocumentAnalysisPipelinePrintResults:
    def test_print_results_outputs_to_stdout(self, capsys):
        from src.core.document_analysis_pipeline import AnalysisResult, DocumentAnalysisPipeline

        pipeline = DocumentAnalysisPipeline()
        result = AnalysisResult(
            source="test.pdf",
            extracted_text="Some text.",
            text_preview="Some",
            summary="Short summary.",
            num_passages=2,
            passages=[],
            all_qa_pairs=[
                {"question": "Q1?", "answer": "A1", "score": 0.8},
                {"question": "Q2?", "answer": "A2", "score": 0.0},
            ],
        )
        pipeline.print_results(result)
        captured = capsys.readouterr()
        assert "test.pdf" in captured.out
        assert "Short summary." in captured.out
        assert "Q1?" in captured.out
        assert "A1" in captured.out

    def test_print_results_shows_score_when_nonzero(self, capsys):
        from src.core.document_analysis_pipeline import AnalysisResult, DocumentAnalysisPipeline

        pipeline = DocumentAnalysisPipeline()
        result = AnalysisResult(
            source="doc.pdf",
            extracted_text="text",
            text_preview="text",
            summary="sum",
            num_passages=1,
            all_qa_pairs=[{"question": "Q?", "answer": "A", "score": 0.95}],
        )
        pipeline.print_results(result)
        captured = capsys.readouterr()
        assert "0.95" in captured.out


class TestDocumentAnalysisPipelineRun:
    """Tests the full pipeline.run() with mocked PDF extractor and mock LLM."""

    def _make_pipeline_with_mocks(self, tmp_path):
        from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

        config = AnalysisConfig(
            provider="llm",
            output_dir=str(tmp_path / "out"),
            cache_dir=str(tmp_path / "cache"),
            passage_word_limit=50,
        )
        mock_llm = MagicMock()
        mock_llm.complete.return_value = LLMResponse(
            content="Mocked LLM response.", model="gpt-4o"
        )
        pipeline = DocumentAnalysisPipeline(config=config, llm=mock_llm)
        return pipeline, mock_llm

    def test_pipeline_run_produces_analysis_result(self, tmp_path):
        from src.core.document_analysis_pipeline import AnalysisResult

        pipeline, _ = self._make_pipeline_with_mocks(tmp_path)

        # Pre-populate the cache so no real PDF is needed
        sample_text = (
            "Artificial intelligence is transforming industries. "
            "Machine learning enables systems to learn from data. "
            "Deep learning uses neural networks with many layers. "
            "Natural language processing allows computers to understand text. "
            "Computer vision helps machines interpret images. " * 5
        )
        fake_pdf = tmp_path / "sample.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")
        import hashlib
        sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()[:16]
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"sample_{sha}.txt").write_text(sample_text, encoding="utf-8")

        result = pipeline.run(fake_pdf)

        assert isinstance(result, AnalysisResult)
        assert result.source == str(fake_pdf)
        assert len(result.extracted_text) > 0
        assert result.num_passages >= 1

    def test_pipeline_save_and_load_results(self, tmp_path):
        import json as _json
        from src.core.document_analysis_pipeline import AnalysisResult

        pipeline, _ = self._make_pipeline_with_mocks(tmp_path)

        sample_text = "Sample content for testing. " * 20
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF")
        import hashlib
        sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()[:16]
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"doc_{sha}.txt").write_text(sample_text, encoding="utf-8")

        result = pipeline.run(fake_pdf)
        out_path = pipeline.save_results(result)

        assert out_path.exists()
        data = _json.loads(out_path.read_text())
        assert data["source"] == str(fake_pdf)
        assert "summary" in data
