"""Tests for LLM client streaming/async, LocalLLM, HF backends, chunking, pdf_extractor, chain."""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.base_llm import LLMConfig, LLMResponse, Message
from src.processing.chunking import ChunkingConfig, TextChunker


# ── GPTClient streaming / async ───────────────────────────────────────────────

class TestGPTClientStreamAndAsync:
    def _make_client(self):
        import src.core.gpt_client  # ensure module is in sys.modules before patching
        with patch("src.core.gpt_client.OpenAI") as mock_sync, \
             patch("src.core.gpt_client.AsyncOpenAI") as mock_async:
            from src.core.gpt_client import GPTClient
            client = GPTClient(LLMConfig(model="gpt-4o"), api_key="sk-test")
        return client, client._client, client._async_client

    def test_stream_yields_tokens(self):
        client, mock_sync, _ = self._make_client()

        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = " world"
        chunk3 = MagicMock()
        chunk3.choices[0].delta.content = None  # None is skipped

        ctx_mgr = MagicMock()
        ctx_mgr.__enter__ = MagicMock(return_value=iter([chunk1, chunk2, chunk3]))
        ctx_mgr.__exit__ = MagicMock(return_value=False)
        mock_sync.chat.completions.stream.return_value = ctx_mgr

        tokens = list(client.stream([Message(role="user", content="Hi")]))
        assert tokens == ["Hello", " world"]

    def test_acomplete_returns_response(self):
        client, _, mock_async = self._make_client()

        mock_choice = MagicMock()
        mock_choice.message.content = "async answer"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 3
        mock_resp.usage.total_tokens = 8

        mock_async.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = asyncio.run(client.acomplete([Message(role="user", content="async?")]))
        assert result.content == "async answer"

    def test_astream_yields_tokens(self):
        client, _, mock_async = self._make_client()

        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "tok1"
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = "tok2"

        async def _async_stream():
            for c in [chunk1, chunk2]:
                yield c

        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=_async_stream())
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_async.chat.completions.stream.return_value = ctx_mgr

        async def collect():
            tokens = []
            async for t in client.astream([Message(role="user", content="go")]):
                tokens.append(t)
            return tokens

        tokens = asyncio.run(collect())
        assert tokens == ["tok1", "tok2"]


# ── ClaudeClient streaming / async ────────────────────────────────────────────

class TestClaudeClientStreamAndAsync:
    def _make_client(self):
        import src.core.claude_client  # ensure module is in sys.modules before patching
        with patch("src.core.claude_client.anthropic.Anthropic") as mock_sync, \
             patch("src.core.claude_client.anthropic.AsyncAnthropic") as mock_async:
            from src.core.claude_client import ClaudeClient
            client = ClaudeClient(LLMConfig(model="claude-3-5-sonnet-20241022"), api_key="ant-test")
        return client, client._client, client._async_client

    def test_stream_yields_tokens(self):
        client, mock_sync, _ = self._make_client()

        ctx_mgr = MagicMock()
        ctx_mgr.__enter__ = MagicMock(return_value=ctx_mgr)
        ctx_mgr.__exit__ = MagicMock(return_value=False)
        ctx_mgr.text_stream = iter(["Hello", " there"])
        mock_sync.messages.stream.return_value = ctx_mgr

        tokens = list(client.stream([Message(role="user", content="Hi")]))
        assert tokens == ["Hello", " there"]

    def test_stream_with_system_message(self):
        client, mock_sync, _ = self._make_client()

        ctx_mgr = MagicMock()
        ctx_mgr.__enter__ = MagicMock(return_value=ctx_mgr)
        ctx_mgr.__exit__ = MagicMock(return_value=False)
        ctx_mgr.text_stream = iter(["ok"])
        mock_sync.messages.stream.return_value = ctx_mgr

        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi"),
        ]
        list(client.stream(messages))
        call_kwargs = mock_sync.messages.stream.call_args[1]
        assert call_kwargs.get("system") == "Be helpful."

    def test_acomplete_returns_response(self):
        client, _, mock_async = self._make_client()

        mock_content_block = MagicMock()
        mock_content_block.text = "Async Claude answer"
        mock_resp = MagicMock()
        mock_resp.content = [mock_content_block]
        mock_resp.model = "claude-3-5-sonnet-20241022"
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 4

        mock_async.messages.create = AsyncMock(return_value=mock_resp)

        result = asyncio.run(client.acomplete([Message(role="user", content="async?")]))
        assert result.content == "Async Claude answer"
        assert result.finish_reason == "end_turn"

    def test_acomplete_empty_content_returns_empty_string(self):
        client, _, mock_async = self._make_client()

        mock_resp = MagicMock()
        mock_resp.content = []
        mock_resp.model = "claude-3-5-sonnet"
        mock_resp.stop_reason = "stop"
        mock_resp.usage.input_tokens = 0
        mock_resp.usage.output_tokens = 0

        mock_async.messages.create = AsyncMock(return_value=mock_resp)

        result = asyncio.run(client.acomplete([Message(role="user", content="test")]))
        assert result.content == ""

    def test_astream_yields_tokens(self):
        client, _, mock_async = self._make_client()

        async def _text_stream():
            for t in ["a", "b", "c"]:
                yield t

        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=ctx_mgr)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        ctx_mgr.text_stream = _text_stream()
        mock_async.messages.stream.return_value = ctx_mgr

        async def collect():
            tokens = []
            async for t in client.astream([Message(role="user", content="go")]):
                tokens.append(t)
            return tokens

        tokens = asyncio.run(collect())
        assert tokens == ["a", "b", "c"]

    def test_complete_with_system_message(self):
        client, mock_sync, _ = self._make_client()
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_resp.model = "claude-3-5-sonnet"
        mock_resp.stop_reason = "stop"
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        mock_sync.messages.create.return_value = mock_resp

        messages = [
            Message(role="system", content="Be concise."),
            Message(role="user", content="Hi"),
        ]
        result = client.complete(messages)
        call_kwargs = mock_sync.messages.create.call_args[1]
        assert call_kwargs.get("system") == "Be concise."
        assert isinstance(result, LLMResponse)

    def test_acomplete_with_system_message(self):
        client, _, mock_async = self._make_client()
        mock_content = MagicMock()
        mock_content.text = "Async response"
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_resp.model = "claude-3-5-sonnet"
        mock_resp.stop_reason = "stop"
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        mock_async.messages.create = AsyncMock(return_value=mock_resp)

        messages = [
            Message(role="system", content="Be an expert."),
            Message(role="user", content="Explain AI"),
        ]
        result = asyncio.run(client.acomplete(messages))
        call_kwargs = mock_async.messages.create.call_args[1]
        assert call_kwargs.get("system") == "Be an expert."
        assert isinstance(result, LLMResponse)

    def test_astream_with_system_message(self):
        client, _, mock_async = self._make_client()

        async def _text_stream():
            for t in ["yes"]:
                yield t

        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=ctx_mgr)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        ctx_mgr.text_stream = _text_stream()
        mock_async.messages.stream.return_value = ctx_mgr

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Test"),
        ]

        async def collect():
            tokens = []
            async for t in client.astream(messages):
                tokens.append(t)
            return tokens

        tokens = asyncio.run(collect())
        call_kwargs = mock_async.messages.stream.call_args[1]
        assert call_kwargs.get("system") == "You are helpful."


# ── LocalLLM ──────────────────────────────────────────────────────────────────

class TestLocalLLM:
    def _make_client(self, mock_pipeline=None):
        from src.core.local_llm import LocalLLM

        if mock_pipeline is None:
            mock_pipeline = MagicMock()
            mock_pipeline.return_value = [{"generated_text": "Generated text."}]

        client = LocalLLM(LLMConfig(model="llama-3.1-8b-instruct"))
        client._pipeline = mock_pipeline
        return client, mock_pipeline

    def test_build_prompt_formats_correctly(self):
        from src.core.local_llm import _build_prompt

        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="What is AI?"),
        ]
        prompt = _build_prompt(messages)
        assert "Be helpful." in prompt
        assert "What is AI?" in prompt

    def test_build_prompt_no_system(self):
        from src.core.local_llm import _build_prompt

        messages = [Message(role="user", content="Hello")]
        prompt = _build_prompt(messages)
        assert "Hello" in prompt

    def test_complete_returns_llm_response(self):
        client, mock_pipe = self._make_client()
        mock_pipe.return_value = [{"generated_text": "AI is a field of computer science."}]

        result = client.complete([Message(role="user", content="What is AI?")])
        assert isinstance(result, LLMResponse)
        assert result.content == "AI is a field of computer science."

    def test_complete_missing_transformers_raises(self):
        from src.core.local_llm import LocalLLM

        client = LocalLLM(LLMConfig(model="test-model"))
        # _pipeline is None, so it will try to import transformers
        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=_block):
            with pytest.raises(ImportError, match="transformers is required"):
                client.complete([Message(role="user", content="hi")])

    def test_stream_yields_tokens(self):
        from src.core.local_llm import LocalLLM

        mock_streamer_tokens = ["Hello", " world", ""]

        class FakeStreamer:
            def __init__(self, tokenizer, **kwargs):
                self._iter = iter(mock_streamer_tokens)

            def __iter__(self):
                return self._iter

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = MagicMock()
        mock_pipe.return_value = [{"generated_text": ""}]

        client = LocalLLM(LLMConfig(model="test"))
        client._pipeline = mock_pipe

        import sys as _sys
        _mock_tx = MagicMock()
        _mock_tx.TextIteratorStreamer = FakeStreamer
        with patch.dict(_sys.modules, {"transformers": _mock_tx}), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            mock_thread.return_value.join = MagicMock()
            tokens = list(client.stream([Message(role="user", content="hi")]))

        assert "Hello" in tokens
        assert " world" in tokens

    def test_acomplete_async(self):
        client, mock_pipe = self._make_client()
        mock_pipe.return_value = [{"generated_text": "async response"}]

        result = asyncio.run(client.acomplete([Message(role="user", content="test")]))
        assert result.content == "async response"

    def test_astream_yields_tokens(self):
        from src.core.local_llm import LocalLLM

        client = LocalLLM(LLMConfig(model="test"))
        client._pipeline = MagicMock()

        # Mock sync stream to avoid transformers/threading complexity
        with patch.object(client, "stream", return_value=iter(["x", "y"])):
            async def collect():
                tokens = []
                async for t in client.astream([Message(role="user", content="go")]):
                    tokens.append(t)
                return tokens

            tokens = asyncio.run(collect())
        assert tokens == ["x", "y"]

    def test_get_pipeline_lazy_init(self):
        """Covers local_llm lines 43-45: lazy pipeline creation on first call."""
        import sys as _sys
        from src.core.local_llm import LocalLLM

        mock_tx = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_tx.pipeline.return_value = mock_pipeline_instance

        with patch.dict(_sys.modules, {"transformers": mock_tx}):
            client = LocalLLM(LLMConfig(model="test-model"))
            # _pipeline is None — calling _get_pipeline() should create it
            pipe = client._get_pipeline()

        assert pipe is mock_pipeline_instance
        mock_tx.pipeline.assert_called_once()

    def test_stream_missing_text_iterator_streamer_raises(self):
        """Covers local_llm lines 73-74: ImportError in stream()."""
        import builtins
        from src.core.local_llm import LocalLLM

        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *a, **kw)

        client = LocalLLM(LLMConfig(model="test-model"))
        client._pipeline = MagicMock()  # pre-set pipeline so _get_pipeline doesn't import

        with patch("builtins.__import__", side_effect=_block):
            with pytest.raises(ImportError, match="transformers is required for streaming"):
                list(client.stream([Message(role="user", content="hi")]))


# ── TextChunker (character-based splits) ─────────────────────────────────────

class TestTextChunkerCharacterSplit:
    def test_no_sentence_split_uses_character_split(self):
        cfg = ChunkingConfig(chunk_size=20, chunk_overlap=0, sentence_split=False)
        chunker = TextChunker(config=cfg)
        text = "a" * 100
        chunks = chunker.split(text)
        for c in chunks:
            assert len(c) <= 20

    def test_character_split_overlap(self):
        cfg = ChunkingConfig(chunk_size=10, chunk_overlap=3, sentence_split=False)
        chunker = TextChunker(config=cfg)
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunker.split(text)
        assert len(chunks) >= 2
        # With overlap, second chunk should start earlier than chunk_size
        if len(chunks) >= 2:
            # step = chunk_size - overlap = 7
            assert chunks[1][0] == text[7]

    def test_empty_text_returns_empty(self):
        chunker = TextChunker()
        assert chunker.split("") == []
        assert chunker.split("   ") == []

    def test_short_text_fits_in_one_chunk(self):
        chunker = TextChunker(ChunkingConfig(chunk_size=1000))
        text = "Short sentence here."
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_single_sentence_longer_than_chunk_size(self):
        """A single sentence longer than chunk_size should still produce a chunk."""
        long_sentence = "word " * 200  # ~1000 chars
        cfg = ChunkingConfig(chunk_size=50, chunk_overlap=5, sentence_split=True)
        chunker = TextChunker(config=cfg)
        chunks = chunker.split(long_sentence)
        assert len(chunks) >= 1

    def test_tail_helper(self):
        assert TextChunker._tail("abcde", 3) == "cde"
        assert TextChunker._tail("ab", 5) == ""   # n > len → empty
        assert TextChunker._tail("abcde", 0) == ""

    def test_overlap_zero_no_repetition(self):
        cfg = ChunkingConfig(chunk_size=10, chunk_overlap=0, sentence_split=False)
        chunker = TextChunker(config=cfg)
        text = "0123456789abcdefghij"  # exactly 20 chars
        chunks = chunker.split(text)
        assert "".join(chunks) == text  # no overlap means exact reconstruction

    def test_flush_current_when_sentence_exceeds_chunk_size(self):
        """Covers chunking.py line 52: flush current when new sentence would exceed."""
        # First sentence fits; second sentence's candidate overflows → flush first
        cfg = ChunkingConfig(chunk_size=15, chunk_overlap=0, sentence_split=True)
        chunker = TextChunker(config=cfg)
        text = "Hi. " + "X" * 50 + "."
        chunks = chunker.split(text)
        assert len(chunks) >= 2
        assert "Hi." in chunks[0]

    def test_character_split_zero_step_uses_chunk_size(self):
        """Covers chunking.py line 72: step falls back to chunk_size when overlap >= size."""
        cfg = ChunkingConfig(chunk_size=10, chunk_overlap=15, sentence_split=False)
        chunker = TextChunker(config=cfg)
        text = "abcdefghijklmnopqrst"  # 20 chars
        chunks = chunker.split(text)
        # step = chunk_size (10) because overlap > size
        assert all(len(c) <= 10 for c in chunks)


# ── PDFExtractor – pdfplumber path (mocked) ───────────────────────────────────

class TestPDFExtractorPdfplumber:
    def test_extract_calls_pdfplumber(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor
        import sys as _sys

        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 text."
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict(_sys.modules, {"pdfplumber": mock_pdfplumber}):
            extractor = PDFExtractor(cache_dir=str(tmp_path))
            result = extractor.extract(fake_pdf)

        assert "Page 1 text." in result

    def test_extract_pages_returns_list(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor
        import sys as _sys

        fake_pdf = tmp_path / "multi.pdf"
        fake_pdf.write_bytes(b"%PDF fake")

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2."
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict(_sys.modules, {"pdfplumber": mock_pdfplumber}):
            extractor = PDFExtractor(cache_dir=None)
            pages = extractor.extract_pages(fake_pdf)

        assert pages == ["Page 1.", "Page 2."]

    def test_extract_pages_not_found_raises(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(cache_dir=None)
        with pytest.raises(FileNotFoundError):
            extractor.extract_pages("/nonexistent/file.pdf")

    def test_none_page_text_skipped(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor
        import sys as _sys

        fake_pdf = tmp_path / "mixed.pdf"
        fake_pdf.write_bytes(b"%PDF fake")

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = None  # no text on this page
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Real text."
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict(_sys.modules, {"pdfplumber": mock_pdfplumber}):
            extractor = PDFExtractor(cache_dir=None)
            result = extractor.extract(fake_pdf)

        assert "Real text." in result

    def test_missing_pdfplumber_raises(self, tmp_path):
        from src.processing.pdf_extractor import PDFExtractor

        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF fake")

        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "pdfplumber":
                raise ImportError("no pdfplumber")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=_block):
            extractor = PDFExtractor(cache_dir=None)
            with pytest.raises(ImportError, match="pdfplumber is required"):
                extractor.extract(fake_pdf)


# ── HF Summarizer backend (mocked transformers) ───────────────────────────────

class TestSummarizerHFBackend:
    def _make_hf_summarizer(self):
        from src.inference.summarizer import Summarizer, SummaryConfig

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"summary_text": "A brief summary."}]

        s = Summarizer(config=SummaryConfig(provider="huggingface"))
        s._hf_pipeline = mock_pipeline
        return s, mock_pipeline

    def test_hf_summarize_returns_string(self):
        s, _ = self._make_hf_summarizer()
        result = s.summarize("This is a long document about artificial intelligence.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hf_summarize_unknown_provider_raises(self):
        from src.inference.summarizer import Summarizer, SummaryConfig

        s = Summarizer(config=SummaryConfig(provider="unknown"))
        with pytest.raises(ValueError, match="Unknown summarizer provider"):
            s.summarize("Some text.")

    def test_hf_missing_transformers_raises(self):
        from src.inference.summarizer import Summarizer, SummaryConfig

        s = Summarizer(config=SummaryConfig(provider="huggingface"))
        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=_block):
            with pytest.raises(ImportError, match="transformers is required"):
                s._get_hf_pipeline()

    def test_get_hf_pipeline_lazy_init(self):
        """Covers summarizer lines 81-82: pipeline created on first call."""
        import sys as _sys
        from src.inference.summarizer import Summarizer, SummaryConfig

        mock_tx = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.return_value = [{"summary_text": "ok"}]
        mock_tx.pipeline.return_value = mock_pipe_instance

        with patch.dict(_sys.modules, {"transformers": mock_tx}):
            s = Summarizer(config=SummaryConfig(provider="huggingface"))
            pipe = s._get_hf_pipeline()

        assert pipe is mock_pipe_instance
        mock_tx.pipeline.assert_called_once_with("summarization", model=s._cfg.hf_model, truncation=True)


# ── HF QuestionGenerator backend (mocked) ────────────────────────────────────

class TestQuestionGeneratorHFBackend:
    def _make_hf_qg(self, output="What is AI?<sep>Why is AI important?"):
        from src.inference.question_generator import QuestionGenerator

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": output}]

        qg = QuestionGenerator(provider="huggingface")
        qg._hf_pipeline = mock_pipeline
        return qg, mock_pipeline

    def test_hf_generate_returns_list(self):
        qg, _ = self._make_hf_qg()
        questions = qg.generate("AI is transforming industries.")
        assert isinstance(questions, list)
        assert len(questions) >= 1

    def test_hf_generates_multiple_questions_from_sep(self):
        qg, _ = self._make_hf_qg("Q1?<sep>Q2?<sep>Q3?")
        questions = qg.generate("Some passage text here for question generation.")
        assert len(questions) == 3

    def test_hf_unknown_provider_raises(self):
        from src.inference.question_generator import QuestionGenerator

        qg = QuestionGenerator(provider="unknown")
        with pytest.raises(ValueError, match="Unknown provider"):
            qg.generate("Some text.")

    def test_hf_missing_transformers_raises(self):
        from src.inference.question_generator import QuestionGenerator

        qg = QuestionGenerator(provider="huggingface")
        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=_block):
            with pytest.raises(ImportError, match="transformers is required"):
                qg._get_hf_pipeline()

    def test_get_hf_pipeline_lazy_init(self):
        """Covers question_generator lines 82-83: pipeline created on first call."""
        import sys as _sys
        from src.inference.question_generator import QuestionGenerator

        mock_tx = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_tx.pipeline.return_value = mock_pipe_instance

        with patch.dict(_sys.modules, {"transformers": mock_tx}):
            qg = QuestionGenerator(provider="huggingface")
            pipe = qg._get_hf_pipeline()

        assert pipe is mock_pipe_instance
        mock_tx.pipeline.assert_called_once()


# ── HF QAEngine backend (mocked) ─────────────────────────────────────────────

class TestQAEngineHFBackend:
    def _make_hf_qa(self):
        from src.inference.qa_engine import QAEngine

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {"answer": "Extractive answer.", "score": 0.85}

        qa = QAEngine(provider="huggingface")
        qa._hf_pipeline = mock_pipeline
        return qa, mock_pipeline

    def test_hf_answer_returns_qa_result(self):
        qa, _ = self._make_hf_qa()
        from src.inference.qa_engine import QAResult

        result = qa.answer("What is AI?", "AI is a field of computer science.")
        assert isinstance(result, QAResult)
        assert result.answer == "Extractive answer."
        assert result.score == pytest.approx(0.85)

    def test_hf_answer_passes_question_context(self):
        qa, mock_pipe = self._make_hf_qa()
        qa.answer("Question?", "Context text.")
        call_args = mock_pipe.call_args[0][0]
        assert call_args["question"] == "Question?"
        assert call_args["context"] == "Context text."

    def test_hf_unknown_provider_raises(self):
        from src.inference.qa_engine import QAEngine

        qa = QAEngine(provider="unknown")
        with pytest.raises(ValueError, match="Unknown provider"):
            qa.answer("Q?", "context")

    def test_hf_missing_transformers_raises(self):
        from src.inference.qa_engine import QAEngine

        qa = QAEngine(provider="huggingface")
        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=_block):
            with pytest.raises(ImportError, match="transformers is required"):
                qa._get_hf_pipeline()

    def test_answer_passages_multi_passage(self):
        qa, _ = self._make_hf_qa()
        passage_questions = [
            {
                "passage_index": 0,
                "passage": "AI is powerful.",
                "questions": ["What is powerful?"],
            },
            {
                "passage_index": 1,
                "passage": "ML is a subset.",
                "questions": ["What is a subset?"],
            },
        ]
        results = qa.answer_passages(passage_questions)
        assert len(results) == 2
        assert results[0].passage_index == 0
        assert results[1].passage_index == 1

    def test_answer_passages_deduplicates(self):
        qa, _ = self._make_hf_qa()
        passage_questions = [
            {
                "passage_index": 0,
                "passage": "Context A.",
                "questions": ["Same question?", "Same question?"],  # duplicate
            },
        ]
        results = qa.answer_passages(passage_questions, deduplicate=True)
        assert len(results) == 1


# ── InferenceEngine streaming and async ──────────────────────────────────────

class TestInferenceEngineStreamAndAsync:
    def _make_engine(self):
        from src.inference.inference_engine import InferenceEngine

        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["tok1", "tok2"])
        mock_llm.acomplete = AsyncMock(
            return_value=LLMResponse(content="async answer", model="gpt-4o")
        )
        mock_llm.config = LLMConfig(model="gpt-4o")

        engine = InferenceEngine(llm=mock_llm, retriever=None)
        return engine, mock_llm

    def test_stream_yields_tokens(self):
        from src.inference.inference_engine import InferenceRequest

        engine, _ = self._make_engine()
        request = InferenceRequest(query="What is AI?", use_rag=False)
        tokens = list(engine.stream(request))
        assert tokens == ["tok1", "tok2"]

    def test_arun_returns_result(self):
        from src.inference.inference_engine import InferenceRequest

        engine, _ = self._make_engine()
        request = InferenceRequest(query="test", use_rag=False)
        result = asyncio.run(engine.arun(request))
        assert result.answer == "async answer"

    def test_arun_with_rag(self):
        from src.inference.inference_engine import InferenceEngine, InferenceRequest
        from src.rag.vector_store import Document, SearchResult

        mock_llm = MagicMock()
        mock_llm.acomplete = AsyncMock(
            return_value=LLMResponse(content="rag async answer", model="gpt-4o")
        )
        mock_llm.config = LLMConfig(model="gpt-4o")

        mock_retriever = MagicMock()
        doc = Document(id="d1", text="RAG context.", metadata={"source": "file.txt"})
        mock_result = MagicMock()
        mock_result.document = doc
        mock_retriever.aretrieve = AsyncMock(return_value=[mock_result])

        engine = InferenceEngine(llm=mock_llm, retriever=mock_retriever)
        request = InferenceRequest(query="test", use_rag=True)
        result = asyncio.run(engine.arun(request))

        assert result.answer == "rag async answer"
        assert result.retrieved_chunks == ["RAG context."]


# ── PromptChain async path ────────────────────────────────────────────────────

class TestPromptChainAsync:
    def test_async_run_chain(self):
        from src.prompts.chain import ChainStep, PromptChain
        from src.prompts.templates import PromptTemplate

        template = PromptTemplate(
            name="test",
            template="$input",
            input_variables=["input"],
        )
        mock_llm = MagicMock()
        mock_llm.acomplete = AsyncMock(
            return_value=LLMResponse(content="async chain result", model="gpt-4o")
        )

        step = ChainStep(name="test_step", template=template)
        chain = PromptChain(llm=mock_llm, steps=[step])

        async def run_chain():
            ctx = await chain.arun({"input": "test input"})
            return ctx.get("test_step")

        result = asyncio.run(run_chain())
        assert result == "async chain result"


# ── Retriever extended ────────────────────────────────────────────────────────

class TestRetrieverExtended:
    def _make_retriever(self, results=None):
        from src.rag.retriever import Retriever, RetrievalConfig
        from src.rag.vector_store import Document, SearchResult

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]

        mock_store = MagicMock()
        if results is None:
            doc = Document(id="d1", text="Result text.", metadata={"source": "f.txt"})
            results = [SearchResult(document=doc, score=0.8)]
        mock_store.search.return_value = results

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_store)
        return retriever

    def test_retrieve_texts_returns_strings(self):
        retriever = self._make_retriever()
        texts = retriever.retrieve_texts("query")
        assert texts == ["Result text."]

    def test_similarity_threshold_filters(self):
        from src.rag.retriever import Retriever, RetrievalConfig
        from src.rag.vector_store import Document, SearchResult

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1]
        mock_store = MagicMock()

        doc_high = Document(id="d1", text="high score", metadata={})
        doc_low = Document(id="d2", text="low score", metadata={})
        mock_store.search.return_value = [
            SearchResult(document=doc_high, score=0.9),
            SearchResult(document=doc_low, score=0.3),
        ]

        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=mock_store,
            config=RetrievalConfig(similarity_threshold=0.5),
        )
        results = retriever.retrieve("query")
        assert len(results) == 1
        assert results[0].document.text == "high score"

    def test_aretrieve_async(self):
        retriever = self._make_retriever()
        results = asyncio.run(retriever.aretrieve("async query"))
        assert len(results) == 1
