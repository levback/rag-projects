"""Document summarisation — supports LLM API and HuggingFace backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.processing.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_HF_DEFAULT_MODEL = "t5-small"
_MAX_HF_INPUT_CHARS = 1000  # t5-small token limit is ~512; we stay conservative
_LLM_CHUNK_TOKENS = 3000   # max tokens per chunk when using LLM on long docs


@dataclass
class SummaryConfig:
    provider: str = "huggingface"   # "huggingface" | "llm"
    hf_model: str = _HF_DEFAULT_MODEL
    max_length: int = 150           # HF: max tokens in summary
    min_length: int = 30            # HF: min tokens in summary
    do_sample: bool = False
    llm_chunk_tokens: int = _LLM_CHUNK_TOKENS


class Summarizer:
    """Summarises text using either HuggingFace models or an LLM client.

    **HuggingFace mode** (default, no API key required):
      Uses ``t5-small`` to produce extractive-style summaries.
      Long documents are split into chunks and summarised per-chunk, then
      the chunk summaries are combined into a final summary.

    **LLM mode** (requires configured API client):
      Uses the :class:`~src.prompts.templates.DOCUMENT_SUMMARY` template
      with any :class:`~src.core.base_llm.BaseLLM` implementation.
    """

    def __init__(self, config: SummaryConfig | None = None, llm=None) -> None:
        self._cfg = config or SummaryConfig()
        self._llm = llm
        self._hf_pipeline = None
        self._tokenizer = Tokenizer()

    # ── Public API ────────────────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """Return a summary string for *text*."""
        if not text or not text.strip():
            return ""

        if self._cfg.provider == "huggingface":
            return self._hf_summarize(text)
        if self._cfg.provider == "llm":
            return self._llm_summarize(text)
        raise ValueError(f"Unknown summarizer provider: {self._cfg.provider!r}")

    def summarize_passages(self, passages: list[str]) -> str:
        """Summarise each passage individually, then combine into one summary."""
        if not passages:
            return ""

        chunk_summaries = [self.summarize(p) for p in passages]
        combined = " ".join(chunk_summaries)
        # Final pass to compress the combined summaries
        if len(combined.split()) > 100:
            return self.summarize(combined)
        return combined

    # ── HuggingFace backend ───────────────────────────────────────────────────

    def _get_hf_pipeline(self):
        if self._hf_pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for HuggingFace summarisation. "
                    "Install with: pip install transformers torch"
                ) from exc
            logger.info("Loading HF summarisation model: %s", self._cfg.hf_model)
            self._hf_pipeline = pipeline(
                "summarization",
                model=self._cfg.hf_model,
                truncation=True,
            )
        return self._hf_pipeline

    def _hf_summarize(self, text: str) -> str:
        pipe = self._get_hf_pipeline()
        # t5-small works best on ≤1 000 chars; truncate gracefully
        chunk = text[:_MAX_HF_INPUT_CHARS]
        result = pipe(
            chunk,
            max_length=self._cfg.max_length,
            min_length=self._cfg.min_length,
            do_sample=self._cfg.do_sample,
        )
        summary: str = result[0]["summary_text"]
        logger.debug("HF summary (%d chars → %d chars)", len(text), len(summary))
        return summary

    # ── LLM backend ───────────────────────────────────────────────────────────

    def _llm_summarize(self, text: str) -> str:
        if self._llm is None:
            raise ValueError(
                "An LLM client must be provided when provider='llm'. "
                "Pass llm=create_llm(...) to Summarizer."
            )
        from src.core.base_llm import Message
        from src.prompts.templates import DOCUMENT_SUMMARY, DOCUMENT_SYSTEM

        # For very long documents, chunk and summarise iteratively
        chunks = self._split_for_llm(text)
        if len(chunks) == 1:
            prompt = DOCUMENT_SUMMARY.format(document=chunks[0])
            messages = [
                Message(role="system", content=DOCUMENT_SYSTEM.format()),
                Message(role="user", content=prompt),
            ]
            response = self._llm.complete(messages)
            return response.content

        logger.info("Document too long — summarising %d chunks individually", len(chunks))
        chunk_summaries = []
        for chunk in chunks:
            prompt = DOCUMENT_SUMMARY.format(document=chunk)
            messages = [
                Message(role="system", content=DOCUMENT_SYSTEM.format()),
                Message(role="user", content=prompt),
            ]
            response = self._llm.complete(messages)
            chunk_summaries.append(response.content)

        combined = "\n\n".join(chunk_summaries)
        # Final consolidation pass
        final_prompt = (
            "The following are partial summaries of a longer document. "
            "Combine them into a single coherent summary:\n\n" + combined
        )
        messages = [
            Message(role="system", content=DOCUMENT_SYSTEM.format()),
            Message(role="user", content=final_prompt),
        ]
        return self._llm.complete(messages).content

    def _split_for_llm(self, text: str) -> list[str]:
        return self._tokenizer.split_by_token_limit(text, self._cfg.llm_chunk_tokens)
