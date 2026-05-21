"""Local model client using HuggingFace Transformers / llama.cpp."""
from __future__ import annotations

import logging
from typing import AsyncIterator, Iterator

from src.core.base_llm import BaseLLM, LLMConfig, LLMResponse, Message

logger = logging.getLogger(__name__)

_CHAT_TEMPLATE = "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def _build_prompt(messages: list[Message]) -> str:
    system = next((m.content for m in messages if m.role == "system"), "")
    user_parts = [m.content for m in messages if m.role == "user"]
    return _CHAT_TEMPLATE.format(system=system, user="\n".join(user_parts))


class LocalLLM(BaseLLM):
    """Runs a local HuggingFace model via the transformers pipeline.

    Requires: ``pip install transformers torch accelerate``
    """

    def __init__(self, config: LLMConfig, model_path: str | None = None) -> None:
        super().__init__(config)
        self._model_path = model_path or config.model
        self._pipeline = None  # lazy-loaded on first call

    # ── Pipeline setup ────────────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for LocalLLM. "
                    "Install it with: pip install transformers torch accelerate"
                ) from exc

            device_map = self.config.extra.get("device", "auto")
            self._logger.info("Loading local model from %s (device=%s)", self._model_path, device_map)
            self._pipeline = pipeline(
                "text-generation",
                model=self._model_path,
                device_map=device_map,
                trust_remote_code=False,
            )
        return self._pipeline

    # ── Sync ─────────────────────────────────────────────────────────────────

    def complete(self, messages: list[Message]) -> LLMResponse:
        pipe = self._get_pipeline()
        prompt = _build_prompt(messages)
        outputs = pipe(
            prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            return_full_text=False,
        )
        text: str = outputs[0]["generated_text"]
        return LLMResponse(content=text, model=self._model_path)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        # Transformers TextIteratorStreamer provides token-by-token output.
        try:
            from transformers import TextIteratorStreamer  # type: ignore[import]
            import threading
        except ImportError as exc:
            raise ImportError("transformers is required for streaming.") from exc

        pipe = self._get_pipeline()
        prompt = _build_prompt(messages)
        streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            text_inputs=prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            streamer=streamer,
        )
        thread = threading.Thread(target=pipe, kwargs=kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()

    # ── Async ─────────────────────────────────────────────────────────────────

    async def acomplete(self, messages: list[Message]) -> LLMResponse:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, messages)

    async def astream(self, messages: list[Message]) -> AsyncIterator[str]:
        # Wrap sync stream in async generator
        import asyncio

        loop = asyncio.get_event_loop()

        def _collect() -> list[str]:
            return list(self.stream(messages))

        tokens = await loop.run_in_executor(None, _collect)
        for token in tokens:
            yield token
