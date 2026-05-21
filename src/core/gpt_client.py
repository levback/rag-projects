"""OpenAI GPT client implementation."""
from __future__ import annotations

import logging
from typing import AsyncIterator, Iterator

from openai import AsyncOpenAI, OpenAI

from src.core.base_llm import BaseLLM, LLMConfig, LLMResponse, Message

logger = logging.getLogger(__name__)


def _to_openai_messages(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class GPTClient(BaseLLM):
    """Wraps the OpenAI Chat Completions API."""

    def __init__(self, config: LLMConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        # api_key falls back to OPENAI_API_KEY env var when None
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)

    # ── Sync ─────────────────────────────────────────────────────────────────

    def complete(self, messages: list[Message]) -> LLMResponse:
        self._logger.debug("Sending %d messages to %s", len(messages), self.config.model)
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=_to_openai_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            **self.config.extra,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            raw=response,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self._logger.debug("Streaming from %s", self.config.model)
        with self._client.chat.completions.stream(
            model=self.config.model,
            messages=_to_openai_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    # ── Async ─────────────────────────────────────────────────────────────────

    async def acomplete(self, messages: list[Message]) -> LLMResponse:
        self._logger.debug("Async sending %d messages to %s", len(messages), self.config.model)
        response = await self._async_client.chat.completions.create(
            model=self.config.model,
            messages=_to_openai_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            **self.config.extra,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            raw=response,
        )

    async def astream(self, messages: list[Message]) -> AsyncIterator[str]:
        self._logger.debug("Async streaming from %s", self.config.model)
        async with self._async_client.chat.completions.stream(
            model=self.config.model,
            messages=_to_openai_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
