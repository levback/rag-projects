"""Anthropic Claude client implementation."""
from __future__ import annotations

import logging
from typing import AsyncIterator, Iterator

import anthropic

from src.core.base_llm import BaseLLM, LLMConfig, LLMResponse, Message

logger = logging.getLogger(__name__)

_SYSTEM_ROLE = "system"


def _split_system(messages: list[Message]) -> tuple[str, list[dict]]:
    """Separate the optional system prompt from the conversation turns."""
    system = ""
    turns: list[dict] = []
    for m in messages:
        if m.role == _SYSTEM_ROLE:
            system = m.content
        else:
            turns.append({"role": m.role, "content": m.content})
    return system, turns


class ClaudeClient(BaseLLM):
    """Wraps the Anthropic Messages API."""

    def __init__(self, config: LLMConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        # api_key falls back to ANTHROPIC_API_KEY env var when None
        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Sync ─────────────────────────────────────────────────────────────────

    def complete(self, messages: list[Message]) -> LLMResponse:
        self._logger.debug("Sending %d messages to %s", len(messages), self.config.model)
        system, turns = _split_system(messages)
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=turns,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "stop",
            raw=response,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self._logger.debug("Streaming from %s", self.config.model)
        system, turns = _split_system(messages)
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=turns,
        )
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    # ── Async ─────────────────────────────────────────────────────────────────

    async def acomplete(self, messages: list[Message]) -> LLMResponse:
        self._logger.debug("Async sending %d messages to %s", len(messages), self.config.model)
        system, turns = _split_system(messages)
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=turns,
        )
        if system:
            kwargs["system"] = system

        response = await self._async_client.messages.create(**kwargs)
        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "stop",
            raw=response,
        )

    async def astream(self, messages: list[Message]) -> AsyncIterator[str]:
        self._logger.debug("Async streaming from %s", self.config.model)
        system, turns = _split_system(messages)
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=turns,
        )
        if system:
            kwargs["system"] = system

        async with self._async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
