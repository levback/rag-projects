"""Common interface shared by all LLM backend implementations."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Unified response envelope returned by every LLM client."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw: Any = field(default=None, repr=False)


@dataclass
class LLMConfig:
    """Runtime configuration passed to an LLM instance."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class every LLM client must implement."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)

    # ── Synchronous interface ─────────────────────────────────────────────────

    @abstractmethod
    def complete(self, messages: list[Message]) -> LLMResponse:
        """Send *messages* to the model and return a complete response."""

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield response tokens one-by-one (streaming mode)."""

    # ── Async interface ───────────────────────────────────────────────────────

    @abstractmethod
    async def acomplete(self, messages: list[Message]) -> LLMResponse:
        """Async variant of :meth:`complete`."""

    @abstractmethod
    async def astream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Async variant of :meth:`stream`."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def chat(self, user_message: str, system_prompt: str | None = None) -> str:
        """Convenience wrapper: build messages and return the response text."""
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))
        response = self.complete(messages)
        return response.content

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(model={self.config.model!r})"
