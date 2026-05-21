"""Abstract base class for autonomous agents."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result returned by every agent ``run()`` call."""

    answer: str
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.answer


class BaseAgent(ABC):
    """Abstract agent that accepts a query and returns an :class:`AgentResult`.

    All concrete agents must implement :meth:`run`.

    Args:
        llm: An optional :class:`~src.core.base_llm.BaseLLM` instance. If *None*,
             the agent uses its default local model.
        verbose: Log intermediate steps to DEBUG.
    """

    def __init__(
        self,
        llm: "BaseLLM | None" = None,  # noqa: F821
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._verbose = verbose

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Execute the agent for a given *query*.

        Args:
            query: The user's question or task.
            **kwargs: Agent-specific parameters.

        Returns:
            An :class:`AgentResult` with the answer and provenance.
        """
        ...

    def _log_step(self, step: str) -> None:
        if self._verbose:
            logger.debug("[%s] %s", type(self).__name__, step)
