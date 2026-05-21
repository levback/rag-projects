"""Multi-step prompt chaining utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.base_llm import BaseLLM, Message
from src.prompts.templates import PromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """A single step in a prompt chain."""

    name: str
    template: PromptTemplate
    # Optional post-processor applied to the LLM output before passing to the next step
    postprocess: Callable[[str], Any] | None = None
    # If True, the raw LLM output is stored in ``context`` as ``step_name``
    capture_output: bool = True


@dataclass
class ChainContext:
    """Mutable context bag shared across all steps in a chain run."""

    variables: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get(self, key: str) -> Any:
        return self.variables[key]

    def update(self, data: dict[str, Any]) -> None:
        self.variables.update(data)


class PromptChain:
    """Executes a sequence of :class:`ChainStep` objects against an LLM.

    Each step renders its template with the current context, calls the LLM,
    optionally post-processes the output, and stores the result back into
    the context under the step name.

    Example::

        chain = PromptChain(llm=my_llm, steps=[step1, step2])
        result = chain.run({"document": raw_text})
    """

    def __init__(self, llm: BaseLLM, steps: list[ChainStep], system_prompt: str = "") -> None:
        self._llm = llm
        self._steps = steps
        self._system_prompt = system_prompt

    def run(self, initial_vars: dict[str, Any] | None = None) -> ChainContext:
        """Run all steps sequentially, returning the final :class:`ChainContext`.

        Args:
            initial_vars: Seed values available to the first (and subsequent) steps.

        Returns:
            :class:`ChainContext` populated with outputs from every step.
        """
        ctx = ChainContext(variables=dict(initial_vars or {}))

        for step in self._steps:
            logger.debug("Running chain step: %s", step.name)

            prompt_text = step.template.format(**ctx.variables)
            messages: list[Message] = []
            if self._system_prompt:
                messages.append(Message(role="system", content=self._system_prompt))
            messages.append(Message(role="user", content=prompt_text))

            response = self._llm.complete(messages)
            output: Any = response.content

            if step.postprocess is not None:
                output = step.postprocess(output)

            if step.capture_output:
                ctx.set(step.name, output)

            logger.debug("Step %s completed. Output preview: %.80s", step.name, str(output))

        return ctx

    async def arun(self, initial_vars: dict[str, Any] | None = None) -> ChainContext:
        """Async version of :meth:`run`."""
        ctx = ChainContext(variables=dict(initial_vars or {}))

        for step in self._steps:
            logger.debug("Async running chain step: %s", step.name)

            prompt_text = step.template.format(**ctx.variables)
            messages: list[Message] = []
            if self._system_prompt:
                messages.append(Message(role="system", content=self._system_prompt))
            messages.append(Message(role="user", content=prompt_text))

            response = await self._llm.acomplete(messages)
            output: Any = response.content

            if step.postprocess is not None:
                output = step.postprocess(output)

            if step.capture_output:
                ctx.set(step.name, output)

        return ctx
