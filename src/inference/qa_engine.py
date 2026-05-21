"""Question-answering engine — supports HuggingFace extractive QA and LLM generative QA."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_HF_QA_MODEL = "deepset/roberta-base-squad2"


@dataclass
class QAResult:
    """Structured answer to a single question."""

    question: str
    answer: str
    score: float = 0.0
    passage_index: int = 0


class QAEngine:
    """Answers questions against a text passage (context).

    **HuggingFace mode** (default):
      Uses ``deepset/roberta-base-squad2``, a RoBERTa model fine-tuned on
      SQuAD 2.0 for extractive question answering.

    **LLM mode**:
      Uses the :class:`~src.prompts.templates.DOCUMENT_QA` template with
      any :class:`~src.core.base_llm.BaseLLM` for generative QA.
    """

    def __init__(
        self,
        provider: str = "huggingface",
        hf_model: str = _HF_QA_MODEL,
        llm=None,
    ) -> None:
        self._provider = provider
        self._hf_model = hf_model
        self._llm = llm
        self._hf_pipeline = None

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, question: str, context: str) -> QAResult:
        """Return a :class:`QAResult` for *question* answered against *context*."""
        if not question.strip() or not context.strip():
            return QAResult(question=question, answer="", score=0.0)

        if self._provider == "huggingface":
            return self._hf_answer(question, context)
        if self._provider == "llm":
            return self._llm_answer(question, context)
        raise ValueError(f"Unknown provider: {self._provider!r}")

    def answer_batch(
        self,
        questions: list[str],
        context: str,
        deduplicate: bool = True,
    ) -> list[QAResult]:
        """Answer multiple *questions* against the same *context*.

        When *deduplicate* is ``True``, identical questions are skipped after
        the first occurrence (matching the article's ``answered_questions`` set).
        """
        seen: set[str] = set()
        results: list[QAResult] = []
        for question in questions:
            q_norm = question.strip().lower()
            if deduplicate and q_norm in seen:
                logger.debug("Skipping duplicate question: %.60s", question)
                continue
            seen.add(q_norm)
            results.append(self.answer(question, context))
        return results

    def answer_passages(
        self,
        passage_questions: list[dict],
        deduplicate: bool = True,
    ) -> list[QAResult]:
        """Answer all questions across all passage dicts from :meth:`~QuestionGenerator.generate_all`.

        *passage_questions* has the shape::

            [{"passage_index": int, "passage": str, "questions": list[str]}, ...]
        """
        seen: set[str] = set()
        all_results: list[QAResult] = []

        for item in passage_questions:
            passage = item["passage"]
            passage_idx = item["passage_index"]
            for question in item["questions"]:
                q_norm = question.strip().lower()
                if deduplicate and q_norm in seen:
                    continue
                seen.add(q_norm)
                result = self.answer(question, passage)
                result.passage_index = passage_idx
                all_results.append(result)

        return all_results

    # ── HuggingFace backend ───────────────────────────────────────────────────

    def _get_hf_pipeline(self):
        if self._hf_pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for HuggingFace QA. "
                    "Install with: pip install transformers torch"
                ) from exc
            logger.info("Loading HF QA model: %s", self._hf_model)
            self._hf_pipeline = pipeline(
                "question-answering",
                model=self._hf_model,
            )
        return self._hf_pipeline

    def _hf_answer(self, question: str, context: str) -> QAResult:
        pipe = self._get_hf_pipeline()
        result = pipe({"question": question, "context": context})
        return QAResult(
            question=question,
            answer=result.get("answer", ""),
            score=float(result.get("score", 0.0)),
        )

    # ── LLM backend ───────────────────────────────────────────────────────────

    def _llm_answer(self, question: str, context: str) -> QAResult:
        if self._llm is None:
            raise ValueError("An LLM client must be provided when provider='llm'.")

        from src.core.base_llm import Message
        from src.prompts.templates import DOCUMENT_QA, DOCUMENT_SYSTEM

        prompt = DOCUMENT_QA.format(passage=context, question=question)
        messages = [
            Message(role="system", content=DOCUMENT_SYSTEM.format()),
            Message(role="user", content=prompt),
        ]
        response = self._llm.complete(messages)
        return QAResult(
            question=question,
            answer=response.content.strip(),
            score=1.0,  # generative answers have no confidence score
        )
