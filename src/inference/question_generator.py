"""Question generation from text passages."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_HF_QG_MODEL = "valhalla/t5-base-qg-hl"


class QuestionGenerator:
    """Generates comprehension questions from a text passage.

    **HuggingFace mode** (default):
      Uses ``valhalla/t5-base-qg-hl``, a T5 model fine-tuned for question
      generation.  Questions are separated by ``<sep>`` tokens in the output.

    **LLM mode**:
      Uses the :class:`~src.prompts.templates.QUESTION_GENERATION` template
      with any :class:`~src.core.base_llm.BaseLLM` implementation.
    """

    def __init__(
        self,
        provider: str = "huggingface",
        hf_model: str = _HF_QG_MODEL,
        llm=None,
        min_questions: int = 3,
    ) -> None:
        self._provider = provider
        self._hf_model = hf_model
        self._llm = llm
        self._min_questions = min_questions
        self._hf_pipeline = None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, passage: str, min_questions: int | None = None) -> list[str]:
        """Return a list of questions generated from *passage*.

        If the model returns fewer than *min_questions*, the passage is split
        in half and generation is retried on each half to top up.
        """
        min_q = min_questions if min_questions is not None else self._min_questions
        if not passage or not passage.strip():
            return []

        if self._provider == "huggingface":
            return self._hf_generate(passage, min_q)
        if self._provider == "llm":
            return self._llm_generate(passage, min_q)
        raise ValueError(f"Unknown provider: {self._provider!r}")

    def generate_all(
        self, passages: list[str], min_questions: int | None = None
    ) -> list[dict]:
        """Generate questions for every passage, returning structured results."""
        results = []
        for i, passage in enumerate(passages):
            questions = self.generate(passage, min_questions=min_questions)
            results.append(
                {
                    "passage_index": i,
                    "passage": passage,
                    "questions": questions,
                }
            )
        return results

    # ── HuggingFace backend ───────────────────────────────────────────────────

    def _get_hf_pipeline(self):
        if self._hf_pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for HuggingFace QG. "
                    "Install with: pip install transformers torch"
                ) from exc
            logger.info("Loading HF question-generation model: %s", self._hf_model)
            self._hf_pipeline = pipeline("text2text-generation", model=self._hf_model)
        return self._hf_pipeline

    def _hf_generate(self, passage: str, min_q: int) -> list[str]:
        pipe = self._get_hf_pipeline()
        input_text = f"generate questions: {passage}"
        results = pipe(input_text)
        raw: str = results[0]["generated_text"]
        questions = [q.strip() for q in raw.split("<sep>") if q.strip()]

        # Top up if we got fewer than min_q
        if len(questions) < min_q:
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            mid = len(sentences) // 2
            for half in [sentences[:mid], sentences[mid:]]:
                sub_passage = " ".join(half)
                if not sub_passage.strip():
                    continue
                sub_input = f"generate questions: {sub_passage}"
                sub_results = pipe(sub_input)
                sub_raw: str = sub_results[0]["generated_text"]
                extra = [q.strip() for q in sub_raw.split("<sep>") if q.strip()]
                for q in extra:
                    if q not in questions:
                        questions.append(q)
                if len(questions) >= min_q:
                    break

        logger.debug("Generated %d questions from passage (%d chars)", len(questions), len(passage))
        return questions[:max(min_q, len(questions))]

    # ── LLM backend ───────────────────────────────────────────────────────────

    def _llm_generate(self, passage: str, min_q: int) -> list[str]:
        if self._llm is None:
            raise ValueError("An LLM client must be provided when provider='llm'.")

        from src.core.base_llm import Message
        from src.prompts.templates import DOCUMENT_SYSTEM, QUESTION_GENERATION

        prompt = QUESTION_GENERATION.format(passage=passage, num_questions=str(min_q))
        messages = [
            Message(role="system", content=DOCUMENT_SYSTEM.format()),
            Message(role="user", content=prompt),
        ]
        response = self._llm.complete(messages)
        raw = response.content.strip()
        questions = [
            line.strip().lstrip("-•*").strip()
            for line in raw.splitlines()
            if line.strip() and "?" in line
        ]
        logger.debug("LLM generated %d questions", len(questions))
        return questions
