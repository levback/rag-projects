"""Unit tests for prompt templates and chaining."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.prompts.templates import (
    CLASSIFICATION,
    EXTRACTION,
    RAG_QA,
    SUMMARIZATION,
    PromptTemplate,
    get_template,
)
from src.prompts.chain import ChainContext, ChainStep, PromptChain
from src.core.base_llm import LLMResponse, Message


# ─── PromptTemplate ───────────────────────────────────────────────────────────

class TestPromptTemplate:
    def test_format_basic(self):
        t = PromptTemplate(
            name="test",
            template="Hello, $name!",
            input_variables=["name"],
        )
        assert t.format(name="World") == "Hello, World!"

    def test_format_missing_variable_raises(self):
        t = PromptTemplate(
            name="test",
            template="Hello, $name!",
            input_variables=["name"],
        )
        with pytest.raises(KeyError, match="name"):
            t.format()

    def test_partial_fills_some_variables(self):
        t = PromptTemplate(
            name="test",
            template="$greeting, $name!",
            input_variables=["greeting", "name"],
        )
        partial = t.partial(greeting="Hi")
        assert partial.input_variables == ["name"]
        assert "Hi" in partial.format(name="Alice")

    def test_rag_qa_template(self):
        rendered = RAG_QA.format(context="Paris is the capital.", question="What is the capital?")
        assert "Paris is the capital." in rendered
        assert "What is the capital?" in rendered

    def test_summarization_template(self):
        rendered = SUMMARIZATION.format(text="Long text here.", max_sentences="3")
        assert "3" in rendered

    def test_extraction_template(self):
        rendered = EXTRACTION.format(schema='{"name": "string"}', text="John lives here.")
        assert "John lives here." in rendered

    def test_classification_template(self):
        rendered = CLASSIFICATION.format(categories="positive, negative", text="Great product!")
        assert "positive, negative" in rendered


class TestTemplateRegistry:
    def test_get_known_template(self):
        t = get_template("rag_qa")
        assert t.name == "rag_qa"

    def test_get_unknown_template_raises(self):
        with pytest.raises(KeyError):
            get_template("nonexistent_template")


# ─── ChainContext ─────────────────────────────────────────────────────────────

class TestChainContext:
    def test_set_and_get(self):
        ctx = ChainContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_update(self):
        ctx = ChainContext(variables={"a": 1})
        ctx.update({"b": 2, "c": 3})
        assert ctx.variables == {"a": 1, "b": 2, "c": 3}


# ─── PromptChain ──────────────────────────────────────────────────────────────

class TestPromptChain:
    def _make_llm(self, responses: list[str]):
        """Return a mock LLM that returns each string in order."""
        llm = MagicMock()
        llm.complete.side_effect = [
            LLMResponse(content=r, model="mock") for r in responses
        ]
        return llm

    def test_single_step_chain(self):
        llm = self._make_llm(["Summary: short text."])
        step = ChainStep(
            name="summary",
            template=SUMMARIZATION,
        )
        chain = PromptChain(llm=llm, steps=[step])
        ctx = chain.run({"text": "A very long document.", "max_sentences": "1"})
        assert ctx.get("summary") == "Summary: short text."

    def test_multi_step_chain_passes_context(self):
        llm = self._make_llm(["Step1 output", "Step2 used Step1 output"])
        step1 = ChainStep(
            name="step1",
            template=SUMMARIZATION,
        )
        step2 = ChainStep(
            name="step2",
            template=PromptTemplate(
                name="t2", template="Classify: $step1", input_variables=["step1"]
            ),
        )
        chain = PromptChain(llm=llm, steps=[step1, step2])
        ctx = chain.run({"text": "Document text", "max_sentences": "2"})
        assert ctx.get("step1") == "Step1 output"
        assert "Step1 output" in ctx.get("step2")

    def test_postprocess_applied(self):
        llm = self._make_llm(["  trimmed  "])
        step = ChainStep(
            name="clean",
            template=SUMMARIZATION,
            postprocess=str.strip,
        )
        chain = PromptChain(llm=llm, steps=[step])
        ctx = chain.run({"text": "Text", "max_sentences": "1"})
        assert ctx.get("clean") == "trimmed"

    def test_capture_output_false(self):
        llm = self._make_llm(["ignored"])
        step = ChainStep(
            name="skip_me",
            template=SUMMARIZATION,
            capture_output=False,
        )
        chain = PromptChain(llm=llm, steps=[step])
        ctx = chain.run({"text": "Text", "max_sentences": "1"})
        assert "skip_me" not in ctx.variables
