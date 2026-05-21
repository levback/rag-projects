"""Prompt templates for common GenAI tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Any


@dataclass
class PromptTemplate:
    """A reusable prompt template with named placeholders (``$variable`` syntax)."""

    name: str
    template: str
    input_variables: list[str] = field(default_factory=list)
    description: str = ""

    def format(self, **kwargs: Any) -> str:
        """Substitute *kwargs* into the template and return the resulting string.

        Raises:
            KeyError: If a required variable is missing.
        """
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            raise KeyError(f"Missing template variables: {missing}")
        return Template(self.template).safe_substitute(**kwargs)

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """Return a new template with some variables pre-filled."""
        pre_filled = Template(self.template).safe_substitute(**kwargs)
        remaining = [v for v in self.input_variables if v not in kwargs]
        return PromptTemplate(
            name=self.name,
            template=pre_filled,
            input_variables=remaining,
            description=self.description,
        )


# ─── Built-in templates ───────────────────────────────────────────────────────

RAG_QA = PromptTemplate(
    name="rag_qa",
    description="Answer a question using retrieved context passages.",
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Answer the question using only the context below.\n"
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n$context\n\n"
        "Question: $question\n\n"
        "Answer:"
    ),
)

SUMMARIZATION = PromptTemplate(
    name="summarization",
    description="Summarise a block of text.",
    input_variables=["text", "max_sentences"],
    template=(
        "Summarise the following text in at most $max_sentences sentences.\n\n"
        "Text:\n$text\n\n"
        "Summary:"
    ),
)

EXTRACTION = PromptTemplate(
    name="extraction",
    description="Extract structured information from text.",
    input_variables=["schema", "text"],
    template=(
        "Extract the following information from the text and return valid JSON matching the schema.\n\n"
        "Schema:\n$schema\n\n"
        "Text:\n$text\n\n"
        "JSON:"
    ),
)

CLASSIFICATION = PromptTemplate(
    name="classification",
    description="Classify text into one of the provided categories.",
    input_variables=["categories", "text"],
    template=(
        "Classify the text into exactly one of the following categories: $categories.\n"
        "Return only the category name, nothing else.\n\n"
        "Text: $text\n\n"
        "Category:"
    ),
)

SYSTEM_DEFAULT = PromptTemplate(
    name="system_default",
    description="Generic helpful-assistant system prompt.",
    input_variables=[],
    template=(
        "You are a knowledgeable, precise, and helpful AI assistant. "
        "Think step-by-step and cite sources when available."
    ),
)

# ─── Document analysis templates ─────────────────────────────────────────────

DOCUMENT_SUMMARY = PromptTemplate(
    name="document_summary",
    description="Produce a structured summary of a full document.",
    input_variables=["document"],
    template=(
        "You are an expert document analyst. Read the document below and produce a "
        "well-structured summary covering: key topics, main points, important clauses or "
        "definitions, and any notable limitations or obligations.\n\n"
        "Document:\n$document\n\n"
        "Summary:"
    ),
)

DOCUMENT_QA = PromptTemplate(
    name="document_qa",
    description="Answer a question using only the provided document passage.",
    input_variables=["passage", "question"],
    template=(
        "Answer the question using ONLY the passage below. "
        "If the answer is not present, respond with 'Not found in passage'.\n\n"
        "Passage:\n$passage\n\n"
        "Question: $question\n\n"
        "Answer:"
    ),
)

QUESTION_GENERATION = PromptTemplate(
    name="question_generation",
    description="Generate comprehension questions from a text passage.",
    input_variables=["passage", "num_questions"],
    template=(
        "Read the passage below and generate exactly $num_questions distinct comprehension "
        "questions whose answers can be found in the passage. "
        "Return only the questions, one per line, with no numbering or bullet points.\n\n"
        "Passage:\n$passage\n\n"
        "Questions:"
    ),
)

KEY_ENTITIES = PromptTemplate(
    name="key_entities",
    description="Extract named entities and key concepts from a passage.",
    input_variables=["passage"],
    template=(
        "Extract all named entities (people, organisations, dates, locations, legal terms) "
        "and key concepts from the passage. Return a JSON object with keys: "
        "'entities', 'key_concepts', 'dates'.\n\n"
        "Passage:\n$passage\n\n"
        "JSON:"
    ),
)

DOCUMENT_SYSTEM = PromptTemplate(
    name="document_system",
    description="System prompt for document analysis tasks.",
    input_variables=[],
    template=(
        "You are a precise legal and business document analyst. "
        "Extract information faithfully from the provided text. "
        "Never fabricate information that is not present in the source material."
    ),
)

# Registry for lookup by name
REGISTRY: dict[str, PromptTemplate] = {
    t.name: t
    for t in [
        RAG_QA,
        SUMMARIZATION,
        EXTRACTION,
        CLASSIFICATION,
        SYSTEM_DEFAULT,
        DOCUMENT_SUMMARY,
        DOCUMENT_QA,
        QUESTION_GENERATION,
        KEY_ENTITIES,
        DOCUMENT_SYSTEM,
    ]
}


def get_template(name: str) -> PromptTemplate:
    """Retrieve a built-in template by *name*.

    Raises:
        KeyError: If *name* is not found in the registry.
    """
    if name not in REGISTRY:
        raise KeyError(f"Template {name!r} not found. Available: {list(REGISTRY)}")
    return REGISTRY[name]
