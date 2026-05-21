# API Reference — `src.prompts`

Source: [`src/prompts/`](../../src/prompts/)  
Cross-references: [Core](core.md) · [Inference](inference.md) · [User Guide → Learn Basics](../user-guide/learn-basics.md#prompt-injection-defence)

---

## `templates.py`

String constants used throughout `src/`. Import the constant, never hardcode prompts inline.

### System prompts

| Constant | Purpose |
|----------|---------|
| `RAG_SYSTEM_PROMPT` | Generic RAG grounding instruction |
| `SUMMARIZE_SYSTEM_PROMPT` | Summarisation role and format |
| `QA_SYSTEM_PROMPT` | Extractive/abstractive QA instruction |
| `RESEARCH_SYSTEM_PROMPT` | Research synthesis with citation guidance |
| `TRIPLE_EXTRACTION_PROMPT` | Entity-relation-entity extraction format |
| `INTENT_CLASSIFICATION_PROMPT` | SEARCH vs DIRECT intent classification |
| `GRAPH_CONTEXT_PROMPT` | Graph-grounded answer generation |

### Source delimiter constants

```python
SOURCE_BEGIN = "=== BEGIN SOURCES ==="
SOURCE_END   = "=== END SOURCES ==="
```

Used by `RealtimeRAGAssistant` and `ResearchAgent` to fence external content
against prompt injection.

### Usage

```python
from src.prompts.templates import RAG_SYSTEM_PROMPT, SOURCE_BEGIN, SOURCE_END

prompt = (
    RAG_SYSTEM_PROMPT
    + f"\n\n{SOURCE_BEGIN}\n"
    + "\n\n".join(passages)
    + f"\n{SOURCE_END}\n\n"
    + f"Question: {question}"
)
```

---

## `chain.py`

LangChain-compatible prompt builders. Used by `LangChainRAGAgent`.

### `build_rag_prompt()`

```python
def build_rag_prompt() -> ChatPromptTemplate:
    """
    Return a LangChain ChatPromptTemplate with:
      - SystemMessage: RAG_SYSTEM_PROMPT
      - HumanMessagePromptTemplate: context + question placeholders
    Variables: {context}, {question}
    """
```

### `build_research_prompt()`

```python
def build_research_prompt() -> ChatPromptTemplate:
    """
    Return a LangChain ChatPromptTemplate for the research synthesis task.
    Variables: {passages}, {query}
    """
```

### `build_qa_prompt()`

```python
def build_qa_prompt() -> ChatPromptTemplate:
    """
    Return a ChatPromptTemplate for QA.
    Variables: {context}, {question}
    """
```

**Used by:** `LangChainRAGAgent._run_chain()`
