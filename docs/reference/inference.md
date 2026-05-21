# API Reference — `src.inference`

Source: [`src/inference/`](../../src/inference/)  
Cross-references: [Core](core.md) · [Processing](processing.md) · [User Guide → Main Components](../user-guide/main-components.md#srcinference--inference-layer)

---

## `inference_engine.py`

### `InferenceRequest`

```python
@dataclass
class InferenceRequest:
    query: str                     # The user question
    passages: list[str]            # Retrieved context passages
    sources: list[str] | None = None  # Optional source labels for passages
    max_tokens: int = 512
```

### `InferenceResult`

```python
@dataclass
class InferenceResult:
    query: str
    answer: str
    passages_used: int
    sources: list[str]
```

### `InferenceEngine`

```python
class InferenceEngine:
    def __init__(self, llm: BaseLLM) -> None: ...

    def answer(self, request: InferenceRequest) -> InferenceResult:
        """
        Build a grounded prompt from query + passages,
        call LLM, parse answer.
        """

    def answer_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResult]:
        """Process multiple requests sequentially."""
```

---

## `summarizer.py`

### `SummaryConfig`

```python
@dataclass
class SummaryConfig:
    provider: str        = "huggingface"   # "huggingface" | "llm"
    model: str           = "t5-small"      # HuggingFace model ID
    max_length: int      = 150             # Max tokens in summary
    min_length: int      = 30              # Min tokens in summary
    chunk_size: int      = 1000            # Input chunking for long texts
```

### `Summarizer`

```python
class Summarizer:
    def __init__(
        self,
        config: SummaryConfig | None = None,
        llm: BaseLLM | None = None,
    ) -> None: ...

    def summarize(self, text: str) -> str:
        """
        Summarise text.
        - HuggingFace: splits to chunk_size, runs T5, joins.
        - LLM: single `complete()` call with summarise system prompt.
        """

    def summarize_passages(self, passages: list[str]) -> str:
        """Join passages and summarise as a unit."""
```

**Default HF model:** `t5-small` (summarization task)  
**Used by:** `DocumentAnalysisPipeline.run()`

---

## `qa_engine.py`

### `QAResult`

```python
@dataclass
class QAResult:
    question: str
    answer: str
    score: float     # Confidence (0.0 – 1.0); 1.0 for LLM mode
    start: int       # Character offset in context (-1 for LLM mode)
    end: int         # Character offset end    (-1 for LLM mode)
```

### `QAEngine`

```python
class QAEngine:
    def __init__(
        self,
        provider: str = "huggingface",
        model: str    = "deepset/roberta-base-squad2",
        llm: BaseLLM | None = None,
    ) -> None: ...

    def answer(self, question: str, context: str) -> QAResult:
        """
        Extractive QA (HuggingFace) or abstractive (LLM).
        HF: runs deepset/roberta-base-squad2 pipeline.
        LLM: builds grounded prompt, calls complete().
        """

    def answer_batch(
        self,
        questions: list[str],
        context: str,
    ) -> list[QAResult]:
        """Answer a list of questions against the same context."""

    def answer_passages(
        self,
        question: str,
        passages: list[str],
    ) -> QAResult:
        """Concatenate passages and answer question against the union."""
```

**Used by:** `DocumentAnalysisPipeline.run()`

---

## `question_generator.py`

### `QuestionGenerator`

```python
class QuestionGenerator:
    def __init__(
        self,
        model: str    = "valhalla/t5-base-qg-hl",
        provider: str = "huggingface",
        llm: BaseLLM | None = None,
    ) -> None: ...

    def generate(self, text: str,
                 num_questions: int = 3) -> list[str]:
        """
        Generate questions from a text passage.
        HuggingFace: highlight-based T5 question generation.
        LLM: prompt-based generation.
        """

    def generate_all(
        self,
        passages: list[str],
        num_questions: int = 3,
    ) -> list[list[str]]:
        """Return per-passage question lists."""
```

**Default HF model:** `valhalla/t5-base-qg-hl`  
**Used by:** `DocumentAnalysisPipeline.run()`

---

## `response_parser.py`

### `ParsedResponse`

```python
@dataclass
class ParsedResponse:
    text: str            # Cleaned answer text
    json_data: dict | None  # Parsed JSON if detected
    confidence: float    # Heuristic confidence (0.0 – 1.0)
```

### `ResponseParser`

Static-method utility class. No instantiation needed.

```python
class ResponseParser:
    @staticmethod
    def extract_json(text: str) -> dict | None:
        """Find and parse first JSON object or array in text."""

    @staticmethod
    def strip_markdown(text: str) -> str:
        """Remove ``` code fences and markdown bold/italic markers."""

    @staticmethod
    def extract_answer(text: str) -> str:
        """
        Extract answer from structured LLM output.
        Handles formats:
          - "Answer: <text>"
          - "**Answer**: <text>"
          - Plain text (returned as-is)
        """

    @staticmethod
    def clean_response(text: str) -> str:
        """strip_markdown → normalise whitespace → strip()."""

    @staticmethod
    def parse(text: str) -> ParsedResponse:
        """Full parse: attempt JSON extraction, clean text, compute confidence."""
```
