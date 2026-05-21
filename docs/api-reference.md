# API Reference

Public Python API for every module in `src/`. All classes use `from __future__ import annotations` and Python dataclasses throughout.

---

## `src.core.document_analysis_pipeline`

### `AnalysisConfig`

Runtime configuration dataclass for the full pipeline.

```python
@dataclass
class AnalysisConfig:
    provider: str = "huggingface"       # "huggingface" | "llm"
    llm_provider: str = "bedrock"       # backend when provider=="llm"
    llm_model: str | None = None        # None = use provider default
    output_dir: str = "data/output"
    cache_dir: str = "data/cache"
    summarization_model: str = "t5-small"
    summary_max_length: int = 150
    summary_min_length: int = 30
    passage_word_limit: int = 200
    qg_model: str = "valhalla/t5-base-qg-hl"
    min_questions_per_passage: int = 3
    qa_model: str = "deepset/roberta-base-squad2"
```

---

### `DocumentAnalysisPipeline`

Orchestrates all 7 pipeline steps.

```python
class DocumentAnalysisPipeline:
    def __init__(
        self,
        config: AnalysisConfig | None = None,
        llm: BaseLLM | None = None,
    ) -> None
```

| Parameter | Type | Description |
|---|---|---|
| `config` | `AnalysisConfig` | Pipeline configuration; uses defaults when omitted |
| `llm` | `BaseLLM` | LLM client; required when `config.provider == "llm"` |

#### `run(pdf_path, preview_chars=500) → AnalysisResult`

Execute all pipeline steps and return a fully populated result.

```python
result = pipeline.run("path/to/doc.pdf", preview_chars=500)
```

#### `save_results(result, output_dir=None) → Path`

Serialise the result to a JSON file and return its path.

```python
out_path = pipeline.save_results(result, output_dir="examples/output")
```

#### `print_results(result) → None`

Pretty-print the result to stdout (text preview, summary, all Q&A pairs).

---

### `AnalysisResult`

```python
@dataclass
class AnalysisResult:
    source: str                          # Absolute path to the PDF
    extracted_text: str                  # Cleaned full text
    text_preview: str                    # First N characters of raw text
    summary: str                         # Document summary
    num_passages: int
    passages: list[PassageAnalysis]      # Per-passage breakdown
    all_qa_pairs: list[dict]             # Flat list of all Q&A dicts

    def to_dict(self) -> dict
    def to_json(indent=2) -> str
```

Each element of `all_qa_pairs`:

```python
{"question": str, "answer": str, "score": float}
```

---

### `PassageAnalysis`

```python
@dataclass
class PassageAnalysis:
    passage_index: int
    passage: str
    questions: list[str]
    qa_pairs: list[dict]    # same structure as AnalysisResult.all_qa_pairs
```

---

## `src.core.model_factory`

### `create_llm`

Factory function that lazy-imports and instantiates the correct LLM client.

```python
def create_llm(
    provider: str | ModelProvider,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
    **extra,
) -> BaseLLM
```

| `provider` | Default model | Extra kwargs |
|---|---|---|
| `"openai"` | `gpt-4o` | — (reads `OPENAI_API_KEY`) |
| `"anthropic"` | `claude-3-5-sonnet-20241022` | — (reads `ANTHROPIC_API_KEY`) |
| `"bedrock"` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | `region_name`, `profile_name`, `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token` |
| `"local"` | `llama-3.1-8b-instruct` | — |

```python
from src.core.model_factory import create_llm

llm = create_llm("bedrock", region_name="us-east-1")
llm = create_llm("openai", model="gpt-4o-mini", temperature=0.3)
llm = create_llm("anthropic")
```

---

## `src.core.base_llm`

### `BaseLLM`

Abstract base class for all LLM clients.

```python
class BaseLLM(ABC):
    config: LLMConfig

    # Synchronous
    def complete(self, messages: list[Message]) -> LLMResponse: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...

    # Async
    async def acomplete(self, messages: list[Message]) -> LLMResponse: ...
    async def astream(self, messages: list[Message]) -> AsyncIterator[str]: ...

    # Convenience
    def chat(self, user_message: str, system_prompt: str | None = None) -> str: ...
```

### `Message`

```python
@dataclass
class Message:
    role: str       # "system" | "user" | "assistant"
    content: str
```

### `LLMResponse`

```python
@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int]       # e.g. {"input_tokens": 100, "output_tokens": 50}
    finish_reason: str          # "stop" | "length" | …
    raw: Any                    # Raw provider response object
```

### `LLMConfig`

```python
@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
```

---

## `src.core.bedrock_client`

### `BedrockClient`

```python
class BedrockClient(BaseLLM):
    def __init__(
        self,
        config: LLMConfig,
        region_name: str = "us-east-1",
        profile_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ) -> None
```

Uses the Bedrock **Converse** and **ConverseStream** APIs so it works with every foundation model on Bedrock without model-specific code.

---

## `src.inference.summarizer`

### `Summarizer`

```python
class Summarizer:
    def __init__(self, config: SummaryConfig | None = None, llm: BaseLLM | None = None)

    def summarize(self, text: str) -> str
    def summarize_passages(self, passages: list[str]) -> str
```

### `SummaryConfig`

```python
@dataclass
class SummaryConfig:
    provider: str = "huggingface"     # "huggingface" | "llm"
    hf_model: str = "t5-small"
    max_length: int = 150
    min_length: int = 30
    do_sample: bool = False
    llm_chunk_tokens: int = 3000
```

---

## `src.inference.question_generator`

### `QuestionGenerator`

```python
class QuestionGenerator:
    def __init__(
        self,
        provider: str = "huggingface",
        hf_model: str = "valhalla/t5-base-qg-hl",
        llm: BaseLLM | None = None,
        min_questions: int = 3,
    )

    def generate(self, passage: str, min_questions: int | None = None) -> list[str]
    def generate_all(self, passages: list[str], min_questions: int | None = None) -> list[dict]
```

---

## `src.inference.qa_engine`

### `QAEngine`

```python
class QAEngine:
    def __init__(
        self,
        provider: str = "huggingface",
        hf_model: str = "deepset/roberta-base-squad2",
        llm: BaseLLM | None = None,
    )

    def answer(self, question: str, context: str) -> QAResult
    def answer_batch(self, questions: list[str], context: str) -> list[QAResult]
```

### `QAResult`

```python
@dataclass
class QAResult:
    question: str
    answer: str
    score: float = 0.0      # Confidence 0.0–1.0 (HF mode); 0.0 in LLM mode
    passage_index: int = 0
```

---

## `src.processing.pdf_extractor`

### `PDFExtractor`

```python
class PDFExtractor:
    def __init__(self, cache_dir: str = "data/cache")

    def extract(self, pdf_path: str | Path) -> str
```

Extracts plain text using `pdfplumber`. On subsequent calls with the same PDF binary content, the cached result is returned instantly. The cache key is `SHA-256(binary) + file_size`.

---

## `src.processing.preprocessing`

### `TextPreprocessor`

```python
class TextPreprocessor:
    def __init__(
        self,
        remove_html: bool = True,
        normalize_ws: bool = True,
        remove_urls: bool = False,
    )

    def process(self, text: str) -> str
```

---

## `src.processing.chunking`

### `SentenceChunker`

Word-limited sentence chunker; used for passage creation.

```python
class SentenceChunker:
    def __init__(self, config: SentenceChunkingConfig | None = None)
    def split(self, text: str) -> list[str]

@dataclass
class SentenceChunkingConfig:
    word_limit: int = 200
```

### `TextChunker`

Character-overlap chunker; used for embedding.

```python
class TextChunker:
    def __init__(self, config: ChunkingConfig | None = None)
    def split(self, text: str) -> list[str]

@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    sentence_split: bool = True
```

---

## `src.rag.embedder`

### `Embedder`

```python
class Embedder:
    def __init__(
        self,
        provider: str = "openai",             # "openai" | "huggingface"
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        api_key: str | None = None,
    )

    def embed(self, text: str) -> list[float]
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]
```

---

## `src.rag.vector_store`

### `VectorStore`

```python
class VectorStore:
    def __init__(
        self,
        provider: str = "chroma",             # "chroma" | "faiss"
        collection_name: str = "default",
        persist_directory: str = "data/vectordb",
        distance_metric: str = "cosine",
    )

    def upsert(self, documents: list[Document]) -> None
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]
    def delete(self, ids: list[str]) -> None
    def count(self) -> int
```

### `Document`

```python
@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
```

### `SearchResult`

```python
@dataclass
class SearchResult:
    document: Document
    score: float          # Similarity score (higher = more similar)
```

---

## `src.rag.retriever`

### `Retriever`

```python
class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
    )

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]

@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.0
    metadata_filter: dict | None = None
```

Returns an empty list for empty or whitespace-only queries.

---

## `src.prompts.templates`

### `PromptTemplate`

```python
@dataclass
class PromptTemplate:
    name: str
    template: str                     # Uses $variable syntax (Python string.Template)
    input_variables: list[str] = field(default_factory=list)
    description: str = ""

    def format(self, **kwargs) -> str          # Raises KeyError for missing vars
    def partial(self, **kwargs) -> PromptTemplate  # Returns new template with vars pre-filled
```

### Built-in Templates

| Name | Variables | Purpose |
|---|---|---|
| `RAG_QA` | `context`, `question` | Answer a question from retrieved passages |
| `SUMMARIZATION` | `text`, `max_sentences` | Summarise a block of text |
| `QUESTION_GENERATION` | `passage`, `num_questions` | Generate comprehension questions |
| `DOCUMENT_QA` | `context`, `question` | QA for the document analysis pipeline |
| `DOCUMENT_SUMMARY` | `text` | Document summarisation for the pipeline |

```python
from src.prompts.templates import RAG_QA

prompt = RAG_QA.format(context="The sky is blue.", question="What colour is the sky?")
```

---

## `src.prompts.chain`

### `PromptChain`

```python
class PromptChain:
    def __init__(self, llm: BaseLLM, steps: list[ChainStep])

    async def arun(self, initial_vars: dict) -> ChainContext
```

### `ChainStep`

```python
@dataclass
class ChainStep:
    name: str
    template: PromptTemplate
    postprocess: Callable[[str], Any] | None = None
    capture_output: bool = True
```

### `ChainContext`

```python
@dataclass
class ChainContext:
    variables: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None
    def get(self, key: str) -> Any
    def update(self, data: dict[str, Any]) -> None
```

Usage:

```python
from src.prompts.chain import PromptChain, ChainStep
from src.prompts.templates import SUMMARIZATION, RAG_QA

chain = PromptChain(llm=llm, steps=[
    ChainStep(name="summary", template=SUMMARIZATION),
    ChainStep(name="answer",  template=RAG_QA),
])

ctx = await chain.arun({"text": document_text, "max_sentences": "3",
                         "context": document_text, "question": "What is the main contribution?"})
print(ctx.get("answer"))
```
