# API Reference — `src.core`

Source: [`src/core/`](../../src/core/)  
Cross-references: [RAG](rag.md) · [Agents](agents.md) · [User Guide → Main Components](../user-guide/main-components.md#srccore--llm-layer)

---

## `base_llm.py`

### `Message`

```python
@dataclass
class Message:
    role: str      # "user" | "assistant" | "system"
    content: str
```

Single conversation turn. Used in `complete()` and `stream()`.

---

### `LLMResponse`

```python
@dataclass
class LLMResponse:
    text: str                    # Generated text
    model: str                   # Model identifier used
    input_tokens: int            # Prompt token count
    output_tokens: int           # Completion token count
    finish_reason: str           # "stop" | "length" | "tool_calls"
```

Returned by `BaseLLM.complete()`.

---

### `LLMConfig`

```python
@dataclass
class LLMConfig:
    model: str          # Model ID (provider-specific)
    temperature: float  # 0.0 – 1.0 (default: 0.7)
    max_tokens: int     # Maximum output tokens (default: 1024)
    timeout: int        # Request timeout seconds (default: 30)
```

Passed to all `BaseLLM` constructors.

---

### `BaseLLM`

```python
class BaseLLM(ABC):
    def __init__(self, config: LLMConfig) -> None: ...

    @abstractmethod
    def complete(self, messages: list[Message]) -> LLMResponse: ...

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]: ...

    def chat(self, user_message: str,
             system_prompt: str | None = None) -> str: ...
```

`chat()` is a convenience wrapper: builds a `[Message]` list and returns
`complete().text`.

**Implemented by:** [`GPTClient`](#gptclient) · [`ClaudeClient`](#claudeclient) ·
[`BedrockClient`](#bedrockclient) · [`LocalLLM`](#localllm)

---

## `gpt_client.py`

### `GPTClient`

```python
class GPTClient(BaseLLM):
    def __init__(self, config: LLMConfig,
                 api_key: str | None = None) -> None: ...
    def complete(self, messages: list[Message]) -> LLMResponse: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
```

Calls `openai.chat.completions.create()`. Uses `OPENAI_API_KEY` env var when
`api_key` is `None`.

**Default model:** `gpt-4o-mini`

---

## `claude_client.py`

### `ClaudeClient`

```python
class ClaudeClient(BaseLLM):
    def __init__(self, config: LLMConfig,
                 api_key: str | None = None) -> None: ...
    def complete(self, messages: list[Message]) -> LLMResponse: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
```

Calls `anthropic.Anthropic().messages.create()`. Uses `ANTHROPIC_API_KEY` env var.

**Default model:** `claude-3-5-sonnet-20241022`

---

## `bedrock_client.py`

### `BedrockClient`

```python
class BedrockClient(BaseLLM):
    def __init__(
        self,
        config: LLMConfig,
        region: str = "us-east-1",
        profile: str | None = None,
        role_arn: str | None = None,
    ) -> None: ...
    def complete(self, messages: list[Message]) -> LLMResponse: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
```

Uses `boto3` `bedrock-runtime` `converse()` / `converse_stream()` APIs.
Supports IAM role assumption via `role_arn`.

**Default model:** `anthropic.claude-3-5-sonnet-20241022-v2:0`

**Cross-reference:** [Installation → AWS Bedrock setup](../installation.md#aws-bedrock--additional-setup)

---

## `local_llm.py`

### `LocalLLM`

```python
class LocalLLM(BaseLLM):
    def __init__(self, config: LLMConfig,
                 model_path: str | None = None) -> None: ...
    def complete(self, messages: list[Message]) -> LLMResponse: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
```

HuggingFace `transformers.pipeline("text-generation")`. No API key needed.
Downloads model on first use to `~/.cache/huggingface/`.

**Default model:** `google/flan-t5-base`

---

## `model_factory.py`

### `ModelProvider`

```python
class ModelProvider(str, Enum):
    OPENAI     = "openai"
    ANTHROPIC  = "anthropic"
    BEDROCK    = "bedrock"
    LOCAL      = "local"
```

### `create_llm()`

```python
def create_llm(
    provider: str | ModelProvider = "bedrock",
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **extra: Any,
) -> BaseLLM:
```

Factory function. Reads API keys / AWS credentials from environment, returns
the appropriate `BaseLLM` subclass.

**`extra` kwargs for Bedrock:** `region`, `profile`, `role_arn`

**Used by:** all 10 examples in `--provider` mode

---

## `document_analysis_pipeline.py`

### `AnalysisConfig`

```python
@dataclass
class AnalysisConfig:
    provider: str           # "huggingface" | "llm"
    llm_provider: str       # "bedrock" | "openai" | "anthropic" | "local"
    batch_size: int         # Chunks processed per LLM call
    max_questions: int      # Questions generated per passage
    chunk_size: int         # Characters per chunk
    chunk_overlap: int      # Overlap between chunks
    summarise: bool         # Whether to run summarisation step
    generate_questions: bool
    answer_questions: bool
    output_dir: str | Path  # Where to write results
```

### `PassageAnalysis`

```python
@dataclass
class PassageAnalysis:
    passage: str
    questions: list[str]
    answers: list[QAResult]
```

### `AnalysisResult`

```python
@dataclass
class AnalysisResult:
    pdf_path: str
    summary: str
    passage_analyses: list[PassageAnalysis]
    metadata: dict

    def to_dict(self) -> dict: ...
    def to_json(self, indent: int = 2) -> str: ...
```

### `DocumentAnalysisPipeline`

```python
class DocumentAnalysisPipeline:
    def __init__(
        self,
        config: AnalysisConfig | None = None,
        llm: BaseLLM | None = None,
    ) -> None: ...

    def run(self, pdf_path: str | Path,
            preview_chars: int = 500) -> AnalysisResult: ...

    def save_results(
        self,
        result: AnalysisResult,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]: ...

    def print_results(self, result: AnalysisResult) -> None: ...
```

**Example:** [`examples/10_document_analysis.py`](../../examples/10_document_analysis.py)  
**Cross-reference:** [User Guide → Main Components](../user-guide/main-components.md#document_analysis_pipelinepy--documentanalysispipeline)
