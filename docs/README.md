# Documentation Index

This directory contains all technical documentation for the Document Analysis project.

---

## Documents

| Document | Description |
|---|---|
| [Architecture](architecture.md) | System design, component diagram, data-flow, design decisions |
| [User Guide](user-guide.md) | Step-by-step usage for every backend (HuggingFace, Bedrock, OpenAI, Anthropic) |
| [Configuration](configuration.md) | `model_config.yaml` reference, environment variables, `AnalysisConfig` fields |
| [API Reference](api-reference.md) | Public Python API for every module in `src/` |
| [Development Guide](development.md) | Dev setup, testing, adding new backends, conventions |

For the quick-start and project overview, see the [root README](../README.md).

---

## Pipeline Overview

```
PDF File
   │
   ▼
[Step 1] PDFExtractor          ← pdfplumber · disk-cached by SHA-256 hash
   │
   ▼
[Step 2] Text Preview          ← first N chars logged / printed
   │
   ▼
[Step 3] TextPreprocessor      ← normalise whitespace, strip HTML
   │
   ▼
[Step 4] SentenceChunker       ← NLTK sent_tokenize · word-limited passages
   │
   ▼
[Step 5] Summarizer            ← t5-small (HF) OR any LLM via DOCUMENT_SUMMARY prompt
   │
   ▼
[Step 6] QuestionGenerator     ← valhalla/t5-base-qg-hl (HF) OR LLM
   │
   ▼
[Step 7] QAEngine              ← deepset/roberta-base-squad2 (HF) OR LLM
   │
   ▼
AnalysisResult  →  JSON file · stdout report
```

---

## Project Structure

```
document_analysis/
├── config/
│   ├── model_config.yaml          # All model + pipeline settings
│   └── logging_config.yaml        # Rotating-file + console logging
├── data/
│   ├── cache/                     # Extracted PDF text cache
│   ├── embeddings/                # (RAG) vector embeddings
│   ├── output/                    # Analysis JSON results
│   ├── sample/                    # Drop your PDF files here
│   └── vectordb/                  # (RAG) vector database files
├── src/
│   ├── core/
│   │   ├── base_llm.py            # BaseLLM ABC (sync + async)
│   │   ├── gpt_client.py          # OpenAI GPT client
│   │   ├── claude_client.py       # Anthropic Claude client
│   │   ├── local_llm.py           # HuggingFace local model client
│   │   ├── model_factory.py       # create_llm() factory
│   │   └── document_analysis_pipeline.py  ← NEW orchestrator
│   ├── processing/
│   │   ├── pdf_extractor.py       ← NEW  PDF → text (pdfplumber + cache)
│   │   ├── chunking.py            ← UPDATED  TextChunker + SentenceChunker
│   │   ├── tokenizer.py           # Token counting & truncation
│   │   └── preprocessing.py       # Text cleaning pipeline
│   ├── inference/
│   │   ├── summarizer.py          ← NEW  t5-small or LLM summarisation
│   │   ├── question_generator.py  ← NEW  valhalla/t5-base-qg-hl or LLM
│   │   ├── qa_engine.py           ← NEW  roberta-base-squad2 or LLM
│   │   ├── inference_engine.py    # RAG-augmented generation
│   │   └── response_parser.py     # JSON / markdown output parsing
│   ├── prompts/
│   │   ├── templates.py           ← UPDATED  + document analysis templates
│   │   └── chain.py               # Multi-step prompt chaining
│   └── rag/
│       ├── embedder.py            # Dense embedding creation
│       ├── retriever.py           # Similarity search
│       ├── vector_store.py        # Chroma / FAISS wrapper
│       └── indexer.py             # File → chunk → embed → store
├── tests/
│   ├── unit/
│   │   ├── test_document_analysis.py  ← NEW
│   │   ├── test_llm_clients.py
│   │   └── test_prompts.py
│   └── integration/
│       ├── test_end_to_end.py
│       └── test_api_integration.py
├── scripts/
│   ├── setup_env.sh               # Bootstrap virtualenv
│   ├── run_tests.sh               # pytest with coverage
│   ├── build_embeddings.sh        # Index a document directory
│   └── cleanup.py                 # Remove generated artefacts
├── main.py                        ← NEW CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

### 2. Add a PDF

```bash
cp /path/to/your/document.pdf data/sample/
```

### 3. Run with local HuggingFace models (no API key required)

```bash
python main.py --pdf data/sample/document.pdf
```

On first run, the three HuggingFace models are downloaded automatically (~1 GB total).  
Results are written to `data/output/<stem>_analysis.json` and printed to the terminal.

### 4. Run with an LLM API (GPT or Claude)

```bash
export OPENAI_API_KEY=sk-...
python main.py --pdf data/sample/document.pdf --provider llm --llm-provider openai
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --pdf data/sample/document.pdf --provider llm --llm-provider anthropic
```

---

## CLI Reference

```
python main.py --pdf PATH [options]

Options:
  --provider {huggingface,llm}     Backend for all generative steps. Default: huggingface
  --llm-provider {openai,anthropic,local}
                                   LLM provider when --provider=llm. Default: openai
  --llm-model MODEL                Override default model for the chosen provider
  --output DIR                     Results output directory. Default: data/output
  --cache-dir DIR                  PDF text cache directory. Default: data/cache
  --word-limit N                   Max words per passage. Default: 200
  --min-questions N                Min questions per passage. Default: 3
  --preview-chars N                Characters shown in text preview. Default: 500
  --preview-only                   Extract and preview only; skip generative steps
  --skip-qa                        Summarise only; skip question generation and QA
  --no-save                        Do not write results to disk
  -v, --verbose                    Enable DEBUG logging
```

---

## Python API

```python
from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

# HuggingFace mode
config = AnalysisConfig(provider="huggingface")
pipeline = DocumentAnalysisPipeline(config=config)
result = pipeline.run("data/sample/document.pdf")

pipeline.print_results(result)
pipeline.save_results(result)

# Access structured results
print(result.summary)
for pair in result.all_qa_pairs:
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer']}")
```

```python
# LLM mode (uses GPT-4o for everything)
from src.core.model_factory import create_llm

llm = create_llm("openai", model="gpt-4o-mini")
config = AnalysisConfig(provider="llm")
pipeline = DocumentAnalysisPipeline(config=config, llm=llm)
result = pipeline.run("data/sample/document.pdf")
```

### Use individual components

```python
from src.processing.pdf_extractor import PDFExtractor
from src.processing.chunking import SentenceChunker
from src.inference.summarizer import Summarizer, SummaryConfig
from src.inference.question_generator import QuestionGenerator
from src.inference.qa_engine import QAEngine

# Extract
extractor = PDFExtractor()
text = extractor.extract("document.pdf")
print(extractor.preview("document.pdf", chars=500))

# Chunk
chunker = SentenceChunker()
passages = chunker.split(text)

# Summarise (HF)
summarizer = Summarizer()
summary = summarizer.summarize_passages(passages)

# Generate questions (HF)
qg = QuestionGenerator()
questions = qg.generate(passages[0], min_questions=3)

# Answer questions (HF)
qa = QAEngine()
for q in questions:
    result = qa.answer(q, passages[0])
    print(f"Q: {result.question}")
    print(f"A: {result.answer}  [{result.score:.3f}]")
```

---

## Output Format

Results are saved as `data/output/<filename>_analysis.json`:

```json
{
  "source": "data/sample/google_tos.pdf",
  "text_preview": "GOOGLE TERMS OF SERVICE ...",
  "summary": "These Terms of Service govern the relationship between ...",
  "num_passages": 12,
  "all_qa_pairs": [
    {
      "question": "What do the Terms of Service help define?",
      "answer": "Google's relationship with you as you interact with our services",
      "score": 0.9821
    }
  ],
  "passages": [
    {
      "passage_index": 0,
      "passage": "GOOGLE TERMS OF SERVICE ...",
      "questions": ["What is covered in these terms?", "..."],
      "qa_pairs": [{"question": "...", "answer": "...", "score": 0.98}]
    }
  ]
}
```

---

## Configuration

Edit `config/model_config.yaml` to change defaults:

```yaml
document_analysis:
  provider: huggingface          # or "llm"
  summarization_model: google/t5-small
  question_generation_model: valhalla/t5-base-qg-hl
  qa_model: deepset/roberta-base-squad2
  passage_word_limit: 200
  min_questions_per_passage: 3
  summary_max_length: 150
  summary_min_length: 30
  output_dir: data/output
  cache_dir: data/cache
```

---

## Running Tests

```bash
# Unit tests only (no API keys required)
bash scripts/run_tests.sh

# With integration tests (requires env vars)
bash scripts/run_tests.sh --integration

# Full coverage report
bash scripts/run_tests.sh --all
```

---

## HuggingFace Models Used

| Step | Model | Purpose |
|------|-------|---------|
| Summarisation | [`google/t5-small`](https://huggingface.co/google-t5/t5-small) | Abstractive text summarisation |
| Question generation | [`valhalla/t5-base-qg-hl`](https://huggingface.co/valhalla/t5-base-qg-hl) | T5 fine-tuned for question generation |
| Question answering | [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2) | Extractive QA (SQuAD 2.0 fine-tuned) |

All models are downloaded automatically on first use and cached in `~/.cache/huggingface/hub/`.

---

## Docker

```bash
# Build and run
docker compose up --build

# Run analysis inside the container
docker compose run --rm app python main.py --pdf data/sample/document.pdf
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Dual backend (`huggingface` / `llm`) | Runs locally without API keys; scales up to production-quality LLMs with a single flag |
| PDF text caching | SHA-256 keyed cache avoids redundant extraction on repeated runs |
| `SentenceChunker` (NLTK) | Preserves sentence boundaries; prevents splitting mid-sentence which degrades QA quality |
| `deduplicate=True` in `QAEngine` | Mirrors the article's `answered_questions` set; avoids redundant LLM/model calls |
| `AnalysisResult.to_json()` | Structured output enables downstream automation and evaluation |
| Component separation | Each step is independently testable and swappable |
