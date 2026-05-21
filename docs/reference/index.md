# API Reference

Complete reference for every public class, method, and function in `src/`.
Every entry links back to the source file.

Cross-references: [User Guide](../user-guide/index.md) · [Examples](../examples.md) · [Installation](../installation.md)

---

## Packages

| Package | Module file | Reference |
|---------|-------------|-----------|
| `src.core` | LLM abstractions + document pipeline | [core.md](core.md) |
| `src.rag` | 7 RAG pipeline implementations | [rag.md](rag.md) |
| `src.agents` | Agent base + 2 agent implementations | [agents.md](agents.md) |
| `src.knowledge_graph` | Triple extraction + graph store | [knowledge_graph.md](knowledge_graph.md) |
| `src.loaders` | Document loaders + web scraper | [loaders.md](loaders.md) |
| `src.processing` | Chunking, PDF extraction, preprocessing | [processing.md](processing.md) |
| `src.inference` | Summariser, QA engine, question generator | [inference.md](inference.md) |
| `src.prompts` | Prompt templates + LangChain chains | [prompts.md](prompts.md) |

---

## Directory map

```
src/
├── __init__.py
├── core/
│   ├── base_llm.py              → BaseLLM, Message, LLMResponse, LLMConfig
│   ├── gpt_client.py            → GPTClient
│   ├── claude_client.py         → ClaudeClient
│   ├── bedrock_client.py        → BedrockClient
│   ├── local_llm.py             → LocalLLM
│   ├── model_factory.py         → ModelProvider, create_llm()
│   └── document_analysis_pipeline.py → AnalysisConfig, AnalysisResult,
│                                        DocumentAnalysisPipeline
├── rag/
│   ├── basic_rag.py             → BasicRAGConfig, BasicRAGResult, BasicRAGPipeline
│   ├── ibm_rag.py               → IBMRAGConfig, IBMRAGResult, IBMProductionRAG
│   ├── graph_rag.py             → GraphRAGConfig, GraphRAGResult, GraphRAGPipeline
│   ├── multi_doc_rag.py         → MultiDocConfig, MultiDocResult, MultiDocumentRAG
│   ├── agentic_rag.py           → AgenticRAGConfig, IntentType, AgenticRAGResult,
│   │                               AgenticRAGPipeline
│   ├── realtime_rag.py          → RealtimeRAGConfig, RealtimeRAGResult,
│   │                               RealtimeRAGAssistant
│   ├── multimodal_rag.py        → MultimodalRAGConfig, MultimodalRAGResult,
│   │                               MultimodalRAGPipeline
│   ├── embedder.py              → EmbedderConfig, Embedder
│   └── indexer.py               → IndexerConfig, Indexer
├── agents/
│   ├── base_agent.py            → AgentResult, BaseAgent
│   ├── research_agent.py        → ResearchAgentConfig, ResearchResult, ResearchAgent
│   └── langchain_rag_agent.py   → LangChainRAGConfig, LangChainRAGResult,
│                                   LangChainRAGAgent
├── knowledge_graph/
│   ├── graph_store.py           → Triple, GraphSearchResult, KnowledgeGraphStore
│   └── triple_extractor.py      → ExtractionResult, TripleExtractor
├── loaders/
│   ├── document_loader.py       → LoadedDocument, DocumentLoader
│   └── web_scraper.py           → ScrapedPage, WebScraper
├── processing/
│   ├── chunking.py              → ChunkingConfig, TextChunker,
│   │                               SentenceChunkingConfig, SentenceChunker
│   ├── pdf_extractor.py         → PDFExtractor
│   ├── preprocessing.py         → normalize_whitespace(), TextPreprocessor
│   └── tokenizer.py             → Tokenizer
├── inference/
│   ├── inference_engine.py      → InferenceRequest, InferenceResult, InferenceEngine
│   ├── summarizer.py            → SummaryConfig, Summarizer
│   ├── qa_engine.py             → QAResult, QAEngine
│   ├── question_generator.py    → QuestionGenerator
│   └── response_parser.py       → ParsedResponse, ResponseParser
└── prompts/
    ├── templates.py             → Prompt string constants
    └── chain.py                 → LangChain prompt builders
```
