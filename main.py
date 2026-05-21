"""main.py — CLI entry point for the Document Analysis pipeline.

Usage examples:

    # Run with HuggingFace models (no API key required)
    python main.py --pdf data/sample/document.pdf

    # Run with Amazon Bedrock (default when provider=llm — reads region/profile from config)
    python main.py --pdf data/sample/document.pdf --provider llm

    # Run with Amazon Bedrock, specific model, named AWS profile
    python main.py --pdf data/sample/document.pdf --provider llm --llm-provider bedrock --aws-profile my-profile

    # Run with OpenAI GPT
    python main.py --pdf data/sample/document.pdf --provider llm --llm-provider openai

    # Run with Claude, custom output directory
    python main.py --pdf data/sample/document.pdf --provider llm --llm-provider anthropic --output data/output

    # Only summarise (skip Q&A)
    python main.py --pdf data/sample/document.pdf --skip-qa

    # Show text preview only
    python main.py --pdf data/sample/document.pdf --preview-only
"""
from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from pathlib import Path

import yaml


def _load_logging_config(config_path: str = "config/logging_config.yaml") -> None:
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        Path("logs").mkdir(exist_ok=True)
        logging.config.dictConfig(cfg)
    except FileNotFoundError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyse a PDF document using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--pdf",
        required=True,
        metavar="PATH",
        help="Path to the input PDF file.",
    )
    p.add_argument(
        "--provider",
        choices=["huggingface", "llm"],
        default="huggingface",
        help="Backend for summarisation, QG and QA. Default: huggingface.",
    )
    p.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "local", "bedrock"],
        default=None,
        help="LLM provider when --provider=llm. Default: read from config (bedrock).",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        metavar="MODEL",
        help="Override the default model for the chosen LLM provider.",
    )
    p.add_argument(
        "--output",
        default="data/output",
        metavar="DIR",
        help="Directory to save the JSON results. Default: data/output.",
    )
    p.add_argument(
        "--cache-dir",
        default="data/cache",
        metavar="DIR",
        help="Directory for caching extracted PDF text. Default: data/cache.",
    )
    p.add_argument(
        "--word-limit",
        type=int,
        default=200,
        metavar="N",
        help="Max words per passage. Default: 200.",
    )
    p.add_argument(
        "--min-questions",
        type=int,
        default=3,
        metavar="N",
        help="Minimum questions per passage. Default: 3.",
    )
    p.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        metavar="N",
        help="Characters shown in the text preview step. Default: 500.",
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help="Extract and preview text only; skip all generative steps.",
    )
    p.add_argument(
        "--skip-qa",
        action="store_true",
        help="Run extraction + summarisation only; skip question generation and QA.",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results to disk; print to stdout only.",
    )
    p.add_argument(
        "--aws-region",
        default=None,
        metavar="REGION",
        help="AWS region for Amazon Bedrock (e.g. us-east-1). Overrides AWS_DEFAULT_REGION.",
    )
    p.add_argument(
        "--aws-profile",
        default=None,
        metavar="PROFILE",
        help="Named AWS CLI profile for Amazon Bedrock. Overrides AWS_PROFILE.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    _load_logging_config()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)

    # ── Validate PDF path ─────────────────────────────────────────────────────
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        return 1

    # ── Preview-only shortcut ─────────────────────────────────────────────────
    if args.preview_only:
        from src.processing.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(cache_dir=args.cache_dir)
        preview = extractor.preview(pdf_path, chars=args.preview_chars)
        print(f"\n--- TEXT PREVIEW ({pdf_path.name}) ---\n{preview}\n…")
        return 0

    # ── Build config ──────────────────────────────────────────────────────────
    from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

    # Load model_config.yaml to get defaults for the document_analysis section.
    _da_cfg: dict = {}
    try:
        with open("config/model_config.yaml", encoding="utf-8") as _f:
            _mc = yaml.safe_load(_f) or {}
        _da_cfg = _mc.get("document_analysis", {})
    except FileNotFoundError:
        pass

    # Resolve the effective LLM provider (CLI arg overrides config).
    effective_llm_provider: str = args.llm_provider or _da_cfg.get("llm_provider", "bedrock")
    effective_llm_model: str | None = args.llm_model or _da_cfg.get("llm_model") or None

    config = AnalysisConfig(
        provider=args.provider,
        llm_provider=effective_llm_provider,
        llm_model=effective_llm_model,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        passage_word_limit=args.word_limit,
        min_questions_per_passage=args.min_questions,
    )

    # ── Optionally create an LLM client ───────────────────────────────────────
    llm = None
    if args.provider == "llm":
        from src.core.model_factory import create_llm

        # Build extra kwargs — pass Bedrock auth overrides if provided.
        extra_kwargs: dict = {}
        if effective_llm_provider == "bedrock":
            _bedrock_cfg = (_mc if '_mc' in dir() else {}).get("bedrock", {})
            extra_kwargs["region_name"] = (
                args.aws_region
                or _bedrock_cfg.get("region", "us-east-1")
            )
            if args.aws_profile:
                extra_kwargs["profile_name"] = args.aws_profile
            elif _bedrock_cfg.get("profile"):
                extra_kwargs["profile_name"] = _bedrock_cfg["profile"]

        llm = create_llm(
            effective_llm_provider,
            model=effective_llm_model,
            **extra_kwargs,
        )
        logger.info("Using LLM: %s", llm)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline = DocumentAnalysisPipeline(config=config, llm=llm)

    if args.skip_qa:
        # Partial run: extract + preprocess + summarise only
        from src.processing.pdf_extractor import PDFExtractor
        from src.processing.preprocessing import TextPreprocessor
        from src.processing.chunking import SentenceChunker, SentenceChunkingConfig
        from src.inference.summarizer import Summarizer, SummaryConfig

        extractor = PDFExtractor(cache_dir=args.cache_dir)
        text = extractor.extract(pdf_path)
        clean = TextPreprocessor().process(text)
        passages = SentenceChunker(SentenceChunkingConfig(word_limit=args.word_limit)).split(clean)
        summarizer = Summarizer(
            config=SummaryConfig(provider=args.provider),
            llm=llm,
        )
        summary = summarizer.summarize_passages(passages)
        print(f"\n--- SUMMARY ({pdf_path.name}) ---\n{summary}\n")
        return 0

    result = pipeline.run(pdf_path, preview_chars=args.preview_chars)
    pipeline.print_results(result)

    if not args.no_save:
        out_path = pipeline.save_results(result)
        print(f"Results saved → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
