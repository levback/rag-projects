"""LLM-based triple extractor for building knowledge graphs."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.knowledge_graph.graph_store import Triple

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract factual (head entity, relation, tail entity) triples from the text below.
Return ONLY a valid JSON array, with no extra commentary.
Each element must have exactly three string fields: "head", "relation", "tail".

Rules:
- Use short, canonical entity names (e.g. "Albert Einstein", "General Relativity").
- Use concise snake_case relation labels (e.g. "developed", "is_part_of", "born_in").
- Ignore pronouns; resolve co-references where obvious.
- Return at most 20 triples per chunk.

TEXT:
{text}

JSON:"""


@dataclass
class ExtractionResult:
    """Result of running triple extraction on a text chunk."""

    triples: list[Triple]
    raw_response: str
    chunk_index: int = 0


class TripleExtractor:
    """Extract (head, relation, tail) triples from text using an LLM.

    The extractor splits long texts into chunks and runs the LLM prompt
    on each chunk independently, merging duplicates afterwards.

    Args:
        llm: Any :class:`~src.core.base_llm.BaseLLM` instance.
            Bedrock models (e.g. ``anthropic.claude-3-haiku-20240307-v1:0``)
            work well for structured extraction tasks.
        chunk_size: Maximum characters per LLM call.
        deduplicate: Whether to remove identical triples.
    """

    def __init__(
        self,
        llm: "BaseLLM",  # noqa: F821
        chunk_size: int = 2000,
        deduplicate: bool = True,
    ) -> None:
        self._llm = llm
        self._chunk_size = chunk_size
        self._deduplicate = deduplicate

    def extract(self, text: str) -> list[Triple]:
        """Extract all triples from *text*, chunking if necessary.

        Args:
            text: Raw text to extract knowledge from.

        Returns:
            Deduplicated list of :class:`Triple` objects.
        """
        chunks = self._split(text)
        all_triples: list[Triple] = []
        for i, chunk in enumerate(chunks):
            result = self.extract_chunk(chunk, chunk_index=i)
            all_triples.extend(result.triples)
            logger.debug("Chunk %d: %d triples", i, len(result.triples))

        if self._deduplicate:
            seen: set[tuple[str, str, str]] = set()
            deduped: list[Triple] = []
            for t in all_triples:
                key = (t.head.lower(), t.relation.lower(), t.tail.lower())
                if key not in seen:
                    seen.add(key)
                    deduped.append(t)
            return deduped
        return all_triples

    def extract_chunk(self, text: str, chunk_index: int = 0) -> ExtractionResult:
        """Run extraction on a single text chunk."""
        prompt = _EXTRACTION_PROMPT.format(text=text)
        try:
            raw = self._llm.complete(prompt)
        except Exception as exc:
            logger.warning("LLM extraction failed for chunk %d: %s", chunk_index, exc)
            return ExtractionResult(triples=[], raw_response="", chunk_index=chunk_index)

        triples = self._parse_response(raw)
        return ExtractionResult(triples=triples, raw_response=raw, chunk_index=chunk_index)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """Split *text* into chunks of at most *chunk_size* characters at sentence boundaries."""
        if len(text) <= self._chunk_size:
            return [text]
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current_len + len(sentence) > self._chunk_size and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _parse_response(self, raw: str) -> list[Triple]:
        """Parse LLM JSON output into :class:`Triple` objects."""
        # Extract the JSON array even if the LLM adds leading/trailing text
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not json_match:
            logger.debug("No JSON array found in LLM response")
            return []
        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.debug("JSON parse error: %s", exc)
            return []

        triples: list[Triple] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()
            if head and relation and tail:
                triples.append(Triple(head=head, relation=relation, tail=tail))
        return triples
