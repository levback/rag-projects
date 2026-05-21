"""Knowledge graph store — NetworkX-backed directed graph of (head, relation, tail) triples."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Triple:
    """An (entity, relation, entity) fact extracted from a document."""

    head: str
    relation: str
    tail: str

    def __str__(self) -> str:  # noqa: D401
        return f"({self.head}) --[{self.relation}]--> ({self.tail})"


@dataclass
class GraphSearchResult:
    """Result of a multi-hop graph search."""

    query_entity: str
    paths: list[list[Triple]]
    context: str          # plain-text summary of all found triples
    total_triples: int


class KnowledgeGraphStore:
    """Directed graph of knowledge triples using NetworkX.

    Each node is an entity string. Each directed edge carries a *relation* label.
    The graph supports:
    - ``add_triple`` / ``add_triples``
    - ``search(entity, max_depth)`` — DFS multi-hop traversal
    - ``get_context(query)`` — returns plain-text triples relevant to *query*
    - ``save`` / ``load`` — JSON persistence

    Args:
        name: Human-readable graph name (used in serialisation).
    """

    def __init__(self, name: str = "knowledge_graph") -> None:
        import networkx as nx  # lazy import

        self._graph: Any = nx.DiGraph()
        self.name = name
        self._triples: list[Triple] = []

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def add_triple(self, triple: Triple) -> None:
        """Add a single *triple* to the graph."""
        self._graph.add_edge(triple.head, triple.tail, relation=triple.relation)
        self._triples.append(triple)

    def add_triples(self, triples: list[Triple]) -> None:
        """Add multiple *triples* at once."""
        for triple in triples:
            self.add_triple(triple)

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(
        self,
        entity: str,
        max_depth: int = 2,
    ) -> GraphSearchResult:
        """DFS multi-hop traversal starting from *entity*.

        Args:
            entity: Starting node. Case-insensitive prefix matching is used
                    to find the closest node if the exact name is missing.
            max_depth: Maximum hops to traverse.

        Returns:
            :class:`GraphSearchResult` with all discovered triples as paths.
        """
        start_node = self._find_node(entity)
        if start_node is None:
            logger.debug("Entity %r not found in graph", entity)
            return GraphSearchResult(
                query_entity=entity,
                paths=[],
                context="No information found for this entity.",
                total_triples=0,
            )

        paths = self._dfs(start_node, max_depth)
        context = self._build_context(paths)
        return GraphSearchResult(
            query_entity=entity,
            paths=paths,
            context=context,
            total_triples=sum(len(p) for p in paths),
        )

    def get_context(self, query: str, max_depth: int = 2) -> str:
        """Return plain-text graph context for *query*.

        Tokenises *query* and searches for each token as a graph entity,
        then merges the results.
        """
        tokens = [t.strip(".,!?;:\"'()[]{}") for t in query.split() if len(t.strip()) > 3]
        seen: set[str] = set()
        context_parts: list[str] = []
        for token in tokens:
            result = self.search(token, max_depth=max_depth)
            snippet = result.context
            if snippet not in seen and snippet != "No information found for this entity.":
                seen.add(snippet)
                context_parts.append(snippet)
        return "\n\n".join(context_parts) if context_parts else "No relevant graph context found."

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def triples(self) -> list[Triple]:
        return list(self._triples)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise all triples to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "triples": [
                {"head": t.head, "relation": t.relation, "tail": t.tail}
                for t in self._triples
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved knowledge graph to %s (%d triples)", path, len(self._triples))

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraphStore":
        """Load a :class:`KnowledgeGraphStore` from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls(name=data.get("name", "knowledge_graph"))
        triples = [
            Triple(head=t["head"], relation=t["relation"], tail=t["tail"])
            for t in data.get("triples", [])
        ]
        store.add_triples(triples)
        logger.info("Loaded knowledge graph from %s (%d triples)", path, len(triples))
        return store

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_node(self, entity: str) -> str | None:
        """Find the best-matching node (exact first, then case-insensitive prefix)."""
        if entity in self._graph:
            return entity
        entity_lower = entity.lower()
        for node in self._graph.nodes():
            if node.lower().startswith(entity_lower) or entity_lower in node.lower():
                return node
        return None

    def _dfs(
        self,
        start: str,
        max_depth: int,
        visited: frozenset[str] | None = None,
        depth: int = 0,
    ) -> list[list[Triple]]:
        """Depth-first search returning all edge paths up to *max_depth*."""
        if depth >= max_depth:
            return []
        visited = (visited or frozenset()) | {start}
        all_paths: list[list[Triple]] = []

        for _head, tail, data in self._graph.out_edges(start, data=True):
            relation = data.get("relation", "related_to")
            triple = Triple(head=start, relation=relation, tail=tail)
            all_paths.append([triple])
            if tail not in visited:
                for sub_path in self._dfs(tail, max_depth, visited, depth + 1):
                    all_paths.append([triple] + sub_path)
        return all_paths

    def _build_context(self, paths: list[list[Triple]]) -> str:
        if not paths:
            return "No information found for this entity."
        seen: set[str] = set()
        lines: list[str] = []
        for path in paths:
            for triple in path:
                rep = str(triple)
                if rep not in seen:
                    seen.add(rep)
                    lines.append(rep)
        return "\n".join(lines)
