# mowm/causal_graph.py
"""Signed directed causal graph with BFS propagation and VSA encoding."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from cubemind.ops.block_codes import BlockCodes

from mowm.base import BaseCausalGraph


@dataclass
class CausalNode:
    """A node in the causal graph with a VSA block-code vector."""

    name: str
    domain: str
    vector: np.ndarray  # shape (k, l), float32


@dataclass
class CausalEdge:
    """A signed directed edge between two causal nodes."""

    source: str
    target: str
    polarity: str  # "+" or "-"
    strength: float  # [0.0, 1.0]


class CausalGraph(BaseCausalGraph):
    """Signed directed causal graph with BFS propagation and VSA encoding.

    Internally uses pure dict-based adjacency lists — no networkx.

    Args:
        k: Number of VSA blocks.
        l: Block length.
    """

    def __init__(self, k: int = 80, l: int = 128) -> None:  # noqa: E741
        self._k = k
        self._l = l
        self._bc = BlockCodes(k=k, l=l)
        self._nodes: dict[str, CausalNode] = {}
        self._adj: dict[str, list[CausalEdge]] = {}       # source -> outgoing edges
        self._rev_adj: dict[str, list[CausalEdge]] = {}   # target -> incoming edges

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def nodes(self) -> dict[str, CausalNode]:
        """Read-only view of nodes keyed by name."""
        return self._nodes

    @property
    def edges(self) -> list[CausalEdge]:
        """All edges as a flat list."""
        return [edge for edges in self._adj.values() for edge in edges]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._adj.values())

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_node(self, name: str, domain: str, vector: np.ndarray) -> None:
        """Add a node. Silently replaces if name already exists."""
        self._nodes[name] = CausalNode(name=name, domain=domain, vector=vector)
        self._adj.setdefault(name, [])
        self._rev_adj.setdefault(name, [])

    def add_edge(self, source: str, target: str, polarity: str, strength: float) -> None:
        """Add a directed edge. Raises KeyError if either node is missing."""
        if source not in self._nodes:
            raise KeyError(f"Source node '{source}' not in graph")
        if target not in self._nodes:
            raise KeyError(f"Target node '{target}' not in graph")
        edge = CausalEdge(source=source, target=target, polarity=polarity, strength=strength)
        self._adj[source].append(edge)
        self._rev_adj[target].append(edge)

    def remove_node(self, name: str) -> None:
        """Remove a node and all its incident edges."""
        if name not in self._nodes:
            return
        del self._nodes[name]
        # Remove outgoing edges
        outgoing = self._adj.pop(name, [])
        for edge in outgoing:
            self._rev_adj[edge.target] = [
                e for e in self._rev_adj[edge.target] if e.source != name
            ]
        # Remove incoming edges
        incoming = self._rev_adj.pop(name, [])
        for edge in incoming:
            self._adj[edge.source] = [
                e for e in self._adj[edge.source] if e.target != name
            ]

    def remove_edge(self, source: str, target: str) -> None:
        """Remove the edge from source to target (if it exists)."""
        if source in self._adj:
            self._adj[source] = [e for e in self._adj[source] if e.target != target]
        if target in self._rev_adj:
            self._rev_adj[target] = [e for e in self._rev_adj[target] if e.source != source]

    # ── Analysis ─────────────────────────────────────────────────────────────

    def propagate(self, node: str, delta: float) -> dict[str, float]:
        """BFS propagation of a signed delta from a source node.

        Args:
            node: Starting node name.
            delta: Initial causal perturbation magnitude.

        Returns:
            Dict mapping each reached downstream node to its signed effect.

        Raises:
            KeyError: If node is not in the graph.
        """
        if node not in self._nodes:
            raise KeyError(f"Node '{node}' not in graph")

        effects: dict[str, float] = {}
        queue: deque[tuple[str, float]] = deque()
        visited: set[str] = {node}

        # Seed the BFS from node's outgoing edges
        for edge in self._adj[node]:
            sign = 1.0 if edge.polarity == "+" else -1.0
            queue.append((edge.target, delta * sign * edge.strength))

        while queue:
            current, effect = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            effects[current] = effect
            for edge in self._adj.get(current, []):
                if edge.target not in visited:
                    sign = 1.0 if edge.polarity == "+" else -1.0
                    queue.append((edge.target, effect * sign * edge.strength))

        return effects

    def counterfactual(self, node: str, value: float) -> dict[str, float]:
        """Hypothetically set node to value and propagate downstream effects.

        Args:
            node: The intervened node.
            value: The counterfactual value to assign.

        Returns:
            Dict of {node_name: value} including the intervened node itself.

        Raises:
            KeyError: If node is not in the graph.
        """
        if node not in self._nodes:
            raise KeyError(f"Node '{node}' not in graph")
        result = {node: value}
        downstream = self.propagate(node, value)
        result.update(downstream)
        return result

    def find_paths(self, source: str, target: str) -> list[list[str]]:
        """DFS to find all simple paths from source to target.

        Args:
            source: Starting node name.
            target: Destination node name.

        Returns:
            List of paths, where each path is a list of node names.
        """
        all_paths: list[list[str]] = []

        def _dfs(current: str, path: list[str], visited: set[str]) -> None:
            if current == target:
                all_paths.append(list(path))
                return
            for edge in self._adj.get(current, []):
                nxt = edge.target
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    _dfs(nxt, path, visited)
                    path.pop()
                    visited.discard(nxt)

        if source not in self._nodes or target not in self._nodes:
            return []

        _dfs(source, [source], {source})
        return all_paths

    def feedback_loops(self) -> list[list[str]]:
        """Detect all feedback loops (simple directed cycles) via DFS.

        Returns:
            List of cycles, each a list of node names forming the loop.
            Cycles are normalized so the lexicographically smallest node
            comes first, and duplicates are removed.
        """
        found: set[tuple[str, ...]] = set()
        cycles: list[list[str]] = []

        def _dfs(start: str, current: str, path: list[str], visited: set[str]) -> None:
            for edge in self._adj.get(current, []):
                nxt = edge.target
                if nxt == start and len(path) >= 2:
                    # Found a cycle back to start
                    cycle = list(path)
                    # Normalize: rotate so minimum element is first
                    min_idx = cycle.index(min(cycle))
                    normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                    if normalized not in found:
                        found.add(normalized)
                        cycles.append(list(normalized))
                elif nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    _dfs(start, nxt, path, visited)
                    path.pop()
                    visited.discard(nxt)

        for node in self._nodes:
            _dfs(node, node, [node], {node})

        return cycles

    def encode_to_vsa(self) -> np.ndarray:
        """Encode the graph structure as a VSA block-code vector.

        Each edge is encoded as bind(source_vec, target_vec), then all
        edge encodings are bundled together.

        Returns:
            Block-code vector of shape (k, l), dtype float32.
            Returns the zero vector if the graph has no edges.
        """
        zero = np.zeros((self._k, self._l), dtype=np.float32)

        edge_vecs: list[np.ndarray] = []
        for edge in self.edges:
            src_vec = self._nodes[edge.source].vector
            tgt_vec = self._nodes[edge.target].vector
            bound = self._bc.bind(src_vec, tgt_vec)
            # Scale by strength; negate for inhibitory edges
            sign = 1.0 if edge.polarity == "+" else -1.0
            edge_vecs.append(bound * (sign * edge.strength))

        if not edge_vecs:
            return zero

        bundled = self._bc.bundle(edge_vecs, normalize=True)
        return bundled.astype(np.float32)
