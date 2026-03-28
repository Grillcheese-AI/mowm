# mowm/reasoning/knowledge_search.py
"""KnowledgeSearch -- multi-hop axiom discovery for reasoning decoder.

Finds relevant axioms by combining:
  1. Keyword matching (question tokens vs axiom name + formula tokens)
  2. VSA similarity (encode keywords -> compare against axiom vectors)
  3. Variable graph traversal (shared variables as edges between axioms)
  4. Domain-scoped filtering with cross-domain fallback
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")

# Constants
_MAX_PRIMARY = 5
_MAX_SUPPORTING = 10
_MAX_PATH_DEPTH = 6


@dataclass
class KnowledgeContext:
    """Structured knowledge retrieved for a question.

    Attributes:
        primary_axioms: Most directly relevant axioms (top 3-5).
        supporting_axioms: Related by domain/similarity (next 5-10).
        concepts: Extracted concept names from matched axioms.
        domain_chain: Sub-domain paths found (e.g. ["physics.dynamics"]).
        variables_mentioned: All variables in relevant formulas.
    """

    primary_axioms: list[Axiom] = field(default_factory=list)
    supporting_axioms: list[Axiom] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    domain_chain: list[str] = field(default_factory=list)
    variables_mentioned: set[str] = field(default_factory=set)


def _tokens(text: str) -> set[str]:
    """Return lowercased tokens from a string."""
    return {m.lower() for m in _TOKEN_RE.findall(text)}


def _extract_formula_vars(formula: str) -> set[str]:
    """Extract variable identifiers from a formula string.

    Filters out known function names and pure numeric tokens.
    """
    _FUNCTIONS = {
        "sin", "cos", "sqrt", "exp", "log", "square", "not",
        "and", "or", "implies", "for", "all", "if", "iff",
        "such", "that", "with", "subject", "to", "max", "min",
        "sum", "prod", "integral", "no", "in", "const",
        "avoid", "consecutive", "between", "voices", "pairs",
    }
    raw = set(_TOKEN_RE.findall(formula))
    return {v for v in raw if v.lower() not in _FUNCTIONS and not v.isdigit()}


class KnowledgeSearch:
    """Multi-hop knowledge search through the axiom library.

    Combines keyword overlap, VSA similarity, and variable-graph
    traversal to find primary and supporting axioms for a question.

    Args:
        library: AxiomLibrary instance.
        k: Number of VSA blocks.
        l: Block length.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        k: int = 80,
        l: int = 128,  # noqa: E741
    ) -> None:
        self._lib = library
        self._k = k
        self._l = l
        # Pre-compute variable sets per axiom for graph traversal
        self._axiom_vars: dict[str, set[str]] = {}
        for ax in self._lib.all_axioms():
            self._axiom_vars[ax.name] = _extract_formula_vars(ax.formula_str)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        keywords: list[str],
        domain: str,
        target_var: str | None = None,
    ) -> KnowledgeContext:
        """Multi-hop knowledge search through the axiom library.

        1. Direct keyword match -> primary axioms
        2. VSA similarity from primary axiom vectors -> supporting axioms
        3. Domain chain: follow sub-domain connections
        4. Variable tracing: find all axioms sharing variables with primaries

        Args:
            keywords: Question keywords (lowercased tokens).
            domain: Top-level domain string (e.g. "physics").
            target_var: Optional target variable to prioritize.

        Returns:
            KnowledgeContext with primary/supporting axioms, concepts,
            domain chain, and variables mentioned.
        """
        all_axioms = self._lib.all_axioms()
        if not all_axioms:
            return KnowledgeContext()

        kw_set = {kw.lower() for kw in keywords}

        # Step 1: score every axiom by keyword overlap + domain bonus
        scored = self._score_axioms(all_axioms, kw_set, domain, target_var)

        # Step 2: split into primary and initial supporting
        primary = [ax for ax, _ in scored[:_MAX_PRIMARY]]
        supporting_candidates = [ax for ax, _ in scored[_MAX_PRIMARY:]]

        # Step 3: variable-based expansion -- find axioms sharing vars
        primary_vars: set[str] = set()
        for ax in primary:
            primary_vars.update(self._axiom_vars.get(ax.name, set()))

        var_related = self._find_by_shared_vars(
            primary_vars, primary, all_axioms,
        )
        supporting_candidates = var_related + supporting_candidates

        # Step 4: VSA similarity expansion from primary vectors
        if primary:
            vsa_related = self._vsa_expand(primary, all_axioms)
            supporting_candidates.extend(vsa_related)

        # Deduplicate supporting, exclude already-primary axioms
        primary_names = {a.name for a in primary}
        seen: set[str] = set()
        supporting: list = []
        for ax in supporting_candidates:
            if ax.name not in primary_names and ax.name not in seen:
                seen.add(ax.name)
                supporting.append(ax)
        supporting = supporting[:_MAX_SUPPORTING]

        # Build context
        all_relevant = primary + supporting
        variables = set()
        for ax in all_relevant:
            variables.update(self._axiom_vars.get(ax.name, set()))

        domain_chain = sorted({ax.domain for ax in all_relevant})
        concepts = self._extract_concepts(all_relevant)

        return KnowledgeContext(
            primary_axioms=primary,
            supporting_axioms=supporting,
            concepts=concepts,
            domain_chain=domain_chain,
            variables_mentioned=variables,
        )

    def find_derivation_path(
        self,
        start_axiom: Axiom,
        target_concept: str,
    ) -> list[Axiom]:
        """Find a chain of axioms from start to target concept.

        Uses shared variables as edges: if axiom A produces variable X
        and axiom B consumes variable X, there's a derivation link.

        BFS through the variable graph, max depth _MAX_PATH_DEPTH.

        Args:
            start_axiom: Starting axiom.
            target_concept: Variable name or keyword to reach.

        Returns:
            List of axioms forming the path (may be empty).
        """
        all_axioms = self._lib.all_axioms()
        target_lower = target_concept.lower()

        # Check if target is already in start axiom
        start_vars = self._axiom_vars.get(start_axiom.name, set())  # noqa: F841

        # BFS: queue of (current_axiom, path_so_far)
        from collections import deque

        queue: deque[tuple[Axiom, list[Axiom]]] = deque()
        queue.append((start_axiom, [start_axiom]))
        visited: set[str] = {start_axiom.name}

        while queue:
            current, path = queue.popleft()
            if len(path) > _MAX_PATH_DEPTH:
                continue

            current_vars = self._axiom_vars.get(current.name, set())

            # Find neighbors: axioms sharing at least one variable
            for ax in all_axioms:
                if ax.name in visited:
                    continue
                ax_vars = self._axiom_vars.get(ax.name, set())
                shared = current_vars & ax_vars
                if not shared:
                    continue

                new_path = path + [ax]

                # Check if target is in this axiom's variables or name
                ax_all_tokens = _tokens(ax.name) | {
                    v.lower() for v in ax_vars
                }
                if target_lower in ax_all_tokens:
                    return new_path[1:]  # exclude start_axiom

                visited.add(ax.name)
                queue.append((ax, new_path))

        return []

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_axioms(
        self,
        axioms: list[Axiom],
        kw_set: set[str],
        domain: str,
        target_var: str | None,
    ) -> list[tuple[Axiom, float]]:
        """Score axioms by keyword overlap + domain bonus + target bonus."""
        scored: list[tuple[Axiom, float]] = []
        for ax in axioms:
            ax_tokens = _tokens(ax.name) | _tokens(ax.formula_str)
            overlap = len(kw_set & ax_tokens)

            # Domain bonus
            domain_bonus = 1.0 if ax.domain.split(".")[0] == domain else 0.0

            # Target variable bonus
            target_bonus = 0.0
            if target_var:
                ax_vars = self._axiom_vars.get(ax.name, set())
                if target_var in ax_vars:
                    target_bonus = 3.0

            score = float(overlap) + domain_bonus + target_bonus
            scored.append((ax, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Variable-based expansion
    # ------------------------------------------------------------------

    def _find_by_shared_vars(
        self,
        target_vars: set[str],
        exclude: list[Axiom],
        all_axioms: list[Axiom],
    ) -> list[Axiom]:
        """Find axioms sharing variables with the target set."""
        exclude_names = {a.name for a in exclude}
        results: list[tuple[Axiom, int]] = []
        for ax in all_axioms:
            if ax.name in exclude_names:
                continue
            ax_vars = self._axiom_vars.get(ax.name, set())
            shared_count = len(target_vars & ax_vars)
            if shared_count > 0:
                results.append((ax, shared_count))
        results.sort(key=lambda x: x[1], reverse=True)
        return [ax for ax, _ in results]

    # ------------------------------------------------------------------
    # VSA similarity expansion
    # ------------------------------------------------------------------

    def _vsa_expand(
        self,
        primary: list[Axiom],
        all_axioms: list[Axiom],
    ) -> list[Axiom]:
        """Find axioms similar to primary via VSA cosine similarity."""
        if not primary or not all_axioms:
            return []
        primary_names = {a.name for a in primary}
        # Average primary vectors
        vecs = np.stack([a.vector for a in primary], axis=0)
        centroid = np.mean(vecs, axis=0)
        # Use library select
        candidates = self._lib.select(centroid, top_n=_MAX_SUPPORTING + len(primary))
        return [a for a in candidates if a.name not in primary_names]

    # ------------------------------------------------------------------
    # Concept extraction
    # ------------------------------------------------------------------

    def _extract_concepts(self, axioms: list[Axiom]) -> list[str]:
        """Extract unique concept names from axiom names."""
        seen: set[str] = set()
        concepts: list[str] = []
        for ax in axioms:
            # Use the axiom name as a concept
            if ax.name not in seen:
                seen.add(ax.name)
                concepts.append(ax.name)
        return concepts
