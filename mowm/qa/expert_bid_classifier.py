# mowm/qa/expert_bid_classifier.py
"""ExpertBidClassifier — experts compete to route questions.

Instead of hardcoded concept maps, each domain's axiom set evaluates
the question and returns a bid score. The highest-bidding domain wins.

Bid factors:
  1. Axiom keyword overlap (how many axiom tokens match the question)
  2. VSA centroid similarity (block-code cosine sim)
  3. Formula coverage (can we actually compute with these axioms?)
  4. Variable match (do the question's variables appear in domain formulas?)

No hardcoded routing tables. Domain competence is derived entirely
from the axiom library content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")
_VAR_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*=\s*[-\d]")

# Stopwords to ignore in keyword matching
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "when", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it",
    "its", "how", "why", "where", "there", "here", "up", "about",
    "then", "them", "they", "he", "she", "we", "you", "me", "him",
    "her", "us", "my", "your", "his", "our", "their", "tell",
    "explain", "describe", "define", "show", "find", "calculate",
    "compute", "compare", "what", "give", "know", "think",
})


@dataclass
class ExpertBid:
    """A single expert's bid on a question."""
    domain: str
    score: float
    keyword_hits: int = 0
    vsa_similarity: float = 0.0
    formula_coverage: float = 0.0
    variable_match: int = 0
    top_axiom: str = ""


class ExpertBidClassifier:
    """Domain classification via expert bidding.

    Each domain's axiom set competes to answer the question.
    No hardcoded concept maps — competence is derived from the axioms.

    Args:
        library: AxiomLibrary with registered axioms.
    """

    def __init__(self, library: "AxiomLibrary") -> None:
        self._lib = library
        self._domain_index: dict[str, list["Axiom"]] = {}
        self._domain_tokens: dict[str, set[str]] = {}
        self._domain_centroids: dict[str, np.ndarray] = {}
        self._domain_variables: dict[str, set[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Index axioms by domain for fast bidding."""
        self._domain_name_tokens: dict[str, set[str]] = {}

        for axiom in self._lib.all_axioms():
            top = axiom.domain.split(".")[0]
            self._domain_index.setdefault(top, []).append(axiom)

            # All tokens (name + formula)
            tokens = self._domain_tokens.setdefault(top, set())
            all_toks = {
                t.lower() for t in _TOKEN_RE.findall(
                    axiom.name + " " + axiom.formula_str,
                ) if t.lower() not in _STOPWORDS and len(t) > 1
            }
            tokens.update(all_toks)

            # Name-only tokens (higher value — these are the concept names)
            name_tokens = self._domain_name_tokens.setdefault(top, set())
            name_tokens.update(
                t.lower() for t in _TOKEN_RE.findall(axiom.name)
                if t.lower() not in _STOPWORDS and len(t) > 1
            )

            variables = self._domain_variables.setdefault(top, set())
            variables.update(
                t for t in _TOKEN_RE.findall(axiom.formula_str)
                if len(t) <= 5 and t[0].islower()
            )

        # Compute centroids
        for domain, axioms in self._domain_index.items():
            vecs = np.stack([a.vector for a in axioms], axis=0)
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                centroid = centroid / norm
            self._domain_centroids[domain] = centroid.astype(np.float32)

    def bid(self, question: str) -> list[ExpertBid]:
        """Each domain bids on the question. Returns ranked bids.

        Args:
            question: Natural language question string.

        Returns:
            List of ExpertBid sorted by descending score.
        """
        if not self._domain_index:
            return []

        q_tokens = {
            t.lower() for t in _TOKEN_RE.findall(question)
            if t.lower() not in _STOPWORDS and len(t) > 1
        }

        # Extract variable names from question (e.g., m=10 -> "m")
        q_vars = {m.group(1) for m in _VAR_RE.finditer(question)}

        # Encode question for VSA similarity
        q_vec = self._lib.encoder.encode_action(question)

        bids: list[ExpertBid] = []
        for domain, axioms in self._domain_index.items():
            bid = self._compute_bid(
                domain, axioms, q_tokens, q_vars, q_vec,
            )
            bids.append(bid)

        bids.sort(key=lambda b: b.score, reverse=True)
        return bids

    def _compute_bid(
        self,
        domain: str,
        axioms: list["Axiom"],
        q_tokens: set[str],
        q_vars: set[str],
        q_vec: np.ndarray,
    ) -> ExpertBid:
        """Compute a single domain's bid score."""
        domain_tokens = self._domain_tokens.get(domain, set())
        domain_vars = self._domain_variables.get(domain, set())

        # Factor 1: Keyword overlap — name matches worth 3x formula matches
        name_tokens = self._domain_name_tokens.get(domain, set())
        name_hits = len(q_tokens & name_tokens)
        formula_only_hits = len(q_tokens & (domain_tokens - name_tokens))
        keyword_hits = name_hits + formula_only_hits
        weighted_hits = name_hits * 3.0 + formula_only_hits * 1.0
        kw_score = weighted_hits / max(len(q_tokens) * 3.0, 1)

        # Factor 2: VSA centroid similarity
        centroid = self._domain_centroids.get(domain)
        vsa_sim = 0.0
        if centroid is not None:
            raw_sim = self._lib.bc.similarity(q_vec, centroid)
            vsa_sim = max(0.0, (raw_sim + 1.0) / 2.0)

        # Factor 3: Formula coverage — can this domain compute something?
        # Check if any axiom has a formula with "=" (computable)
        computable = sum(
            1 for ax in axioms
            if "=" in ax.formula_str and "==" not in ax.formula_str
        )
        formula_coverage = min(1.0, computable / max(len(axioms), 1))

        # Factor 4: Variable match — do question variables appear in domain?
        var_match = len(q_vars & domain_vars) if q_vars else 0
        var_score = var_match / max(len(q_vars), 1) if q_vars else 0.0

        # Find top matching axiom for evidence
        best_axiom = ""
        best_axiom_score = 0
        for ax in axioms:
            ax_tokens = {
                t.lower() for t in _TOKEN_RE.findall(
                    ax.name + " " + ax.formula_str,
                )
            }
            overlap = len(q_tokens & ax_tokens)
            if overlap > best_axiom_score:
                best_axiom_score = overlap
                best_axiom = ax.name

        # Weighted combination
        score = (
            0.35 * kw_score
            + 0.25 * vsa_sim
            + 0.20 * formula_coverage
            + 0.20 * var_score
        )

        return ExpertBid(
            domain=domain,
            score=score,
            keyword_hits=keyword_hits,
            vsa_similarity=vsa_sim,
            formula_coverage=formula_coverage,
            variable_match=var_match,
            top_axiom=best_axiom,
        )

    def classify(self, question: str) -> list[tuple[str, float]]:
        """Classify question into ranked (domain, score) pairs.

        Drop-in replacement for DomainClassifier.classify().
        """
        bids = self.bid(question)
        return [(b.domain, b.score) for b in bids]

    def top_domain(self, question: str) -> str:
        """Return the highest-bidding domain.

        Drop-in replacement for DomainClassifier.top_domain().
        """
        bids = self.bid(question)
        if not bids:
            return "general"
        return bids[0].domain

    def explain_routing(self, question: str) -> str:
        """Explain why a domain was chosen — full bid breakdown."""
        bids = self.bid(question)
        if not bids:
            return "No domains available."

        lines = [f"**Question:** {question}\n"]
        lines.append("| Domain | Score | Keywords | VSA Sim | Formulas | Vars | Top Axiom |")
        lines.append("|--------|-------|----------|---------|----------|------|-----------|")
        for b in bids[:8]:
            lines.append(
                f"| {b.domain} | {b.score:.3f} | {b.keyword_hits} "
                f"| {b.vsa_similarity:.3f} | {b.formula_coverage:.2f} "
                f"| {b.variable_match} | {b.top_axiom[:30]} |"
            )

        winner = bids[0]
        lines.append(f"\n**Winner:** `{winner.domain}` (score: {winner.score:.3f})")
        if winner.top_axiom:
            lines.append(f"**Top axiom:** {winner.top_axiom}")
        return "\n".join(lines)
