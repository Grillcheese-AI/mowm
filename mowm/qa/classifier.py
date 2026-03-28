# mowm/qa/classifier.py
"""DomainClassifier — blend VSA cosine similarity with keyword matching.

Routes questions to domains using:
  - 60% weight: VSA cosine similarity (encode question → compare against domain centroids)
  - 40% weight: keyword matching (question tokens vs axiom names and formula tokens)

Centroids are computed at init from AxiomLibrary by averaging axiom vectors per domain
then L2-normalising. Keywords are extracted from axiom names and formula strings.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary


_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")

# Weight split
_VSA_WEIGHT = 0.6
_KW_WEIGHT = 0.4


def _tokens(text: str) -> set[str]:
    """Return lowercased tokens from a string."""
    return {m.lower() for m in _TOKEN_RE.findall(text)}


class DomainClassifier:
    """Classify a question into a ranked list of (domain, score) pairs.

    Args:
        library: AxiomLibrary instance (may be empty — returns empty results).
        vsa_weight: Weight for VSA cosine similarity (default 0.6).
        kw_weight: Weight for keyword matching (default 0.4).
    """

    def __init__(
        self,
        library: "AxiomLibrary",
        vsa_weight: float = _VSA_WEIGHT,
        kw_weight: float = _KW_WEIGHT,
    ) -> None:
        self._lib = library
        self._vsa_weight = vsa_weight
        self._kw_weight = kw_weight

        # Build per-domain centroids and keyword sets from current library contents
        self._centroids: dict[str, np.ndarray] = {}
        self._domain_keywords: dict[str, set[str]] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build centroids and keyword index from AxiomLibrary."""
        domain_vecs: dict[str, list[np.ndarray]] = {}
        domain_kws: dict[str, set[str]] = {}

        for axiom in self._lib.all_axioms():
            top = axiom.domain.split(".")[0]
            domain_vecs.setdefault(top, []).append(axiom.vector)
            kws = domain_kws.setdefault(top, set())
            kws.update(_tokens(axiom.name))
            kws.update(_tokens(axiom.formula_str))

        for domain, vecs in domain_vecs.items():
            centroid = np.mean(np.stack(vecs, axis=0), axis=0)  # shape (k, l)
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                centroid = centroid / norm
            self._centroids[domain] = centroid.astype(np.float32)

        self._domain_keywords = domain_kws

    # ------------------------------------------------------------------
    # VSA helpers
    # ------------------------------------------------------------------

    def _encode_question(self, question: str) -> np.ndarray:
        """Encode a question string into a VSA vector via WorldEncoder."""
        return self._lib.encoder.encode_action(question)

    def _vsa_scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Cosine similarity between *embedding* and each domain centroid."""
        if not self._centroids:
            return {}
        # Flatten for dot product — centroids are already L2-normalised per (k,l)
        # Use BlockCodes.similarity_batch for consistency with the rest of the codebase.
        codebook = np.stack(list(self._centroids.values()), axis=0)  # (D, k, l)
        sims = self._lib.bc.similarity_batch(embedding, codebook)
        # similarity_batch returns values in [-1, 1]; shift to [0, 1]
        sims_shifted = (sims + 1.0) / 2.0
        return {domain: float(sims_shifted[i]) for i, domain in enumerate(self._centroids)}

    # ------------------------------------------------------------------
    # Keyword helpers
    # ------------------------------------------------------------------

    def _kw_scores(self, question: str) -> dict[str, float]:
        """Compute keyword overlap score for each domain."""
        if not self._domain_keywords:
            return {}
        q_tokens = _tokens(question)
        scores: dict[str, float] = {}
        for domain, kws in self._domain_keywords.items():
            if not kws:
                scores[domain] = 0.0
                continue
            overlap = len(q_tokens & kws)
            scores[domain] = overlap / len(kws)
        return scores

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, question: str) -> list[tuple[str, float]]:
        """Classify *question* into a ranked list of (domain, score) pairs.

        Score is a weighted blend:  0.6 * vsa_score + 0.4 * kw_score

        Args:
            question: Natural-language question string.

        Returns:
            List of (domain, score) tuples, highest score first.
            Empty list when the library contains no axioms.
        """
        if not self._centroids:
            return []

        embedding = self._encode_question(question)
        return self.classify_embedding(embedding, question=question)

    def classify_embedding(
        self,
        embedding: np.ndarray,
        question: str = "",
    ) -> list[tuple[str, float]]:
        """Classify using a pre-computed VSA *embedding*.

        Args:
            embedding: Block-code vector of shape (k, l).
            question: Optional question text used for keyword matching.
                      Pass an empty string to skip keyword matching (kw_score=0).

        Returns:
            List of (domain, score) tuples, highest score first.
        """
        if not self._centroids:
            return []

        vsa = self._vsa_scores(embedding)
        kw = self._kw_scores(question) if question else {}

        all_domains = set(vsa) | set(kw)
        blended: list[tuple[str, float]] = []
        for domain in all_domains:
            score = (
                self._vsa_weight * vsa.get(domain, 0.0)
                + self._kw_weight * kw.get(domain, 0.0)
            )
            blended.append((domain, score))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended

    def top_domain(self, question: str) -> str:
        """Return the highest-scoring domain for *question*.

        Args:
            question: Natural-language question string.

        Returns:
            Domain name string, or "" when the library is empty.
        """
        ranked = self.classify(question)
        if not ranked:
            return ""
        return ranked[0][0]
