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
_VSA_WEIGHT = 0.4
_KW_WEIGHT = 0.3
_CONCEPT_WEIGHT = 0.3

# Common concepts → domain mapping (fast semantic routing)
_CONCEPT_DOMAIN: dict[str, str] = {
    # Physics
    "force": "physics", "velocity": "physics", "acceleration": "physics",
    "momentum": "physics", "energy": "physics", "kinetic": "physics",
    "potential": "physics", "gravity": "physics", "newton": "physics",
    "mass": "physics", "weight": "physics", "friction": "physics",
    "wave": "physics", "frequency": "physics", "wavelength": "physics",
    "pressure": "physics", "fluid": "physics", "bernoulli": "physics",
    "ohm": "physics", "voltage": "physics", "current": "physics",
    "resistance": "physics", "magnetic": "physics", "electric": "physics",
    "faraday": "physics", "maxwell": "physics", "coulomb": "physics",
    "relativity": "physics", "einstein": "physics", "lorentz": "physics",
    "thermodynamics": "physics", "entropy": "physics", "carnot": "physics",
    "temperature": "physics", "heat": "physics",
    # Atomic / Quantum
    "quantum": "atomic", "superposition": "atomic", "entanglement": "atomic",
    "schrodinger": "atomic", "heisenberg": "atomic", "uncertainty": "atomic",
    "electron": "atomic", "photon": "atomic", "atom": "atomic",
    "nucleus": "atomic", "orbital": "atomic", "spin": "atomic",
    "fermion": "atomic", "boson": "atomic", "pauli": "atomic",
    "planck": "atomic", "wavefunction": "atomic", "collapse": "atomic",
    "qubit": "atomic", "decoherence": "atomic",
    # Logic
    "modus": "logic", "ponens": "logic", "syllogism": "logic",
    "deductive": "logic", "inductive": "logic", "proposition": "logic",
    "implication": "logic", "contradiction": "logic", "tautology": "logic",
    "obligation": "logic", "permission": "logic", "deontic": "logic",
    "modal": "logic", "boolean": "logic", "truth": "logic",
    # Economics
    "supply": "economics", "demand": "economics", "equilibrium": "economics",
    "inflation": "economics", "gdp": "economics", "market": "economics",
    "price": "economics", "cost": "economics", "profit": "economics",
    "nash": "economics", "utility": "economics", "elasticity": "economics",
    "fiscal": "economics", "monetary": "economics", "interest": "economics",
    "stock": "economics", "bond": "economics", "currency": "economics",
    # Bio
    "dna": "bio", "gene": "bio", "evolution": "bio", "mutation": "bio",
    "selection": "bio", "fitness": "bio", "species": "bio",
    "ecology": "bio", "population": "bio", "predator": "bio",
    "prey": "bio", "enzyme": "bio", "protein": "bio",
    "heredity": "bio", "allele": "bio", "chromosome": "bio",
    "altruism": "bio", "kin": "bio", "hamilton": "bio",
    # Music
    "harmonic": "music", "chord": "music", "melody": "music",
    "rhythm": "music", "tempo": "music", "octave": "music",
    "consonance": "music", "dissonance": "music", "counterpoint": "music",
    "fugue": "music", "bach": "music", "phasor": "music",
    # Causal
    "causal": "causal", "causation": "causal", "counterfactual": "causal",
    "intervention": "causal", "bayes": "causal", "bayesian": "causal",
    "probability": "causal", "conditional": "causal", "granger": "causal",
    # Historical
    "civilization": "historical", "empire": "historical", "war": "historical",
    "revolution": "historical", "dynasty": "historical", "era": "historical",
    "medieval": "historical", "renaissance": "historical", "ancient": "historical",
    # Identity
    "personality": "identity", "emotion": "identity", "belief": "identity",
    "value": "identity", "family": "identity", "identity": "identity",
    # Security
    "encryption": "security", "cipher": "security", "hash": "security",
    "authentication": "security", "firewall": "security", "vulnerability": "security",
    "cyber": "security", "malware": "security", "phishing": "security",
    # Social
    "network": "social", "pagerank": "social", "centrality": "social",
    "diffusion": "social", "homophily": "social", "clustering": "social",
    # Colors
    "color": "colors", "wavelength": "colors", "rgb": "colors",
    "hue": "colors", "saturation": "colors", "spectrum": "colors",
}


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
        codebook_path: str | None = None,
        **kwargs,
    ) -> None:
        self._lib = library
        self._vsa_weight = vsa_weight
        self._kw_weight = kw_weight

        # Build per-domain centroids and keyword sets from current library contents
        self._centroids: dict[str, np.ndarray] = {}
        self._domain_keywords: dict[str, set[str]] = {}
        self._build_index()

        # Load NV-Embed codebook centroids if available (much richer domain knowledge)
        self._nvembed_centroids: dict[str, np.ndarray] = {}
        if codebook_path:
            self._load_codebook_centroids(codebook_path)

    def _load_codebook_centroids(self, path: str) -> None:
        """Load domain centroids from NV-Embed codebook."""
        try:
            cb = np.load(path, allow_pickle=True)
            centroids = cb["centroids"]
            names = list(cb["domain_names"])
            for i, name in enumerate(names):
                self._nvembed_centroids[name] = centroids[i]
        except Exception:
            pass

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

    def _concept_scores(self, question: str) -> dict[str, float]:
        """Score domains by concept mapping — fast semantic routing."""
        q_tokens = _tokens(question)
        domain_hits: dict[str, int] = {}
        for token in q_tokens:
            domain = _CONCEPT_DOMAIN.get(token)
            if domain:
                domain_hits[domain] = domain_hits.get(domain, 0) + 1
        total_hits = sum(domain_hits.values()) or 1
        return {d: hits / total_hits for d, hits in domain_hits.items()}

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
        concept = self._concept_scores(question) if question else {}

        all_domains = set(vsa) | set(kw) | set(concept)
        blended: list[tuple[str, float]] = []
        for domain in all_domains:
            score = (
                self._vsa_weight * vsa.get(domain, 0.0)
                + self._kw_weight * kw.get(domain, 0.0)
                + _CONCEPT_WEIGHT * concept.get(domain, 0.0)
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
