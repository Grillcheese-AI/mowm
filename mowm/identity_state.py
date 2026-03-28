# mowm/identity_state.py
"""IdentityState — personal world model with axiom extraction and emotion tracking.

Implements the "funneling conversations" pattern:
  1. Chiseling Phase: extract core beliefs from casual statements
  2. Axiom Binding: map beliefs to block-code identity vectors
  3. Consistency: maintain congruent world view

Tracks:
  - Core beliefs as axioms (e.g. "Family is most important")
  - Emotional baseline (Plutchik 8-emotion model)
  - Identity vector V_Me = bundled belief vectors
  - Alignment scoring for proposed actions
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from cubemind.ops.block_codes import BlockCodes

from mowm.axiom_library import Axiom, AxiomLibrary


PLUTCHIK_EMOTIONS: list[str] = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

# Keyword patterns for extracting beliefs from statements
_VALUE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:most important|highest priority|value most)\b", "identity.values"),
    (r"\b(?:believe in|strongly feel|convinced)\b", "identity.beliefs"),
    (r"\b(?:love|passionate about|enjoy)\b", "identity.passions"),
    (r"\b(?:fear|worried about|anxious)\b", "identity.fears"),
    (r"\b(?:goal|aspire|dream|ambition)\b", "identity.goals"),
    (r"\b(?:I am|I'm|my identity)\b", "identity.core"),
    (r"\b(?:family|friend|relationship)\b", "identity.social"),
    (r"\b(?:work|career|profession)\b", "identity.professional"),
]

# Default pattern if nothing specific matches
_DEFAULT_DOMAIN = "identity.general"


@dataclass
class Belief:
    """A single personal belief extracted from conversation."""

    text: str
    domain: str
    axiom: Axiom
    strength: float = 1.0


class IdentityState:
    """Personal world model — extracts axioms from conversation.

    Args:
        k: Number of blocks.
        l: Block length.
        library: AxiomLibrary to register discovered personal axioms.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int,
        l: int,  # noqa: E741
        library: AxiomLibrary,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self._library = library
        self._seed = seed

        # Beliefs
        self._beliefs: list[Belief] = []

        # Plutchik emotion state (8 dimensions, all start at 0.5 baseline)
        self._emotions: dict[str, float] = {
            e: 0.5 for e in PLUTCHIK_EMOTIONS
        }

        # Emotion vectors (one per emotion for VSA encoding)
        self._emotion_vecs: dict[str, np.ndarray] = {}
        for i, emotion in enumerate(PLUTCHIK_EMOTIONS):
            self._emotion_vecs[emotion] = self.bc.random_discrete(
                seed=seed + 2000 + i
            )

    @property
    def belief_count(self) -> int:
        """Number of registered personal beliefs."""
        return len(self._beliefs)

    def process_statement(self, text: str) -> list[Axiom]:
        """Extract axioms from a natural language statement.

        Scans the statement for known patterns (values, beliefs, passions,
        etc.) and creates axioms for each match. Each axiom is registered
        in the library.

        Args:
            text: Natural language statement.

        Returns:
            List of newly created Axiom instances.
        """
        text = text.strip()
        if not text:
            return []

        # Find matching domains
        matched_domains: list[str] = []
        for pattern, domain in _VALUE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matched_domains.append(domain)

        if not matched_domains:
            matched_domains = [_DEFAULT_DOMAIN]

        axioms: list[Axiom] = []
        for domain in matched_domains:
            # Create a condensed name from the text
            name = self._text_to_name(text)

            axiom = self._library.make_axiom(
                name=name,
                domain=domain,
                formula_str=text,
                operators=[],
            )
            self._library.register(axiom)

            belief = Belief(
                text=text,
                domain=domain,
                axiom=axiom,
            )
            self._beliefs.append(belief)
            axioms.append(axiom)

        return axioms

    def emotional_state(self) -> dict[str, float]:
        """Current Plutchik 8-emotion vector.

        Returns:
            Dict mapping emotion name -> intensity in [0, 1].
        """
        return dict(self._emotions)

    def update_emotion(self, emotion: str, value: float) -> None:
        """Update an emotion's intensity.

        Args:
            emotion: One of the 8 Plutchik emotions.
            value: New intensity (will be clamped to [0, 1]).
        """
        if emotion in self._emotions:
            self._emotions[emotion] = float(np.clip(value, 0.0, 1.0))

    def identity_vector(self) -> np.ndarray:
        """Compose the full identity vector from all personal axioms.

        V_Me = bundle(all_belief_vectors + weighted_emotion_vectors)

        Returns:
            Block-code vector of shape (k, l).
        """
        vecs: list[np.ndarray] = []

        # Add belief vectors
        for belief in self._beliefs:
            vecs.append(belief.axiom.vector)

        # Add emotion vectors weighted by intensity
        for emotion, intensity in self._emotions.items():
            if intensity > 0.1:
                emotion_vec = self._emotion_vecs[emotion]
                vecs.append(float(intensity) * emotion_vec)

        if not vecs:
            return np.zeros((self.k, self.l), dtype=np.float32)

        bundled = self.bc.bundle(vecs, normalize=True)
        # Normalize per block to unit norm so similarity() returns correct values
        norms = np.linalg.norm(bundled, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return (bundled / norms).astype(np.float32)

    def alignment_score(self, action_vec: np.ndarray) -> float:
        """How aligned is a proposed action with the identity?

        Uses block-code cosine similarity between the identity
        vector and the action vector.

        Args:
            action_vec: Block-code vector representing a proposed action.

        Returns:
            Similarity score in [-1, 1]. Higher = more aligned.
        """
        identity = self.identity_vector()
        return float(self.bc.similarity(identity, action_vec))

    @staticmethod
    def _text_to_name(text: str) -> str:
        """Convert text to a short axiom name.

        Takes the first 5 significant words and joins with underscore.
        """
        words = re.findall(r"[a-zA-Z]+", text.lower())
        # Filter stopwords
        stop = {"the", "a", "an", "is", "are", "was", "were", "i", "my", "in", "of"}
        significant = [w for w in words if w not in stop]
        if not significant:
            significant = words[:3]
        return "_".join(significant[:5])
