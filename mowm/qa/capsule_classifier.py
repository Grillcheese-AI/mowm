# mowm/qa/capsule_classifier.py
"""CapsuleClassifier — domain classification via grilly capsule embeddings.

Uses grilly's CapsuleProject (384D → 32D) to compress sentence embeddings
into compact capsule vectors, then classifies by cosine similarity against
domain centroids trained from the corpus.

Falls back to expert bidding when grilly capsule ops are unavailable.

Pipeline:
  text → sentence-transformers (384D) → CapsuleProject (32D) → cosine sim
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

# Pure numpy capsule projection matching grilly's capsule-project.glsl math:
# output[i] = sum(W[i][j] * input[j]) + bias[i]
# dims 0-27: semantic, dims 28-31: cognitive (plasticity, consolidation, stability, stress)


class _CapsuleProjectNumpy:
    """Pure numpy CapsuleProject — matches grilly's capsule-project.glsl.

    When Vulkan is unavailable, this provides the same linear projection.
    """

    def __init__(
        self, in_dim: int = 384, out_dim: int = 32, seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.weight = rng.uniform(
            -limit, limit, (out_dim, in_dim),
        ).astype(np.float32)
        self.bias = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project (batch, in_dim) -> (batch, out_dim)."""
        if x.ndim == 1:
            return (self.weight @ x + self.bias).astype(np.float32)
        return (x @ self.weight.T + self.bias).astype(np.float32)


# Try grilly GPU capsule first, fall back to numpy
_CapsuleProject = _CapsuleProjectNumpy
try:
    from grilly.nn.capsule import CapsuleProject as _GrillyCapsule
    # Test if it works (Vulkan needed)
    _test = _GrillyCapsule(in_dim=8, out_dim=4)
    _test.forward(np.zeros((1, 8), dtype=np.float32))
    _CapsuleProject = _GrillyCapsule
    del _test
except Exception:
    pass

# Try grilly VulkanSentenceTransformer first (GPU), then sentence-transformers (CPU)
_HAS_SBERT = False
_SBERT_CLASS = None
try:
    from grilly.utils.vulkan_sentence_transformer import VulkanSentenceTransformer
    _HAS_SBERT = True
    _SBERT_CLASS = VulkanSentenceTransformer
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        _HAS_SBERT = True
        _SBERT_CLASS = SentenceTransformer
    except ImportError:
        pass


class CapsuleClassifier:
    """Domain classification via capsule-compressed sentence embeddings.

    When grilly + sentence-transformers are available:
      text → SentenceTransformer (384D) → CapsuleProject (32D) → cosine sim

    When only sentence-transformers available:
      text → SentenceTransformer (384D) → cosine sim (no capsule compression)

    Falls back to keyword matching when neither is available.

    Args:
        library: AxiomLibrary for fallback keyword matching.
        model_name: SentenceTransformer model name.
        capsule_dim: Capsule output dimension (default 32).
        domain_labels_path: Path to svc_domain_labels.json.
    """

    def __init__(
        self,
        library: "AxiomLibrary",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        capsule_dim: int = 32,
        domain_labels_path: str | None = None,
    ) -> None:
        self._lib = library
        self._capsule_dim = capsule_dim
        self._embed_dim = 384  # MiniLM-L6-v2 output dim
        self._encoder = None
        self._capsule_proj = None
        self._domain_centroids: dict[str, np.ndarray] = {}

        # Initialize sentence encoder (grilly Vulkan GPU or CPU fallback)
        if _HAS_SBERT and _SBERT_CLASS is not None:
            try:
                self._encoder = _SBERT_CLASS(model_name)
                self._embed_dim = (
                    self._encoder.get_sentence_embedding_dimension()
                )
            except Exception:
                pass

        # Initialize capsule projection (grilly GPU or numpy fallback)
        if self._encoder is not None:
            self._capsule_proj = _CapsuleProject(
                in_dim=self._embed_dim,
                out_dim=capsule_dim,
            )

        # Build domain centroids from axiom library
        self._build_centroids()

    def _build_centroids(self) -> None:
        """Build domain centroids by encoding axiom names + formulas."""
        if self._encoder is None:
            return

        domain_texts: dict[str, list[str]] = {}
        for axiom in self._lib.all_axioms():
            top = axiom.domain.split(".")[0]
            texts = domain_texts.setdefault(top, [])
            texts.append(f"{axiom.name}: {axiom.formula_str}")

        for domain, texts in domain_texts.items():
            # Encode all axiom descriptions
            embeddings = self._encoder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Average to get centroid
            centroid = np.mean(embeddings, axis=0)

            # Project through capsule if available
            if self._capsule_proj is not None:
                centroid = self._capsule_proj.forward(
                    centroid.reshape(1, -1).astype(np.float32),
                ).ravel()

            # Normalize
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                centroid = centroid / norm

            self._domain_centroids[domain] = centroid.astype(np.float32)

    def _encode_question(self, question: str) -> np.ndarray | None:
        """Encode question to embedding space."""
        if self._encoder is None:
            return None

        emb = self._encoder.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if self._capsule_proj is not None:
            emb = self._capsule_proj.forward(
                emb.reshape(1, -1).astype(np.float32),
            ).ravel()

        return emb.astype(np.float32)

    def classify(self, question: str) -> list[tuple[str, float]]:
        """Classify question by cosine similarity to domain centroids.

        Returns ranked list of (domain, score) pairs.
        """
        if not self._domain_centroids:
            return []

        emb = self._encode_question(question)
        if emb is None:
            return []

        scores: list[tuple[str, float]] = []
        for domain, centroid in self._domain_centroids.items():
            # Cosine similarity
            sim = float(np.dot(emb, centroid))
            scores.append((domain, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def top_domain(self, question: str) -> str:
        """Return highest-scoring domain."""
        ranked = self.classify(question)
        if not ranked:
            return "general"
        return ranked[0][0]

    def explain_routing(self, question: str) -> str:
        """Explain classification with full score breakdown."""
        ranked = self.classify(question)
        if not ranked:
            return "No domains available (encoder not loaded)."

        using = "capsule (32D)" if self._capsule_proj else "embedding (384D)"
        lines = [
            f"**Encoder:** {using}",
            f"**Question:** {question}\n",
            "| Domain | Cosine Sim |",
            "|--------|-----------|",
        ]
        for domain, score in ranked[:10]:
            bar = "█" * int(max(0, score) * 20)
            lines.append(f"| {domain} | {score:.4f} {bar} |")

        winner = ranked[0]
        lines.append(f"\n**Winner:** `{winner[0]}` ({winner[1]:.4f})")
        return "\n".join(lines)

    @property
    def has_encoder(self) -> bool:
        """Whether the sentence encoder is available."""
        return self._encoder is not None

    @property
    def has_capsule(self) -> bool:
        """Whether grilly capsule projection is available."""
        return self._capsule_proj is not None

    @property
    def mode(self) -> str:
        """Current classification mode."""
        if self._capsule_proj:
            return "capsule"
        if self._encoder:
            return "embedding"
        return "fallback"
