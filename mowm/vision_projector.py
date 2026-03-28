# mowm/vision_projector.py
"""VisionProjector — map image features into VSA block-code space.

Projects visual features (from DenseNet, ResNet, ViT, or any CNN backbone)
into the same block-code VSA space as text axioms, enabling mathematical
queries based on visual input.

Uses a Contrastive Projection Matrix (trained via Additive CE or
QR orthogonal) to align the visual feature space with the VSA codebook.

Reference: cnn_paper.md (Additive Cross-Entropy for visual perception)
"""
from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes


class VisionProjector:
    """Project image features into the block-code VSA space.

    Takes dense feature vectors from any vision backbone (DenseNet, ViT, etc.)
    and projects them into the (k, l) block-code space for cosine similarity
    search against the axiom library.

    Two modes:
      1. Pre-trained projection matrix (loaded from file or codebook)
      2. Random orthogonal projection (QR-based, for prototyping)

    Args:
        k: Number of VSA blocks.
        l: Block length.
        visual_dim: Dimension of the vision backbone output (e.g., 1024 for DenseNet).
        projection_matrix: Pre-trained or pre-computed P matrix (vsa_dim, visual_dim).
        seed: Random seed for QR fallback.
    """

    def __init__(
        self,
        k: int = 80,
        l: int = 128,  # noqa: E741
        visual_dim: int = 1024,
        projection_matrix: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        self._k = k
        self._l = l
        self._bc = BlockCodes(k=k, l=l)
        self._visual_dim = visual_dim
        vsa_dim = k * l

        if projection_matrix is not None:
            self._P = projection_matrix.astype(np.float32)
            self._visual_dim = projection_matrix.shape[1]
        elif vsa_dim >= visual_dim:
            # Full QR — orthogonal columns
            rng = np.random.default_rng(seed)
            Q, _ = np.linalg.qr(
                rng.standard_normal((vsa_dim, visual_dim)),
            )
            self._P = Q.astype(np.float32)
        else:
            # JL random projection (vsa_dim < visual_dim)
            rng = np.random.default_rng(seed)
            scale = 1.0 / np.sqrt(vsa_dim)
            self._P = (
                scale * rng.standard_normal((vsa_dim, visual_dim))
            ).astype(np.float32)

    def project(self, visual_features: np.ndarray) -> np.ndarray:
        """Project a visual feature vector into VSA block-code space.

        Args:
            visual_features: 1-D array of shape (visual_dim,).

        Returns:
            Block-code vector of shape (k, l), dtype float32.
        """
        features = visual_features.astype(np.float32).ravel()
        if features.shape[0] != self._P.shape[1]:
            raise ValueError(
                f"Feature dim {features.shape[0]} != "
                f"expected {self._P.shape[1]}"
            )

        projected = self._P @ features  # (vsa_dim,)
        return projected.reshape(self._k, self._l).astype(np.float32)

    def project_batch(
        self, features_batch: np.ndarray,
    ) -> np.ndarray:
        """Project a batch of visual features into VSA space.

        Args:
            features_batch: 2-D array of shape (N, visual_dim).

        Returns:
            Array of shape (N, k, l), dtype float32.
        """
        batch = features_batch.astype(np.float32)
        projected = (self._P @ batch.T).T  # (N, vsa_dim)
        return projected.reshape(
            -1, self._k, self._l,
        ).astype(np.float32)

    def find_matching_axioms(
        self,
        visual_features: np.ndarray,
        library: "AxiomLibrary",
        top_n: int = 5,
    ) -> list:
        """Find axioms most similar to visual input.

        Projects the image features into VSA space, then does cosine
        similarity search against the axiom library.

        Args:
            visual_features: 1-D array from vision backbone.
            library: AxiomLibrary to search.
            top_n: Number of results to return.

        Returns:
            List of Axiom instances sorted by similarity.
        """
        vsa_vec = self.project(visual_features)
        return library.select(vsa_vec, top_n=top_n)

    def save_projection(self, path: str) -> None:
        """Save projection matrix to .npy file."""
        np.save(path, self._P)

    @classmethod
    def from_file(
        cls,
        path: str,
        k: int = 80,
        l: int = 128,  # noqa: E741
    ) -> "VisionProjector":
        """Load a VisionProjector from a saved projection matrix."""
        P = np.load(path)
        return cls(k=k, l=l, projection_matrix=P)

    @property
    def visual_dim(self) -> int:
        """Dimension of expected visual input features."""
        return self._visual_dim

    @property
    def vsa_dim(self) -> int:
        """Dimension of VSA output (k * l)."""
        return self._k * self._l
