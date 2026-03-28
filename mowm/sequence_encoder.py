# mowm/sequence_encoder.py
"""SequenceEncoder — encode DNA, temporal event streams, and token sequences into VSA vectors.

Uses BlockCodes from cubemind for all VSA operations (bind, bundle, random_discrete).
Positional encoding is achieved via cyclic per-block roll (_permute).
"""
from __future__ import annotations

import hashlib

import numpy as np

from cubemind.ops.block_codes import BlockCodes

_NUCLEOTIDES = ("A", "C", "G", "T")


class SequenceEncoder:
    """Encode sequences into block-code VSA vectors.

    Supports three modalities:
    - DNA: k-mer fragmentation with permuted nucleotide binding
    - Temporal: time-sorted events with exponential recency weighting
    - Token: positional binding for ordered token sequences

    Args:
        k: Number of blocks per VSA vector.
        l: Block length (number of elements per block).  # noqa: E741
        seed: RNG seed for reproducible nucleotide and position vectors.
    """

    def __init__(self, k: int = 80, l: int = 128, seed: int = 0) -> None:  # noqa: E741
        self._bc = BlockCodes(k=k, l=l)
        self._k = k
        self._l = l
        self._seed = seed

        # Pre-generate one random_discrete vector per nucleotide, reproducibly.
        self._nuc_vecs: dict[str, np.ndarray] = {
            nuc: self._bc.random_discrete(seed=seed + i)
            for i, nuc in enumerate(_NUCLEOTIDES)
        }

        # Cache for position vectors (position index → vector).
        self._pos_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pos_vec(self, pos: int) -> np.ndarray:
        """Return a deterministic position vector for index *pos*."""
        if pos not in self._pos_cache:
            self._pos_cache[pos] = self._bc.random_discrete(seed=self._seed + 1000 + pos)
        return self._pos_cache[pos]

    def _str_vec(self, text: str) -> np.ndarray:
        """Derive a deterministic VSA vector from an arbitrary string."""
        digest = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        seed = (self._seed + digest) % (2**31)
        return self._bc.random_discrete(seed=seed)

    @staticmethod
    def _permute(vec: np.ndarray, steps: int) -> np.ndarray:
        """Cyclic roll each block of *vec* by *steps* positions (positional encoding)."""
        if steps == 0:
            return vec
        # vec shape: (k, l) — roll along the l axis independently per block.
        return np.roll(vec, steps, axis=1)

    def _zero(self) -> np.ndarray:
        """Return a zero vector with the correct shape and dtype."""
        return np.zeros((self._k, self._l), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_dna(self, sequence: str, k: int = 3) -> np.ndarray:
        """Encode a DNA sequence into a block-code VSA vector via k-mer fragmentation.

        Each valid k-mer is represented by binding the permuted nucleotide vectors of
        its constituent bases.  All k-mer vectors are bundled into a single output.
        Invalid nucleotide characters are silently ignored when filtering bases.

        Args:
            sequence: DNA string (characters A/C/G/T; unknown characters are skipped).
            k: K-mer length (default 3).

        Returns:
            Block-code VSA vector of shape ``(self._k, self._l)``, dtype float32.
        """
        # Filter to valid nucleotides only.
        clean = [b for b in sequence.upper() if b in self._nuc_vecs]

        kmers: list[np.ndarray] = []
        for start in range(max(1, len(clean) - k + 1)):
            fragment = clean[start : start + k]
            if not fragment:
                continue
            # Bind permuted nucleotide vectors: pos 0 stays, pos i rolls by i.
            kmer_vec = self._permute(self._nuc_vecs[fragment[0]], 0)
            for offset, base in enumerate(fragment[1:], start=1):
                kmer_vec = self._bc.bind(kmer_vec, self._permute(self._nuc_vecs[base], offset))
            kmers.append(kmer_vec)

        if not kmers:
            return self._zero()

        return self._bc.bundle(kmers).astype(np.float32)

    def encode_temporal(
        self,
        events: list[tuple[float, str]],
    ) -> np.ndarray:
        """Encode a temporal event stream into a block-code VSA vector.

        Events are sorted chronologically, bound with a quantised time vector, then
        bundled with exponential recency weighting (most-recent event has weight 1.0).

        Args:
            events: List of ``(timestamp, content_string)`` pairs.

        Returns:
            Block-code VSA vector of shape ``(self._k, self._l)``, dtype float32.
        """
        if not events:
            return self._zero()

        sorted_events = sorted(events, key=lambda e: e[0])
        n = len(sorted_events)

        vecs: list[np.ndarray] = []
        weights: list[float] = []

        for rank, (timestamp, content) in enumerate(sorted_events):
            # Quantise timestamp to an integer bucket for a reproducible time vector.
            time_bucket = int(abs(timestamp)) % 10_000
            time_vec = self._bc.random_discrete(seed=self._seed + 500_000 + time_bucket)
            content_vec = self._str_vec(content)
            bound = self._bc.bind(time_vec, content_vec)
            vecs.append(bound)
            # Recency weight: most-recent (highest rank) gets weight 1.0.
            recency = np.exp(rank - (n - 1))  # exp(0)=1 for last, decays for earlier
            weights.append(float(recency))

        # Weighted bundle: scale each vector then bundle.
        weighted = [v * w for v, w in zip(vecs, weights)]
        result = self._bc.bundle(weighted)

        # L2-normalise per block so the output lives on the VSA manifold.
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (result / norms).astype(np.float32)

    def encode_tokens(self, tokens: list[str]) -> np.ndarray:
        """Encode an ordered token sequence into a block-code VSA vector.

        Each token is bound with a position vector (so order is preserved) and all
        bound pairs are bundled into a single output.

        Args:
            tokens: Ordered list of string tokens.

        Returns:
            Block-code VSA vector of shape ``(self._k, self._l)``, dtype float32.
        """
        if not tokens:
            return self._zero()

        bound_vecs: list[np.ndarray] = []
        for pos, token in enumerate(tokens):
            tok_vec = self._str_vec(token)
            pos_vec = self._pos_vec(pos)
            bound_vecs.append(self._bc.bind(tok_vec, pos_vec))

        return self._bc.bundle(bound_vecs).astype(np.float32)
