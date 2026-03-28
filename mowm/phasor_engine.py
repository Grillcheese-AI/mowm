# mowm/phasor_engine.py
"""PhasorEngine — harmonic phasor bank for frequency-domain VSA encoding.

Adapted from AURA_GENESIS core/phasor.py pattern. Implements a bank of
leaky resonators that decompose signals into harmonic components.

Each phasor: x(t) = A * exp(i * (omega * t + phi))
The bank of H harmonics decomposes signals into frequency components.

Used by Music, Colors, and Ensemble domains.
"""
from __future__ import annotations

import numpy as np

from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes


class PhasorEngine:
    """Harmonic phasor bank with leaky resonator integration.

    Args:
        k: Number of blocks for VSA vectors.
        l: Block length for VSA vectors.
        n_harmonics: Number of harmonic frequencies in the bank.
        rho: Leaky integrator decay factor (0 < rho < 1).
            Higher = longer memory.
        f_base: Base frequency (Hz) for harmonic series.
        sample_rate: Internal sample rate for phase advancement.
    """

    def __init__(
        self,
        k: int,
        l: int,  # noqa: E741
        n_harmonics: int = 64,
        rho: float = 0.985,
        f_base: float = 55.0,
        sample_rate: float = 44100.0,
    ) -> None:
        self.k = k
        self.l = l
        self.n_harmonics = n_harmonics
        self.rho = rho
        self.f_base = f_base
        self.sample_rate = sample_rate

        self.bc = BlockCodes(k=k, l=l)
        self.encoder = WorldEncoder(k=k, l=l)

        # Harmonic frequencies: f_base * (1, 2, 3, ..., n_harmonics)
        self._freqs = f_base * np.arange(1, n_harmonics + 1, dtype=np.float64)

        # Angular frequencies
        self._omegas = 2.0 * np.pi * self._freqs / sample_rate

        # Phasor state: complex amplitude for each harmonic
        self._state = np.zeros(n_harmonics, dtype=np.complex128)

        # Time step counter
        self._t = 0

    def reset(self) -> None:
        """Reset phasor bank state to zero."""
        self._state = np.zeros(self.n_harmonics, dtype=np.complex128)
        self._t = 0

    def step(self, signal: float) -> np.ndarray:
        """Process one time step through the leaky resonator bank.

        Each harmonic's state is updated:
            state_h = rho * state_h * exp(i * omega_h) + signal

        Args:
            signal: Input signal value at this time step.

        Returns:
            Harmonic feature vector of shape (n_harmonics,) — magnitudes.
        """
        # Phase rotation for each harmonic
        rotation = np.exp(1j * self._omegas)

        # Leaky resonator update
        self._state = self.rho * self._state * rotation + signal
        self._t += 1

        # Return magnitudes as features
        return np.abs(self._state).astype(np.float64)

    def encode_chord(self, frequencies: list[float]) -> np.ndarray:
        """Encode a chord as superposition of harmonic phasors in VSA space.

        Each frequency is mapped to the nearest harmonic in the bank,
        and its contribution is encoded as a block-code vector. All
        frequency vectors are bundled.

        Args:
            frequencies: List of frequencies in Hz.

        Returns:
            Block-code vector of shape (k, l).
        """
        if not frequencies:
            return np.zeros((self.k, self.l), dtype=np.float32)

        freq_vecs: list[np.ndarray] = []
        for freq in frequencies:
            # Gaussian spread around freq across harmonic bins
            spacing = self.f_base  # Hz between harmonics

            # Create a feature vector using Gaussian spread around the freq
            features = np.zeros(self.n_harmonics, dtype=np.float64)
            for j in range(self.n_harmonics):
                hz_dist = abs(self._freqs[j] - freq)
                features[j] = np.exp(-0.5 * (hz_dist / spacing) ** 2)

            # Encode features into VSA space
            vec = self._features_to_vsa(features)
            freq_vecs.append(vec)

        bundled = self.bc.bundle(freq_vecs, normalize=True)
        return self._unit_normalize(bundled)

    def encode_spectrum(
        self, signal: np.ndarray, sample_rate: float,
    ) -> np.ndarray:
        """FFT-based spectral encoding into block-code space.

        Computes the magnitude spectrum of the signal, maps it to
        harmonic bins, and encodes the result as a VSA vector.

        Args:
            signal: 1D signal array.
            sample_rate: Signal sample rate in Hz.

        Returns:
            Block-code vector of shape (k, l).
        """
        if len(signal) == 0:
            return np.zeros((self.k, self.l), dtype=np.float32)

        # Compute FFT
        fft_vals = np.fft.rfft(signal)
        magnitudes = np.abs(fft_vals)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)

        # Map FFT bins to harmonic bins
        features = np.zeros(self.n_harmonics, dtype=np.float64)
        for i, harm_freq in enumerate(self._freqs):
            # Find nearest FFT bin
            if len(fft_freqs) > 0:
                idx = int(np.argmin(np.abs(fft_freqs - harm_freq)))
                if idx < len(magnitudes):
                    features[i] = float(magnitudes[idx])

        # Normalize features
        max_feat = features.max()
        if max_feat > 0:
            features = features / max_feat

        return self._features_to_vsa(features)

    def harmonic_similarity(
        self, a: np.ndarray, b: np.ndarray,
    ) -> float:
        """Consonance measure between two phasor encodings.

        Uses block-code cosine similarity as the consonance metric.

        Args:
            a: First block-code vector (k, l).
            b: Second block-code vector (k, l).

        Returns:
            Similarity score in [-1, 1].
        """
        return float(self.bc.similarity(a, b))

    @staticmethod
    def _unit_normalize(vec: np.ndarray) -> np.ndarray:
        """Normalize each block to unit L2 norm.

        Args:
            vec: Block-code vector of shape (k, l).

        Returns:
            Per-block unit-normalized array of shape (k, l).
        """
        norms = np.linalg.norm(vec, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return (vec / norms).astype(np.float32)

    def _features_to_vsa(self, features: np.ndarray) -> np.ndarray:
        """Convert a harmonic feature vector to a block-code VSA vector.

        Each active harmonic contributes a weighted base vector. All are
        bundled into a single representation.

        Args:
            features: Feature vector of shape (n_harmonics,).

        Returns:
            Block-code vector of shape (k, l).
        """
        active_indices = np.where(features > 1e-10)[0]

        if len(active_indices) == 0:
            return np.zeros((self.k, self.l), dtype=np.float32)

        vecs: list[np.ndarray] = []
        for idx in active_indices:
            # Deterministic base vector for each harmonic
            base_vec = self.bc.random_discrete(seed=42 + int(idx))
            vecs.append(float(features[idx]) * base_vec)

        stacked = np.stack(vecs, axis=0)
        bundled = stacked.sum(axis=0).astype(np.float32)

        # Normalize per block
        block_norms = np.linalg.norm(bundled, axis=-1, keepdims=True)
        block_norms = np.where(block_norms < 1e-10, 1.0, block_norms)
        return (bundled / block_norms).astype(np.float32)
