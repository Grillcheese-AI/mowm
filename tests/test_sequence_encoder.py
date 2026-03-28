# tests/test_sequence_encoder.py
"""Tests for SequenceEncoder — DNA, temporal, and token encoding."""
from __future__ import annotations
import numpy as np
from cubemind.ops.block_codes import BlockCodes
from mowm.sequence_encoder import SequenceEncoder

K, L = 4, 32

class TestDNAEncoding:
    def setup_method(self):
        self.enc = SequenceEncoder(k=K, l=L, seed=42)
    def test_encode_dna_shape(self):
        assert self.enc.encode_dna("ACGTACGT").shape == (K, L)
    def test_encode_dna_dtype(self):
        assert self.enc.encode_dna("ACGT").dtype == np.float32
    def test_encode_dna_nonzero(self):
        assert np.any(self.enc.encode_dna("ACGTACGT") != 0)
    def test_encode_dna_deterministic(self):
        e1 = SequenceEncoder(k=K, l=L, seed=42)
        e2 = SequenceEncoder(k=K, l=L, seed=42)
        np.testing.assert_array_equal(e1.encode_dna("ACGT"), e2.encode_dna("ACGT"))
    def test_encode_dna_different_sequences(self):
        v1 = self.enc.encode_dna("ACGTACGT")
        v2 = self.enc.encode_dna("TGCATGCA")
        assert BlockCodes(k=K, l=L).similarity(v1, v2) < 0.95
    def test_encode_dna_kmer_size(self):
        assert self.enc.encode_dna("ACGTACGT", k=4).shape == (K, L)
    def test_encode_dna_short_sequence(self):
        assert self.enc.encode_dna("AC").shape == (K, L)
    def test_encode_dna_single_base(self):
        assert self.enc.encode_dna("A").shape == (K, L)
    def test_encode_dna_invalid_base_ignored(self):
        assert self.enc.encode_dna("ACXGT").shape == (K, L)

class TestTemporalEncoding:
    def setup_method(self):
        self.enc = SequenceEncoder(k=K, l=L, seed=42)
    def test_encode_temporal_shape(self):
        assert self.enc.encode_temporal([(0.0, "start"), (1.0, "middle"), (2.0, "end")]).shape == (K, L)
    def test_encode_temporal_dtype(self):
        assert self.enc.encode_temporal([(0.0, "start")]).dtype == np.float32
    def test_encode_temporal_nonzero(self):
        assert np.any(self.enc.encode_temporal([(0.0, "a"), (1.0, "b")]) != 0)
    def test_encode_temporal_empty(self):
        assert self.enc.encode_temporal([]).shape == (K, L)
    def test_encode_temporal_order_matters(self):
        v1 = self.enc.encode_temporal([(0.0, "a"), (1.0, "b")])
        v2 = self.enc.encode_temporal([(0.0, "b"), (1.0, "a")])
        assert BlockCodes(k=K, l=L).similarity(v1, v2) < 0.95
    def test_encode_temporal_recency_weighting(self):
        assert self.enc.encode_temporal([(0.0, "old"), (100.0, "recent")]).shape == (K, L)

class TestTokenEncoding:
    def setup_method(self):
        self.enc = SequenceEncoder(k=K, l=L, seed=42)
    def test_encode_tokens_shape(self):
        assert self.enc.encode_tokens(["the", "cat", "sat"]).shape == (K, L)
    def test_encode_tokens_dtype(self):
        assert self.enc.encode_tokens(["hello"]).dtype == np.float32
    def test_encode_tokens_nonzero(self):
        assert np.any(self.enc.encode_tokens(["hello", "world"]) != 0)
    def test_encode_tokens_empty(self):
        assert self.enc.encode_tokens([]).shape == (K, L)
    def test_encode_tokens_order_matters(self):
        v1 = self.enc.encode_tokens(["cat", "dog"])
        v2 = self.enc.encode_tokens(["dog", "cat"])
        assert BlockCodes(k=K, l=L).similarity(v1, v2) < 0.99
    def test_encode_tokens_deterministic(self):
        e1 = SequenceEncoder(k=K, l=L, seed=42)
        e2 = SequenceEncoder(k=K, l=L, seed=42)
        np.testing.assert_array_equal(e1.encode_tokens(["a", "b", "c"]), e2.encode_tokens(["a", "b", "c"]))
