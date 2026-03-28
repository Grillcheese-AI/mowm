# tests/test_identity_state.py
"""Tests for IdentityState — personal world model and axiom extraction."""
from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes

from mowm.axiom_library import AxiomLibrary
from mowm.identity_state import IdentityState, PLUTCHIK_EMOTIONS

K, L = 4, 32


class TestPlutchikEmotions:
    def test_eight_emotions(self):
        assert len(PLUTCHIK_EMOTIONS) == 8

    def test_known_emotions(self):
        expected = {
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation",
        }
        assert set(PLUTCHIK_EMOTIONS) == expected


class TestIdentityStateInit:
    def test_creation(self):
        lib = AxiomLibrary(k=K, l=L)
        state = IdentityState(k=K, l=L, library=lib)
        assert state.k == K
        assert state.l == L

    def test_initial_emotional_state(self):
        lib = AxiomLibrary(k=K, l=L)
        state = IdentityState(k=K, l=L, library=lib)
        emotions = state.emotional_state()
        assert len(emotions) == 8
        for name, val in emotions.items():
            assert name in PLUTCHIK_EMOTIONS
            assert 0.0 <= val <= 1.0


class TestProcessStatement:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)
        self.state = IdentityState(k=K, l=L, library=self.lib)

    def test_extract_value_axiom(self):
        axioms = self.state.process_statement(
            "Family is the most important value"
        )
        assert len(axioms) >= 1
        assert axioms[0].domain.startswith("identity")

    def test_extract_belief_axiom(self):
        axioms = self.state.process_statement(
            "I believe in hard work and perseverance"
        )
        assert len(axioms) >= 1
        for ax in axioms:
            assert ax.vector.shape == (K, L)

    def test_axioms_registered_in_library(self):
        initial = len(self.lib)
        self.state.process_statement("Music is my passion")
        assert len(self.lib) > initial

    def test_empty_statement(self):
        axioms = self.state.process_statement("")
        assert axioms == []

    def test_multiple_statements_accumulate(self):
        self.state.process_statement("I love physics")
        self.state.process_statement("I value creativity")
        assert self.state.belief_count >= 2


class TestEmotionalState:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)
        self.state = IdentityState(k=K, l=L, library=self.lib)

    def test_update_emotion(self):
        self.state.update_emotion("joy", 0.8)
        emotions = self.state.emotional_state()
        assert emotions["joy"] == pytest.approx(0.8, abs=0.1)

    def test_emotion_clamped(self):
        self.state.update_emotion("anger", 5.0)
        emotions = self.state.emotional_state()
        assert emotions["anger"] <= 1.0

    def test_emotion_non_negative(self):
        self.state.update_emotion("fear", -1.0)
        emotions = self.state.emotional_state()
        assert emotions["fear"] >= 0.0


class TestIdentityVector:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)
        self.state = IdentityState(k=K, l=L, library=self.lib)

    def test_identity_vector_shape(self):
        vec = self.state.identity_vector()
        assert vec.shape == (K, L)
        assert vec.dtype == np.float32

    def test_identity_vector_changes_with_beliefs(self):
        bc = BlockCodes(k=K, l=L)
        v1 = self.state.identity_vector()
        self.state.process_statement("I am a scientist")
        v2 = self.state.identity_vector()
        sim = bc.similarity(v1, v2)
        assert sim < 0.99  # should change after adding belief

    def test_identity_vector_deterministic(self):
        self.state.process_statement("I love math")
        v1 = self.state.identity_vector()
        v2 = self.state.identity_vector()
        np.testing.assert_array_equal(v1, v2)


class TestAlignmentScore:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)
        self.state = IdentityState(k=K, l=L, library=self.lib)

    def test_alignment_returns_float(self):
        bc = BlockCodes(k=K, l=L)
        action_vec = bc.random_discrete(seed=42)
        score = self.state.alignment_score(action_vec)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_aligned_action_scores_higher(self):
        self.state.process_statement("I value honesty")
        identity_vec = self.state.identity_vector()
        # An action vector similar to identity should score high
        score = self.state.alignment_score(identity_vec)
        assert score > 0.5
