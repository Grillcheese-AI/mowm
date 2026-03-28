# tests/test_expert_bid.py
"""Tests for ExpertBidClassifier — experts compete to route questions."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.domains import seed_all
from mowm.qa.expert_bid_classifier import ExpertBid, ExpertBidClassifier

K, L = 4, 32


@pytest.fixture
def full_lib() -> AxiomLibrary:
    """Library with all domains seeded."""
    for d in [
        "physics", "logic", "economics", "social", "linguistics",
        "causal", "atomic", "bio", "music", "identity", "colors",
        "language", "historical", "ensemble", "security",
    ]:
        try:
            __import__(f"mowm.domains.{d}")
        except ImportError:
            pass
    lib = AxiomLibrary(k=K, l=L)
    seed_all(lib)
    return lib


@pytest.fixture
def clf(full_lib) -> ExpertBidClassifier:
    return ExpertBidClassifier(full_lib)


class TestExpertBid:
    def test_bid_returns_ranked_list(self, clf):
        bids = clf.bid("What is force?")
        assert isinstance(bids, list)
        assert len(bids) > 0
        assert all(isinstance(b, ExpertBid) for b in bids)

    def test_bids_are_sorted_descending(self, clf):
        bids = clf.bid("Calculate kinetic energy")
        scores = [b.score for b in bids]
        assert scores == sorted(scores, reverse=True)

    def test_physics_wins_for_force(self, clf):
        bids = clf.bid("What is the force when m=10 and a=9.8?")
        assert bids[0].domain == "physics"

    def test_economics_wins_for_supply(self, clf):
        bids = clf.bid("What determines the market equilibrium price?")
        assert bids[0].domain == "economics"

    def test_logic_wins_for_modus_ponens(self, clf):
        bids = clf.bid("Explain modus ponens and logical implication")
        assert bids[0].domain == "logic"

    def test_bio_wins_for_dna(self, clf):
        bids = clf.bid("How does DNA encode genetic information?")
        assert bids[0].domain == "bio"

    def test_atomic_wins_for_quantum(self, clf):
        bids = clf.bid("Explain the Schrodinger equation in quantum mechanics")
        assert bids[0].domain == "atomic"

    def test_music_wins_for_harmony(self, clf):
        bids = clf.bid("What is the harmonic series and equal temperament?")
        assert bids[0].domain == "music"

    def test_security_wins_for_encryption(self, clf):
        bids = clf.bid("How does RSA encryption and authentication work?")
        assert bids[0].domain == "security"

    def test_historical_wins_for_civilization(self, clf):
        bids = clf.bid("Explain the civilizational cycle and empire duration in history")
        assert bids[0].domain == "historical"

    def test_variable_match_boosts_physics(self, clf):
        bids = clf.bid("What is F when m=10 and a=9.8?")
        physics_bid = next(b for b in bids if b.domain == "physics")
        assert physics_bid.variable_match >= 1

    def test_top_axiom_populated(self, clf):
        bids = clf.bid("What is Newton's Second Law?")
        physics_bid = next(b for b in bids if b.domain == "physics")
        assert "Newton" in physics_bid.top_axiom


class TestClassifyInterface:
    """Test drop-in replacement interface."""

    def test_classify_returns_tuples(self, clf):
        result = clf.classify("What is force?")
        assert isinstance(result, list)
        assert all(
            isinstance(d, str) and isinstance(s, float)
            for d, s in result
        )

    def test_top_domain_returns_string(self, clf):
        domain = clf.top_domain("What is force?")
        assert isinstance(domain, str)
        assert len(domain) > 0

    def test_top_domain_physics(self, clf):
        domain = clf.top_domain("Calculate F = m * a with m=10")
        assert domain == "physics"

    def test_empty_question(self, clf):
        domain = clf.top_domain("")
        assert isinstance(domain, str)


class TestExplainRouting:
    def test_explain_returns_markdown(self, clf):
        explanation = clf.explain_routing("What is force?")
        assert "Domain" in explanation
        assert "Score" in explanation
        assert "Winner" in explanation

    def test_explain_shows_winner(self, clf):
        explanation = clf.explain_routing(
            "What is the force when m=10 and a=9.8?"
        )
        assert "physics" in explanation.lower()
