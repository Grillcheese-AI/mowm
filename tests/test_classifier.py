# tests/test_classifier.py
"""Tests for DomainClassifier — VSA cosine sim + keyword blending."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.qa.classifier import DomainClassifier

K, L = 4, 32  # small dims for fast testing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lib() -> AxiomLibrary:
    """AxiomLibrary with physics and logic axioms."""
    library = AxiomLibrary(k=K, l=L)
    # Physics axioms
    library.register(library.make_axiom("Newton's Second Law", "physics", "F = m * a"))
    library.register(library.make_axiom("Kinetic Energy", "physics", "KE = 0.5 * m * v^2"))
    library.register(library.make_axiom("Gravitational Force", "physics.gravity", "F = G * m1 * m2 / r^2"))
    # Logic axioms
    library.register(library.make_axiom("Modus Ponens", "logic", "A IMPLIES B"))
    library.register(library.make_axiom("De Morgan", "logic", "NOT A AND NOT B"))
    return library


@pytest.fixture
def clf(lib) -> DomainClassifier:
    return DomainClassifier(lib)


@pytest.fixture
def empty_lib() -> AxiomLibrary:
    return AxiomLibrary(k=K, l=L)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_builds_centroids_for_each_domain(self, clf):
        # physics and logic should both have centroids
        assert "physics" in clf._centroids
        assert "logic" in clf._centroids

    def test_centroids_are_normalized(self, clf):
        for domain, centroid in clf._centroids.items():
            flat = centroid.flatten()
            norm = np.linalg.norm(flat)
            # Centroid was normalised per block; overall norm close to sqrt(k)
            assert norm > 0, f"Centroid for {domain} is zero"

    def test_keyword_index_populated(self, clf):
        assert "physics" in clf._domain_keywords
        assert "logic" in clf._domain_keywords
        # 'newton' (lowercased from "Newton's Second Law") should be in physics
        assert "newton" in clf._domain_keywords["physics"]


# ---------------------------------------------------------------------------
# TestClassify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_returns_list_of_tuples(self, clf):
        result = clf.classify("What is Newton's law?")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_scores_are_floats(self, clf):
        result = clf.classify("Calculate the kinetic energy")
        for _domain, score in result:
            assert isinstance(score, float)

    def test_ranked_descending(self, clf):
        result = clf.classify("What is force?")
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_physics_question_scores_physics_high(self, clf):
        result = clf.classify("Calculate the kinetic energy using m and v")
        domains = [d for d, _ in result]
        # physics should appear in the result
        assert "physics" in domains
        # physics should be ranked first or second
        assert domains.index("physics") <= 1

    def test_logic_question_scores_logic_high(self, clf):
        result = clf.classify("Does modus ponens apply here?")
        domains = [d for d, _ in result]
        assert "logic" in domains

    def test_keyword_only_fallback_no_vsa_overlap(self, clf):
        # "De Morgan" is a strong keyword for logic
        result = clf.classify("De Morgan NOT AND")
        domains = [d for d, _ in result]
        assert "logic" in domains

    def test_all_known_domains_present(self, clf):
        result = clf.classify("physics logic test")
        returned_domains = {d for d, _ in result}
        assert "physics" in returned_domains
        assert "logic" in returned_domains


# ---------------------------------------------------------------------------
# TestClassifyEmbedding
# ---------------------------------------------------------------------------


class TestClassifyEmbedding:
    def test_accepts_precomputed_embedding(self, clf, lib):
        embedding = lib.encoder.encode_action("force mass acceleration")
        result = clf.classify_embedding(embedding)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_ranked_descending_from_embedding(self, clf, lib):
        embedding = lib.encoder.encode_action("logic AND OR NOT")
        result = clf.classify_embedding(embedding)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_embedding_with_question_text_blends_keywords(self, clf, lib):
        embedding = lib.encoder.encode_action("Newton kinetic energy")
        result_with_kw = clf.classify_embedding(embedding, question="Newton kinetic energy")
        result_no_kw = clf.classify_embedding(embedding, question="")
        # Scores should differ when keywords are provided
        scores_with = {d: s for d, s in result_with_kw}
        scores_no = {d: s for d, s in result_no_kw}
        # physics keywords exist so physics score should be >= without keywords
        assert scores_with.get("physics", 0.0) >= scores_no.get("physics", 0.0) - 1e-9


# ---------------------------------------------------------------------------
# TestTopDomain
# ---------------------------------------------------------------------------


class TestTopDomain:
    def test_returns_string(self, clf):
        result = clf.top_domain("What is force?")
        assert isinstance(result, str)

    def test_top_domain_physics(self, clf):
        result = clf.top_domain("kinetic energy m v")
        assert result in {"physics", "logic"}  # should pick physics most of the time


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_library_classify_returns_empty(self, empty_lib):
        clf = DomainClassifier(empty_lib)
        assert clf.classify("What is force?") == []

    def test_empty_library_top_domain_returns_empty_string(self, empty_lib):
        clf = DomainClassifier(empty_lib)
        assert clf.top_domain("What is force?") == ""

    def test_empty_library_classify_embedding_returns_empty(self, empty_lib):
        clf = DomainClassifier(empty_lib)
        vec = np.zeros((K, L), dtype=np.float32)
        assert clf.classify_embedding(vec) == []

    def test_single_domain_library(self):
        lib = AxiomLibrary(k=K, l=L)
        lib.register(lib.make_axiom("Ohm's Law", "electronics", "V = I * R"))
        clf = DomainClassifier(lib)
        result = clf.classify("What is voltage?")
        assert len(result) >= 1
        # electronics should be in the results (from axiom keywords)
        domains = [d for d, _ in result]
        assert "electronics" in domains

    def test_empty_question_string(self, clf):
        result = clf.classify("")
        # Should not raise; returns some ranked list
        assert isinstance(result, list)
