# tests/test_domains.py
"""Tests for domain modules — axiom seeding and registration."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.domains import seed_all, available_domains

K, L = 4, 32


class TestPhysicsDomain:
    def setup_method(self):
        import mowm.domains.physics  # noqa: F401
        self.lib = AxiomLibrary(k=K, l=L)

    def test_physics_registered(self):
        assert "physics" in available_domains()

    def test_seed_registers_axioms(self):
        seed_all(self.lib, domains=["physics"])
        assert len(self.lib) >= 10

    def test_axiom_vectors_valid_shape(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert ax.vector.shape == (K, L)
            assert ax.vector.dtype == np.float32

    def test_axiom_vectors_nonzero(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert np.any(ax.vector != 0)

    def test_axioms_have_operators(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert isinstance(ax.operators, list)

    def test_axioms_have_domain_prefix(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert ax.domain.startswith("physics")

    def test_known_axioms_present(self):
        seed_all(self.lib, domains=["physics"])
        names = [ax.name for ax in self.lib.all_axioms()]
        assert "Newton's Second Law" in names
        assert "Conservation of Energy" in names

    def test_axiom_vectors_are_deterministic(self):
        lib2 = AxiomLibrary(k=K, l=L)
        seed_all(self.lib, domains=["physics"])
        seed_all(lib2, domains=["physics"])
        axioms1 = sorted(self.lib.all_axioms(), key=lambda a: a.name)
        axioms2 = sorted(lib2.all_axioms(), key=lambda a: a.name)
        for a1, a2 in zip(axioms1, axioms2):
            sim = self.lib.bc.similarity(a1.vector, a2.vector)
            assert sim > 0.99, f"{a1.name} not deterministic: sim={sim}"


EXPECTED_DOMAINS = [
    "physics", "logic", "economics", "social", "linguistics",
    "causal", "atomic", "bio", "music", "identity", "colors",
    "language", "historical", "ensemble",
]


class TestAllDomains:
    EXPECTED_DOMAINS = EXPECTED_DOMAINS

    @classmethod
    def setup_class(cls):
        for d in cls.EXPECTED_DOMAINS:
            try:
                __import__(f"mowm.domains.{d}")
            except ImportError:
                pass

    def test_all_14_domains_registered(self):
        domains = available_domains()
        for d in self.EXPECTED_DOMAINS:
            assert d in domains, f"Domain '{d}' not registered"

    @pytest.mark.parametrize("domain_name", EXPECTED_DOMAINS)
    def test_domain_seeds_axioms(self, domain_name):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib, domains=[domain_name])
        assert len(lib) >= 5, f"{domain_name} seeded only {len(lib)} axioms"

    @pytest.mark.parametrize("domain_name", EXPECTED_DOMAINS)
    def test_domain_axiom_shapes(self, domain_name):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib, domains=[domain_name])
        for ax in lib.all_axioms():
            assert ax.vector.shape == (K, L)
            assert ax.vector.dtype == np.float32

    @pytest.mark.parametrize("domain_name", EXPECTED_DOMAINS)
    def test_domain_axioms_have_domain_prefix(self, domain_name):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib, domains=[domain_name])
        for ax in lib.all_axioms():
            assert ax.domain.startswith(domain_name)

    def test_total_axiom_count(self):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib)
        assert len(lib) >= 200, f"Expected 200+ axioms, got {len(lib)}"
