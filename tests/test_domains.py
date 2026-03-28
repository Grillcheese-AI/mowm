# tests/test_domains.py
"""Tests for domain modules — axiom seeding and registration."""
from __future__ import annotations

import numpy as np

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
