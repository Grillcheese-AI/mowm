# tests/test_security_domain.py
"""Tests for security domain — 15th domain module."""
from __future__ import annotations

import numpy as np

from mowm.axiom_library import AxiomLibrary
from mowm.domains import available_domains, seed_all

K, L = 4, 32


class TestSecurityDomain:
    def setup_method(self):
        import mowm.domains.security  # noqa: F401
        self.lib = AxiomLibrary(k=K, l=L)

    def test_registered(self):
        assert "security" in available_domains()

    def test_seeds_axioms(self):
        seed_all(self.lib, domains=["security"])
        assert len(self.lib) >= 10

    def test_axiom_shapes(self):
        seed_all(self.lib, domains=["security"])
        for ax in self.lib.all_axioms():
            assert ax.vector.shape == (K, L)
            assert ax.vector.dtype == np.float32

    def test_domain_prefix(self):
        seed_all(self.lib, domains=["security"])
        for ax in self.lib.all_axioms():
            assert ax.domain.startswith("security")

    def test_known_axioms(self):
        seed_all(self.lib, domains=["security"])
        names = [ax.name for ax in self.lib.all_axioms()]
        assert "CIA Triad" in names
        assert "Brute Force Complexity" in names
        assert "Multi-Factor Auth" in names

    def test_deterministic(self):
        lib2 = AxiomLibrary(k=K, l=L)
        seed_all(self.lib, domains=["security"])
        seed_all(lib2, domains=["security"])
        a1 = sorted(self.lib.all_axioms(), key=lambda a: a.name)
        a2 = sorted(lib2.all_axioms(), key=lambda a: a.name)
        for x, y in zip(a1, a2):
            sim = self.lib.bc.similarity(x.vector, y.vector)
            assert sim > 0.99
