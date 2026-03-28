# tests/test_vision_projector.py
"""Tests for VisionProjector — image features to VSA space."""
from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes

from mowm.axiom_library import AxiomLibrary
from mowm.vision_projector import VisionProjector

K, L = 4, 32
VISUAL_DIM = 64  # small for tests


class TestVisionProjectorInit:
    def test_default_init(self):
        vp = VisionProjector(k=K, l=L, visual_dim=VISUAL_DIM)
        assert vp.visual_dim == VISUAL_DIM
        assert vp.vsa_dim == K * L

    def test_with_projection_matrix(self):
        P = np.random.default_rng(42).standard_normal(
            (K * L, VISUAL_DIM),
        ).astype(np.float32)
        vp = VisionProjector(k=K, l=L, projection_matrix=P)
        assert vp.visual_dim == VISUAL_DIM


class TestProject:
    def setup_method(self):
        self.vp = VisionProjector(k=K, l=L, visual_dim=VISUAL_DIM)

    def test_project_shape(self):
        features = np.random.default_rng(42).standard_normal(
            VISUAL_DIM,
        ).astype(np.float32)
        result = self.vp.project(features)
        assert result.shape == (K, L)
        assert result.dtype == np.float32

    def test_project_nonzero(self):
        features = np.random.default_rng(42).standard_normal(
            VISUAL_DIM,
        ).astype(np.float32)
        result = self.vp.project(features)
        assert np.any(result != 0)

    def test_project_deterministic(self):
        features = np.random.default_rng(42).standard_normal(
            VISUAL_DIM,
        ).astype(np.float32)
        r1 = self.vp.project(features)
        r2 = self.vp.project(features)
        np.testing.assert_array_equal(r1, r2)

    def test_project_different_inputs_differ(self):
        f1 = np.random.default_rng(42).standard_normal(
            VISUAL_DIM,
        ).astype(np.float32)
        f2 = np.random.default_rng(99).standard_normal(
            VISUAL_DIM,
        ).astype(np.float32)
        r1 = self.vp.project(f1)
        r2 = self.vp.project(f2)
        # Different inputs should produce different projections
        assert not np.allclose(r1, r2)

    def test_project_wrong_dim_raises(self):
        features = np.zeros(VISUAL_DIM + 10, dtype=np.float32)
        with pytest.raises(ValueError, match="Feature dim"):
            self.vp.project(features)


class TestProjectBatch:
    def setup_method(self):
        self.vp = VisionProjector(k=K, l=L, visual_dim=VISUAL_DIM)

    def test_batch_shape(self):
        batch = np.random.default_rng(42).standard_normal(
            (5, VISUAL_DIM),
        ).astype(np.float32)
        result = self.vp.project_batch(batch)
        assert result.shape == (5, K, L)

    def test_batch_matches_individual(self):
        rng = np.random.default_rng(42)
        batch = rng.standard_normal(
            (3, VISUAL_DIM),
        ).astype(np.float32)
        batch_result = self.vp.project_batch(batch)
        for i in range(3):
            individual = self.vp.project(batch[i])
            np.testing.assert_array_almost_equal(
                batch_result[i], individual, decimal=5,
            )


class TestFindMatchingAxioms:
    def test_find_axioms(self):
        lib = AxiomLibrary(k=K, l=L)
        lib.register(lib.make_axiom(
            "Newton's Second Law", "physics.dynamics", "F = m * a",
        ))
        lib.register(lib.make_axiom(
            "Ohm's Law", "physics.em", "V = I * R",
        ))

        vp = VisionProjector(k=K, l=L, visual_dim=VISUAL_DIM)
        features = np.random.default_rng(42).standard_normal(
            VISUAL_DIM,
        ).astype(np.float32)
        results = vp.find_matching_axioms(features, lib, top_n=2)
        assert len(results) == 2
        assert all(hasattr(ax, "name") for ax in results)


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        vp = VisionProjector(k=K, l=L, visual_dim=VISUAL_DIM)
        path = str(tmp_path / "vision_P.npy")
        vp.save_projection(path)

        vp2 = VisionProjector.from_file(path, k=K, l=L)
        features = np.random.default_rng(42).standard_normal(
            VISUAL_DIM,
        ).astype(np.float32)
        r1 = vp.project(features)
        r2 = vp2.project(features)
        np.testing.assert_array_equal(r1, r2)
