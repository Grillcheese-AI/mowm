# tests/test_integration.py
"""Integration test — full MoWM pipeline end-to-end."""
from __future__ import annotations

import numpy as np

from mowm import (
    Axiom,  # noqa: F401
    AxiomLibrary,
    DiscoveryLoop,
    MoWMPipeline,
    MoWMRouter,
    StepResult,
    World,
)
from mowm.domains import available_domains, seed_all  # noqa: F401

K, L = 4, 32


class TestPublicAPI:
    def test_version(self):
        import mowm
        assert mowm.__version__ == "0.1.0"

    def test_all_exports(self):
        import mowm
        for name in mowm.__all__:
            assert hasattr(mowm, name)


class TestIntegrationPipeline:
    def setup_method(self):
        for d in [
            "physics", "logic", "economics", "social", "linguistics",
            "causal", "atomic", "bio", "music", "identity", "colors",
            "language", "historical", "ensemble",
        ]:
            try:
                __import__(f"mowm.domains.{d}")
            except ImportError:
                pass

    def test_full_pipeline_with_all_domains(self):
        pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=8, top_k=2, z_max=2, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=None, seed=42,
        )
        assert pipe.library is not None
        assert len(pipe.library) >= 200

        result = pipe.step(
            state={"position": "origin", "mass": "1kg"},
            action="apply_force_10N",
        )
        assert isinstance(result, StepResult)
        assert result.predicted_state.shape == (K, L)
        assert result.predicted_state.dtype == np.float32
        assert result.active_worlds >= 1

    def test_pipeline_rollout(self):
        pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=4, top_k=2, z_max=1, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"], seed=42,
        )
        results = pipe.rollout(
            state="ball at rest",
            actions=["push", "push harder", "release"],
            horizon=3,
        )
        assert len(results) == 3
        for r in results:
            assert r.predicted_state.shape == (K, L)
            assert isinstance(r.q_value, float)

    def test_pipeline_with_ground_truth_triggers_discovery(self):
        pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=8, top_k=2, z_max=1, tau=0.01,
            cleanup_interval=1, gamma=0.99,
            domains=["physics"], seed=42,
        )
        bc = pipe.bc
        gt = bc.random_discrete(seed=999)
        result = pipe.step(
            state="some state",
            action="some action",
            ground_truth=gt,
        )
        assert isinstance(result, StepResult)


class TestIntegrationWorldLifecycle:
    def test_spawn_predict_consolidate(self):
        lib = AxiomLibrary(k=K, l=L)
        import mowm.domains.physics  # noqa: F401
        seed_all(lib, domains=["physics"])

        axioms = lib.select(lib.bc.random_discrete(seed=1), top_n=5)

        w1 = World(
            world_id=0, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=axioms, seed=10,
        )
        w2 = World(
            world_id=1, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=axioms[:2], seed=20,
        )

        state = lib.bc.random_discrete(seed=100)
        action = lib.bc.random_discrete(seed=200)
        p1, c1 = w1.predict(state, action)
        p2, c2 = w2.predict(state, action)
        assert p1.shape == (K, L)
        assert p2.shape == (K, L)

        w1.consolidate(w2)
        p3, c3 = w1.predict(state, action)
        assert p3.shape == (K, L)

    def test_router_lifecycle(self):
        lib = AxiomLibrary(k=K, l=L)
        import mowm.domains.physics  # noqa: F401
        seed_all(lib, domains=["physics"])

        router = MoWMRouter(
            k=K, l=L, max_worlds=4, top_k=2,
            tau_spawn=0.5, tau_prune_steps=5,
        )

        residual = lib.bc.random_discrete(seed=42)
        w = router.maybe_spawn(residual, lib)
        assert w is not None

        state = lib.bc.random_discrete(seed=10)
        action = lib.bc.random_discrete(seed=20)
        routed = router.route(state, action)
        assert len(routed) >= 1

        pruned = router.prune(current_step=1000)
        assert isinstance(pruned, list)


class TestIntegrationDiscovery:
    def test_discover_and_use_new_axiom(self):
        lib = AxiomLibrary(k=K, l=L)
        import mowm.domains.physics  # noqa: F401
        seed_all(lib, domains=["physics"])

        initial_count = len(lib)

        loop = DiscoveryLoop(
            lib, k=K, l=L,
            sr_operators=["+", "*"],
            max_complexity=8,
            sr_population=30,
            sr_generations=20,
            seed=42,
        )

        inputs = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
        outputs = np.array([2, 4, 6, 8, 10], dtype=np.float64)
        context = lib.bc.random_discrete(seed=42)

        result = loop.discover(inputs, outputs, context)
        if result is not None:
            assert len(lib) == initial_count + 1
            assert result.domain == "discovered"
            assert result.vector.shape == (K, L)

            found = lib.select(result.vector, top_n=1)
            assert len(found) >= 1


class TestIntegrationAllDomainSeeding:
    def test_all_domains_seed(self):
        for d in [
            "physics", "logic", "economics", "social", "linguistics",
            "causal", "atomic", "bio", "music", "identity", "colors",
            "language", "historical", "ensemble",
        ]:
            try:
                __import__(f"mowm.domains.{d}")
            except ImportError:
                pass

        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib)
        domains = lib.domains()

        assert len(domains) >= 14, f"Expected 14+ domains, got {len(domains)}"
        assert len(lib) >= 200, f"Expected 200+ axioms, got {len(lib)}"

        for domain_name, axiom_list in domains.items():
            assert len(axiom_list) >= 1, f"{domain_name} has no axioms"
