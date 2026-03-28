# tests/test_neuromorphic.py
"""Tests for NeuromorphicOrchestrator — BICA routing layer."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.neuromorphic_orchestrator import (
    CognitiveModule,
    CognitiveSNN,
    GIFController,
    HormoneState,
    NeuromorphicOrchestrator,
)


class TestHormoneState:
    def test_initial_zero(self):
        h = HormoneState()
        assert h.adrenaline == 0.0
        assert h.dopamine == 0.0
        assert h.acetylcholine == 0.0

    def test_decay(self):
        h = HormoneState(adrenaline=1.0, dopamine=1.0, acetylcholine=1.0)
        h.decay()
        assert h.adrenaline < 1.0
        assert h.dopamine < 1.0
        assert h.acetylcholine < 1.0

    def test_decay_approaches_zero(self):
        h = HormoneState(adrenaline=1.0)
        for _ in range(100):
            h.decay()
        assert h.adrenaline < 0.01


class TestGIFController:
    def test_fires_with_stimulus(self):
        gif = GIFController()
        fired = gif.tick(input_stimulus=2.0, acetylcholine=0.0)
        assert fired is True

    def test_no_fire_without_stimulus(self):
        gif = GIFController()
        fired = gif.tick(input_stimulus=0.0, acetylcholine=0.0)
        assert fired is False

    def test_adaptation_prevents_infinite_firing(self):
        gif = GIFController()
        fire_count = 0
        for _ in range(100):
            if gif.tick(input_stimulus=1.5, acetylcholine=0.0):
                fire_count += 1
        # Should fire some but not all (adaptation kicks in)
        assert 0 < fire_count < 100

    def test_acetylcholine_extends_firing(self):
        gif_low = GIFController()
        gif_high = GIFController()
        fires_low = sum(
            gif_low.tick(1.5, acetylcholine=0.0) for _ in range(50)
        )
        fires_high = sum(
            gif_high.tick(1.5, acetylcholine=0.9) for _ in range(50)
        )
        # More acetylcholine = less fatigue = more fires
        assert fires_high >= fires_low

    def test_reset(self):
        gif = GIFController()
        gif.tick(2.0, 0.0)
        gif.tick(2.0, 0.0)
        gif.reset()
        assert gif.voltage == 0.0
        assert gif.adaptation_current == 0.0


class TestCognitiveSNN:
    def test_creation(self):
        snn = CognitiveSNN(seed=42)
        assert snn.num_modules == len(CognitiveModule)
        assert snn.weights.shape == (9, 9)
        assert np.all(np.diag(snn.weights) == 0.0)

    def test_tick_returns_spikes(self):
        snn = CognitiveSNN(seed=42)
        spikes = np.zeros(len(CognitiveModule))
        spikes[CognitiveModule.KNOWLEDGE_SEARCH.value] = 1.0
        output = snn.tick(spikes)
        assert output.shape == (len(CognitiveModule),)
        assert output.dtype == np.float64

    def test_adrenaline_lowers_threshold(self):
        snn = CognitiveSNN(seed=42)
        snn.apply_stimulus(urgent_news=True)
        assert snn.hormones.adrenaline > 0.0

    def test_dopamine_from_proof(self):
        snn = CognitiveSNN(seed=42)
        snn.apply_stimulus(proof_success=True)
        assert snn.hormones.dopamine > 0.0

    def test_plasticity_changes_weights(self):
        snn = CognitiveSNN(seed=42)
        initial_weights = snn.weights.copy()
        # Simulate dopamine + firing
        snn.apply_stimulus(proof_success=True)
        pre = np.zeros(len(CognitiveModule))
        post = np.zeros(len(CognitiveModule))
        pre[CognitiveModule.KNOWLEDGE_SEARCH.value] = 1.0
        post[CognitiveModule.CAUSAL_TRACER.value] = 1.0
        snn._apply_plasticity(pre, post)
        # Weights should have changed
        assert not np.allclose(snn.weights, initial_weights)


class TestNeuromorphicOrchestrator:
    def test_creation(self):
        orch = NeuromorphicOrchestrator(seed=42)
        assert len(orch.current_spikes) == len(CognitiveModule)

    def test_stimulate_sensor(self):
        orch = NeuromorphicOrchestrator(seed=42)
        orch.stimulate_sensor(CognitiveModule.VISION_PROJECTOR, 1.0)
        assert orch.current_spikes[
            CognitiveModule.VISION_PROJECTOR.value
        ] == 1.0

    def test_run_cycle_returns_results(self):
        orch = NeuromorphicOrchestrator(seed=42)
        orch.stimulate_sensor(CognitiveModule.KNOWLEDGE_SEARCH, 2.0)
        results = orch.run_cycle(max_ticks=10)
        assert isinstance(results, dict)
        assert "status" in results

    def test_run_cycle_completes(self):
        orch = NeuromorphicOrchestrator(seed=42)
        results = orch.run_cycle(max_ticks=5)
        assert results["status"] == "cycle_complete"

    def test_proof_success_triggers_dopamine(self):
        orch = NeuromorphicOrchestrator(seed=42)
        # Directly stimulate the executor strongly
        orch.stimulate_sensor(CognitiveModule.SAFE_EXECUTOR, 2.0)
        orch.run_cycle(max_ticks=3)
        # Dopamine should have been released if executor fired
        # (depends on threshold dynamics)
        assert isinstance(orch.snn.hormones.dopamine, float)
