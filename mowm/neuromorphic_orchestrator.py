# mowm/neuromorphic_orchestrator.py
"""NeuromorphicOrchestrator — Biologically Inspired Cognitive Architecture (BICA).

Replaces standard procedural orchestration with a Spiking Neural Network (SNN).
Uses LIF neurons for module routing, GIF neurons for cycle-breaking in graph
traversal, STDP + Oja's rule for learning cognitive pathways, and artificial
hormones for global hyperparameter modulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class CognitiveModule(IntEnum):
    """Indices for the structural brain regions (modules)."""
    VISION_PROJECTOR = 0
    NEWS_INJECTOR = 1
    KNOWLEDGE_SEARCH = 2
    CAUSAL_TRACER = 3
    CHAIN_BUILDER = 4
    SAFE_EXECUTOR = 5
    COMPOSER = 6
    LANGUAGE_REALIZER = 7
    VOCAL_TRACT = 8


@dataclass
class HormoneState:
    """Global neuromodulators that alter the physics of the SNN."""
    adrenaline: float = 0.0
    dopamine: float = 0.0
    acetylcholine: float = 0.0

    def decay(self) -> None:
        """Hormones naturally metabolize and decay over time."""
        self.adrenaline *= 0.90
        self.dopamine *= 0.85
        self.acetylcholine *= 0.95


class GIFController:
    """Generalized Integrate-and-Fire neuron with spike-frequency adaptation.

    Acts as a biological circuit breaker. If a module (like the CausalTracer)
    gets stuck in an infinite logical loop, the adaptation current (fatigue)
    builds up until it suppresses firing, naturally breaking the loop.
    """

    def __init__(self) -> None:
        self.voltage = 0.0
        self.adaptation_current = 0.0

        self.threshold = 1.0
        self.leak = 0.1
        self.adaptation_decay = 0.05
        self.base_fatigue_increment = 0.25

    def tick(self, input_stimulus: float, acetylcholine: float) -> bool:
        """Advance the neuron by one time-step. Returns True if it fires."""
        current_increment = self.base_fatigue_increment * (
            1.0 - (acetylcholine * 0.5)
        )

        # Decay fatigue
        self.adaptation_current -= (
            self.adaptation_current * self.adaptation_decay
        )

        # Integrate voltage
        dv = (
            -self.leak * self.voltage
            + input_stimulus
            - self.adaptation_current
        )
        self.voltage = max(0.0, self.voltage + dv)

        # Fire and adapt
        if self.voltage >= self.threshold:
            self.voltage = 0.0
            self.adaptation_current += current_increment
            return True

        return False

    def reset(self) -> None:
        """Clear memory for a new task."""
        self.voltage = 0.0
        self.adaptation_current = 0.0


class CognitiveSNN:
    """The Spiking routing matrix that dictates the engine's flow."""

    def __init__(
        self, num_modules: int = len(CognitiveModule), seed: int = 42,
    ) -> None:
        self.num_modules = num_modules
        self.hormones = HormoneState()
        rng = np.random.default_rng(seed)

        # Synaptic Weights [pre_synaptic, post_synaptic]
        self.weights = rng.uniform(0.1, 0.3, (num_modules, num_modules))
        np.fill_diagonal(self.weights, 0.0)

        # LIF Neuron states
        self.membrane_potentials = np.zeros(num_modules)
        self.base_threshold = 1.0
        self.leak_rate = 0.15

        # Specialized GIF controller for graph traversal
        self.tracer_gif = GIFController()

    def apply_stimulus(
        self,
        urgent_news: bool = False,
        proof_success: bool = False,
        novel_vision: bool = False,
    ) -> None:
        """Update hormonal state based on environmental context."""
        if urgent_news:
            self.hormones.adrenaline = min(
                1.0, self.hormones.adrenaline + 0.6,
            )
        if proof_success:
            self.hormones.dopamine = min(
                1.0, self.hormones.dopamine + 0.8,
            )
        if novel_vision:
            self.hormones.acetylcholine = min(
                1.0, self.hormones.acetylcholine + 0.5,
            )

    def tick(self, input_spikes: np.ndarray) -> np.ndarray:
        """Advance the SNN dynamics. Returns which modules fire."""
        self.hormones.decay()

        # 1. LIF Integration
        excitation = np.dot(input_spikes, self.weights)
        self.membrane_potentials += excitation
        self.membrane_potentials -= self.leak_rate
        self.membrane_potentials = np.clip(
            self.membrane_potentials, 0.0, 2.0,
        )

        # 2. Hormonal threshold modulation
        current_threshold = self.base_threshold - (
            self.hormones.adrenaline * 0.4
        )

        # 3. Fire spikes (except tracer, which uses GIF)
        output_spikes = (
            self.membrane_potentials >= current_threshold
        ).astype(float)

        # Special GIF logic for the CausalTracer
        tracer_idx = CognitiveModule.CAUSAL_TRACER.value
        tracer_stimulus = self.membrane_potentials[tracer_idx]
        if self.tracer_gif.tick(
            tracer_stimulus, self.hormones.acetylcholine,
        ):
            output_spikes[tracer_idx] = 1.0
        else:
            output_spikes[tracer_idx] = 0.0

        # Reset fired LIF neurons
        self.membrane_potentials[output_spikes > 0] = 0.0

        # 4. Plasticity: Oja's Rule modulated by Dopamine
        self._apply_plasticity(input_spikes, output_spikes)

        return output_spikes

    def _apply_plasticity(
        self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
    ) -> None:
        """Hebbian learning stabilized by Oja's rule."""
        eta = 0.05 * (1.0 + self.hormones.dopamine)

        for i in range(self.num_modules):
            for j in range(self.num_modules):
                if i == j:
                    continue
                y = post_spikes[j]
                if y > 0:
                    x = pre_spikes[i]
                    w = self.weights[i, j]
                    # Oja's Update: dw = eta * y * (x - y * w)
                    dw = eta * y * (x - y * w)
                    self.weights[i, j] = np.clip(w + dw, 0.0, 1.0)


class NeuromorphicOrchestrator:
    """Wraps the CognitiveSNN to orchestrate Python module execution.

    Instead of rigid if/else routing, inputs are dropped onto sensory
    neurons. The SNN integrates voltage, passes through learned weights,
    and triggers modules in sequence. Successful proofs release Dopamine,
    permanently strengthening the reasoning pathway via Oja's rule.
    """

    def __init__(
        self,
        modules_dict: dict[CognitiveModule, Any] | None = None,
        seed: int = 42,
    ) -> None:
        self.snn = CognitiveSNN(seed=seed)
        self.modules = modules_dict or {}
        self.current_spikes = np.zeros(len(CognitiveModule))

    def stimulate_sensor(
        self, module: CognitiveModule, intensity: float = 1.0,
    ) -> None:
        """Trigger an input sensor."""
        self.current_spikes[module.value] = intensity

    def run_cycle(
        self,
        context_data: dict[str, Any] | None = None,
        max_ticks: int = 50,
    ) -> dict[str, Any]:
        """Run the cognitive loop until conclusion or max_ticks."""
        results: dict[str, Any] = {}
        context = context_data or {}

        for tick in range(max_ticks):
            firing_modules = self.snn.tick(self.current_spikes)

            # Reset input spikes for next tick
            self.current_spikes = np.zeros(len(CognitiveModule))

            # Execute Python logic for any module that spiked
            for idx, spiked in enumerate(firing_modules):
                if spiked > 0:
                    mod_enum = CognitiveModule(idx)

                    if mod_enum == CognitiveModule.KNOWLEDGE_SEARCH:
                        self.current_spikes[
                            CognitiveModule.CAUSAL_TRACER.value
                        ] = 1.0
                        self.current_spikes[
                            CognitiveModule.CHAIN_BUILDER.value
                        ] = 1.0

                    elif mod_enum == CognitiveModule.SAFE_EXECUTOR:
                        self.snn.apply_stimulus(proof_success=True)
                        self.current_spikes[
                            CognitiveModule.COMPOSER.value
                        ] = 1.0

                    elif mod_enum == CognitiveModule.VOCAL_TRACT:
                        results["status"] = "speech_synthesized"
                        return results

            # If nothing is firing, chain is complete
            if (
                np.sum(firing_modules) == 0
                and np.sum(self.current_spikes) == 0
            ):
                break

        self.snn.tracer_gif.reset()
        results.setdefault("status", "cycle_complete")
        return results
