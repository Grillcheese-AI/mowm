from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "exp", "log"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Self-Consistency",
        "identity.core",
        "belief(t+1) consistent with belief(t) unless evidence E exceeds threshold",
        ["-"],
    ),
    (
        "Trait Stability",
        "identity.personality",
        "trait(t) = trait(0) * exp(-lambda * t) + trait_base * (1 - exp(-lambda * t))",
        ["*", "exp", "-", "+"],
    ),
    (
        "Value Hierarchy",
        "identity.values",
        "priority(v_i) > priority(v_j) IMPLIES choose(v_i) when conflict",
        [">"],
    ),
    (
        "Emotional Baseline",
        "identity.affect",
        "affect(t) = affect_set_point + transient(t)",
        ["+"],
    ),
    (
        "Plutchik Primary",
        "identity.affect",
        "emotion = weighted_sum(joy, trust, fear, surprise, sadness, disgust, anger, anticipation)",
        ["+", "*"],
    ),
    (
        "IQ Calibration",
        "identity.cognitive",
        "score(t+1) = score(t) + alpha * (observed - predicted)",
        ["+", "-", "*"],
    ),
    (
        "Memory Strength",
        "identity.memory",
        "m(t) = m_0 * exp(-t / tau)",
        ["*", "exp", "/", "-"],
    ),
    (
        "Habit Formation",
        "identity.behavior",
        "P(habit) = 1 - exp(-n / N_0)",
        ["-", "exp", "/"],
    ),
    (
        "Lineage Ledger",
        "identity.lineage",
        "lineage_score = sum(contribution_i * recency_weight_i)",
        ["+", "*"],
    ),
    (
        "Growth Mindset",
        "identity.core",
        "capability(t+dt) = capability(t) + effort * learning_rate",
        ["+", "*"],
    ),
]


@register_domain("identity")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
