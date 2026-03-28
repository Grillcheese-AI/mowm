from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "AND", "OR", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Temporal Before",
        "historical.logic",
        "event_A before event_B IFF t(A) < t(B)",
        ["-", "IMPLIES"],
    ),
    (
        "Temporal After",
        "historical.logic",
        "event_A after event_B IFF t(A) > t(B)",
        ["-", "IMPLIES"],
    ),
    (
        "Temporal During",
        "historical.logic",
        "event_A during period_P IFF t_start(P) <= t(A) <= t_end(P)",
        ["AND"],
    ),
    (
        "Temporal Overlap",
        "historical.logic",
        "period_A overlaps period_B IFF t_start(A) < t_end(B) AND t_start(B) < t_end(A)",
        ["AND", "-"],
    ),
    (
        "Event Causation",
        "historical.causal",
        "event_A causes event_B IFF A before B AND P(B|A) > P(B|not A)",
        ["AND", "IMPLIES", "-"],
    ),
    (
        "Periodicity",
        "historical.pattern",
        "period(event_type) = mean(t_{i+1} - t_i) for recurrent events",
        ["-", "+", "/"],
    ),
    (
        "Epoch Binding",
        "historical.encoding",
        "epoch_vector = sum(event_vector_i * recency_weight_i for i in epoch)",
        ["+", "*"],
    ),
    (
        "Historical Analogy",
        "historical.reasoning",
        "sim(epoch_A, epoch_B) > threshold IMPLIES analogous patterns",
        ["IMPLIES"],
    ),
    (
        "Counterfactual History",
        "historical.reasoning",
        "CF(event_A removed) = trajectory without A given same initial conditions",
        ["-"],
    ),
    (
        "Civilizational Cycle",
        "historical.pattern",
        "T_cycle approx 250 years: rise + peak + decline + fall",
        ["+"],
    ),
    (
        "Technology S-Curve",
        "historical.statistics",
        "adoption(t) = K / (1 + exp(-r*(t - t0)))",
        ["/", "+", "exp", "-", "*"],
    ),
    (
        "Generational Period",
        "historical.statistics",
        "generation_span approx 25 years; T_gen = birth_year_child - birth_year_parent",
        ["-"],
    ),
    (
        "Empire Duration",
        "historical.statistics",
        "mean_empire_lifespan approx 250 years; sd approx 150 years",
        ["+", "-"],
    ),
    (
        "Demographic Transition",
        "historical.dynamics",
        "stage: high birth+death -> high birth+low death -> low birth+low death",
        ["-", "+"],
    ),
    (
        "Power Transition",
        "historical.dynamics",
        "conflict_risk = f(power_parity(A,B)) peaks when relative_power(A,B) approx 1",
        ["/", "-"],
    ),
]


@register_domain("historical")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
