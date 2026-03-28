from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "AND", "OR", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Bayes' Theorem",
        "causal.probability",
        "P(A|B) = P(B|A) * P(A) / P(B)",
        ["*", "/"],
    ),
    (
        "Chain Rule",
        "causal.probability",
        "P(X1,...,Xn) = prod(P(Xi | X1,...,X{i-1}))",
        ["*"],
    ),
    (
        "Marginalization",
        "causal.probability",
        "P(X) = sum(P(X,Y) for all Y)",
        ["+"],
    ),
    (
        "Do-Calculus Rule 1",
        "causal.intervention",
        "P(Y|do(X), Z) = P(Y|do(X)) if Z ind Y | X in G_X",
        ["AND", "IMPLIES"],
    ),
    (
        "Do-Calculus Rule 2",
        "causal.intervention",
        "P(Y|do(X), Z) = P(Y|X,Z) if Z not descendant of X in G",
        ["AND", "IMPLIES"],
    ),
    (
        "Do-Calculus Rule 3",
        "causal.intervention",
        "P(Y|do(X), Z) = P(Y|Z) if X not ancestor of Y in G_Z",
        ["AND", "IMPLIES"],
    ),
    (
        "Signed Positive Cause",
        "causal.graph",
        "X ->+ Y: increasing X increases P(Y=1)",
        ["IMPLIES", "+"],
    ),
    (
        "Signed Negative Cause",
        "causal.graph",
        "X ->- Y: increasing X decreases P(Y=1)",
        ["IMPLIES", "-"],
    ),
    (
        "Causal Markov",
        "causal.graph",
        "Xi ind non-descendants(Xi) | parents(Xi) in DAG G",
        ["AND"],
    ),
    (
        "Faithfulness",
        "causal.graph",
        "all conditional independencies in P arise from d-separation in G",
        ["IMPLIES"],
    ),
    (
        "Counterfactual",
        "causal.probability",
        "Y_{x}(u) = Y value in world where X=x and context U=u",
        ["+", "-"],
    ),
    (
        "Average Treatment Effect",
        "causal.intervention",
        "ATE = E[Y|do(X=1)] - E[Y|do(X=0)]",
        ["-"],
    ),
    (
        "Instrumental Variable",
        "causal.discovery",
        "beta = Cov(Z,Y) / Cov(Z,X) where Z instrument for X on Y",
        ["/"],
    ),
    (
        "PC Algorithm Step",
        "causal.discovery",
        "remove edge (X,Y) if exists S: X ind Y | S",
        ["AND", "IMPLIES"],
    ),
    (
        "Granger Causality",
        "causal.discovery",
        "X Granger-causes Y if P(Y_t | Y_{t-1},..., X_{t-1},...) > P(Y_t | Y_{t-1},...)",
        ["+", ">"],
    ),
]


@register_domain("causal")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
