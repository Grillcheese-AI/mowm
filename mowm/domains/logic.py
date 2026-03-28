from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["AND", "OR", "NOT", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Modus Ponens",
        "logic.deductive",
        "(P AND (P IMPLIES Q)) IMPLIES Q",
        ["AND", "IMPLIES"],
    ),
    (
        "Modus Tollens",
        "logic.deductive",
        "((P IMPLIES Q) AND NOT Q) IMPLIES NOT P",
        ["AND", "IMPLIES", "NOT"],
    ),
    (
        "Hypothetical Syllogism",
        "logic.deductive",
        "((P IMPLIES Q) AND (Q IMPLIES R)) IMPLIES (P IMPLIES R)",
        ["AND", "IMPLIES"],
    ),
    (
        "Disjunctive Syllogism",
        "logic.deductive",
        "((P OR Q) AND NOT P) IMPLIES Q",
        ["OR", "AND", "NOT", "IMPLIES"],
    ),
    (
        "De Morgan 1",
        "logic.propositional",
        "NOT (P AND Q) = (NOT P) OR (NOT Q)",
        ["NOT", "AND", "OR"],
    ),
    (
        "De Morgan 2",
        "logic.propositional",
        "NOT (P OR Q) = (NOT P) AND (NOT Q)",
        ["NOT", "OR", "AND"],
    ),
    (
        "Double Negation",
        "logic.propositional",
        "NOT (NOT P) = P",
        ["NOT"],
    ),
    (
        "Contrapositive",
        "logic.propositional",
        "(P IMPLIES Q) = (NOT Q IMPLIES NOT P)",
        ["IMPLIES", "NOT"],
    ),
    (
        "Law of Excluded Middle",
        "logic.propositional",
        "P OR NOT P = TRUE",
        ["OR", "NOT"],
    ),
    (
        "Deontic Obligation",
        "logic.deontic",
        "O(P) IMPLIES P is obligatory",
        ["IMPLIES"],
    ),
    (
        "Deontic Permission",
        "logic.deontic",
        "P(P) = NOT O(NOT P)",
        ["NOT"],
    ),
    (
        "Modal Necessity",
        "logic.modal",
        "BOX P IMPLIES P",
        ["IMPLIES"],
    ),
    (
        "Modal Possibility",
        "logic.modal",
        "DIAMOND P = NOT BOX NOT P",
        ["NOT"],
    ),
    (
        "Modal K Axiom",
        "logic.modal",
        "BOX (P IMPLIES Q) IMPLIES (BOX P IMPLIES BOX Q)",
        ["IMPLIES"],
    ),
    (
        "Distribution",
        "logic.propositional",
        "P AND (Q OR R) = (P AND Q) OR (P AND R)",
        ["AND", "OR"],
    ),
]


@register_domain("logic")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
