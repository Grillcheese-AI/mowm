from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "*", "AND", "OR"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "S -> NP VP",
        "linguistics.cfg",
        "S = NP + VP",
        ["+"],
    ),
    (
        "NP -> Det N",
        "linguistics.cfg",
        "NP = Det + N",
        ["+"],
    ),
    (
        "VP -> V NP",
        "linguistics.cfg",
        "VP = V + NP",
        ["+"],
    ),
    (
        "VP -> V NP PP",
        "linguistics.cfg",
        "VP = V + NP + PP",
        ["+"],
    ),
    (
        "PP -> P NP",
        "linguistics.cfg",
        "PP = P + NP",
        ["+"],
    ),
    (
        "X-bar Schema",
        "linguistics.syntax",
        "XP = Spec + X-bar, X-bar = X + Complement",
        ["+"],
    ),
    (
        "Head Parameter",
        "linguistics.syntax",
        "X-bar = X + Complement OR X-bar = Complement + X",
        ["+", "OR"],
    ),
    (
        "Dependency Arc",
        "linguistics.dependency",
        "arc(head, dependent) with label rel",
        ["+"],
    ),
    (
        "Projectivity",
        "linguistics.dependency",
        "for arc(h,d): no word w between h,d has head outside interval(h,d)",
        ["AND"],
    ),
    (
        "Theta Criterion",
        "linguistics.syntax",
        "each argument receives exactly one theta role AND each role assigned once",
        ["AND"],
    ),
    (
        "Binding Principle A",
        "linguistics.syntax",
        "anaphor must be bound in its local binding domain",
        ["AND"],
    ),
    (
        "Binding Principle B",
        "linguistics.syntax",
        "pronoun must be free in its local binding domain",
        ["AND"],
    ),
]


@register_domain("linguistics")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
