from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "*", "AND", "OR", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Morpheme Composition",
        "language.morphology",
        "word = root + affix_1 + ... + affix_n",
        ["+"],
    ),
    (
        "Inflectional Paradigm",
        "language.morphology",
        "paradigm(lexeme) = {stem + infl_suffix for infl in morphosyntax}",
        ["+"],
    ),
    (
        "Derivational Rule",
        "language.morphology",
        "derived_word = base + derivational_affix -> new_category",
        ["+", "IMPLIES"],
    ),
    (
        "Constituency",
        "language.syntax",
        "XP = head(X) + complements + specifier",
        ["+"],
    ),
    (
        "Merge Operation",
        "language.syntax",
        "Merge(alpha, beta) = {alpha, {alpha, beta}}",
        ["+"],
    ),
    (
        "Case Assignment",
        "language.syntax",
        "NP gets Case from governing head under agreement",
        ["AND"],
    ),
    (
        "Theta Role Assignment",
        "language.syntax",
        "verb assigns theta roles to arguments: agent, patient, goal, theme",
        ["+"],
    ),
    (
        "Compositionality",
        "language.semantics",
        "meaning(XP) = f(meaning(X), meaning(complement))",
        ["*"],
    ),
    (
        "Lambda Calculus",
        "language.semantics",
        "[[VP]]([[NP]]) = lambda x . P(x) applied to individual a",
        ["*", "IMPLIES"],
    ),
    (
        "Distributional Hypothesis",
        "language.distributional",
        "meaning(w) approx context_vector(w) = sum(PMI(w,c)*e_c)",
        ["+", "*"],
    ),
    (
        "PMI",
        "language.distributional",
        "PMI(w,c) = log(P(w,c) / (P(w)*P(c)))",
        ["*", "/"],
    ),
    (
        "Grice Maxim Quality",
        "language.pragmatics",
        "assert P only if believe P AND have evidence for P",
        ["AND"],
    ),
    (
        "Grice Maxim Quantity",
        "language.pragmatics",
        "be as informative as required AND no more informative than required",
        ["AND"],
    ),
    (
        "Speech Act",
        "language.pragmatics",
        "utterance U performs illocutionary act F with propositional content P",
        ["AND"],
    ),
    (
        "Zipf's Law",
        "language.distributional",
        "freq(word_r) proportional to 1/r",
        ["/"],
    ),
]


@register_domain("language")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
