from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "exp", "log", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Hardy-Weinberg",
        "bio.genetics",
        "p^2 + 2*p*q + q^2 = 1",
        ["+", "*"],
    ),
    (
        "Mendelian Segregation",
        "bio.genetics",
        "P(allele A from parent) = 0.5",
        [],
    ),
    (
        "Independent Assortment",
        "bio.genetics",
        "P(A and B) = P(A) * P(B) for unlinked loci",
        ["*"],
    ),
    (
        "Hamilton's Rule",
        "bio.evolution",
        "r * B > C",
        ["*", "-"],
    ),
    (
        "Price Equation",
        "bio.evolution",
        "w * delta_z = Cov(w_i, z_i) + E(w_i * delta_z_i)",
        ["*", "+"],
    ),
    (
        "Fisher's Fundamental Theorem",
        "bio.evolution",
        "dw/dt = Var(w) / w",
        ["/"],
    ),
    (
        "Mutation-Selection Balance",
        "bio.evolution",
        "q_hat = sqrt(mu / s)",
        ["sqrt", "/"],
    ),
    (
        "DNA Complementarity",
        "bio.molecular",
        "A pairs T, C pairs G (Watson-Crick base pairing)",
        [],
    ),
    (
        "Central Dogma",
        "bio.molecular",
        "DNA -> RNA -> Protein",
        [],
    ),
    (
        "K-mer Frequency",
        "bio.molecular",
        "freq(k-mer) = count(k-mer) / (len(seq) - k + 1)",
        ["/", "+", "-"],
    ),
    (
        "Michaelis-Menten",
        "bio.biochem",
        "v = V_max * [S] / (K_m + [S])",
        ["*", "/", "+"],
    ),
    (
        "Lotka-Volterra Prey",
        "bio.ecology",
        "dx/dt = alpha*x - beta*x*y",
        ["*", "-"],
    ),
    (
        "Lotka-Volterra Predator",
        "bio.ecology",
        "dy/dt = delta*x*y - gamma*y",
        ["*", "-"],
    ),
    (
        "Logistic Growth",
        "bio.ecology",
        "dN/dt = r*N*(1 - N/K)",
        ["*", "-", "/"],
    ),
    (
        "Shannon Biodiversity",
        "bio.ecology",
        "H = -sum(p_i * log(p_i))",
        ["-", "*", "log"],
    ),
]


@register_domain("bio")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
