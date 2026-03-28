from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "*", "/", "sqrt", "log"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "PageRank",
        "social.graph",
        "PR(u) = (1-d)/N + d * sum(PR(v)/L(v) for v in B_u)",
        ["+", "*", "/"],
    ),
    (
        "Degree Centrality",
        "social.graph",
        "C_D(v) = deg(v) / (n - 1)",
        ["/"],
    ),
    (
        "Betweenness Centrality",
        "social.graph",
        "C_B(v) = sum(sigma_st(v) / sigma_st for s != t != v)",
        ["/", "+"],
    ),
    (
        "Closeness Centrality",
        "social.graph",
        "C_C(v) = (n-1) / sum(d(v, u) for u != v)",
        ["-", "/", "+"],
    ),
    (
        "Eigenvector Centrality",
        "social.graph",
        "x_v = (1/lambda) * sum(x_u for u in N(v))",
        ["*", "/", "+"],
    ),
    (
        "Network Diffusion",
        "social.dynamics",
        "dI/dt = beta * S * I - gamma * I",
        ["*", "-", "+"],
    ),
    (
        "SIR Model",
        "social.dynamics",
        "dS/dt = -beta*S*I, dI/dt = beta*S*I - gamma*I, dR/dt = gamma*I",
        ["*", "-", "+"],
    ),
    (
        "Homophily",
        "social.formation",
        "P(edge(u,v)) proportional to sim(u, v)",
        ["*", "/"],
    ),
    (
        "Preferential Attachment",
        "social.formation",
        "P(connect to v) = k_v / sum(k_i)",
        ["/", "+"],
    ),
    (
        "Granovetter Strength",
        "social.formation",
        "strength(A,B) = overlap(N(A), N(B)) / union(N(A), N(B))",
        ["/"],
    ),
    (
        "Clustering Coefficient",
        "social.graph",
        "C(v) = 2*t(v) / (k_v * (k_v - 1))",
        ["*", "/", "-"],
    ),
]


@register_domain("social")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
