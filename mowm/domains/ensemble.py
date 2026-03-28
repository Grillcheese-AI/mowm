from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "exp", "log", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Population Superposition",
        "ensemble.vsa",
        "V_pop = sum(V_i) / N",
        ["+", "/"],
    ),
    (
        "Mean-Field Approximation",
        "ensemble.mf",
        "E[f(x_i, x_j)] approx f(E[x_i], E[x_j]) when correlations weak",
        ["*", "+"],
    ),
    (
        "Mean-Field Nash",
        "ensemble.mf",
        "mu* satisfies: s*(x,mu*) is optimal for agent x given distribution mu*",
        ["*"],
    ),
    (
        "Boltzmann Distribution",
        "ensemble.stat",
        "P(state_i) = exp(-E_i / (k_B*T)) / Z",
        ["exp", "-", "*", "/"],
    ),
    (
        "Phase Transition",
        "ensemble.stat",
        "order_param(T) = 0 for T>T_c, nonzero for T<T_c",
        ["-"],
    ),
    (
        "Critical Exponent",
        "ensemble.stat",
        "m ~ |T - T_c|^beta near critical point",
        ["-", "exp"],
    ),
    (
        "Fractal Causal Graph",
        "ensemble.fractal",
        "dim_H = lim(log N(epsilon) / log(1/epsilon)) as epsilon->0",
        ["/", "log"],
    ),
    (
        "Self-Similarity",
        "ensemble.fractal",
        "S(lambda*r) = lambda^d * S(r) for fractal S",
        ["*", "exp"],
    ),
    (
        "Renormalization Group",
        "ensemble.fractal",
        "R(H) = H' where H' is effective Hamiltonian at coarser scale",
        ["+", "*"],
    ),
    (
        "Wisdom of Crowds",
        "ensemble.collective",
        "E[(mean(X_i) - mu)^2] = sigma^2 / N + bias^2",
        ["/", "+"],
    ),
    (
        "Condorcet Jury Theorem",
        "ensemble.collective",
        "P(majority correct) -> 1 as N->inf when p_individual > 0.5",
        ["*", "+"],
    ),
    (
        "Social Choice Arrow",
        "ensemble.collective",
        "no social welfare function satisfies unanimity + IIA + non-dictatorship",
        ["+"],
    ),
    (
        "Voter Model",
        "ensemble.dynamics",
        "state_i(t+1) = state_j(t) where j chosen uniformly from N(i)",
        ["+"],
    ),
    (
        "Schelling Segregation",
        "ensemble.dynamics",
        "agent_i moves if fraction_similar(N(i)) < threshold_i",
        ["/", "-"],
    ),
    (
        "Emergent Behavior",
        "ensemble.dynamics",
        "macro_pattern not predictable from individual rules alone",
        ["+"],
    ),
]


@register_domain("ensemble")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
