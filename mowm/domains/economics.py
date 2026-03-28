from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "log", "exp", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Supply-Demand Equilibrium",
        "economics.micro",
        "Q_d(P*) = Q_s(P*)",
        ["+", "-"],
    ),
    (
        "Price Elasticity",
        "economics.micro",
        "epsilon = (dQ/Q) / (dP/P)",
        ["/"],
    ),
    (
        "Utility Maximization",
        "economics.micro",
        "max U(x) subject to P*x = I",
        ["*", "+"],
    ),
    (
        "Marginal Utility",
        "economics.micro",
        "MU_i = dU / dx_i",
        ["/"],
    ),
    (
        "Consumer Surplus",
        "economics.micro",
        "CS = integral(D(Q), 0, Q*) - P* * Q*",
        ["-", "*"],
    ),
    (
        "Nash Equilibrium",
        "economics.game_theory",
        "u_i(s_i*, s_{-i}*) >= u_i(s_i, s_{-i}*) for all s_i",
        ["+", "-"],
    ),
    (
        "Pareto Optimality",
        "economics.game_theory",
        "no s' such that u_i(s') >= u_i(s) for all i with strict for some",
        ["+"],
    ),
    (
        "Mixed Strategy Nash",
        "economics.game_theory",
        "E[u_i | sigma_i*] = E[u_i | sigma_i] for all sigma_i in support",
        ["*", "+"],
    ),
    (
        "Fisher Equation",
        "economics.macro",
        "i = r + pi",
        ["+"],
    ),
    (
        "Quantity Theory",
        "economics.macro",
        "M * V = P * Y",
        ["*"],
    ),
    (
        "GDP Identity",
        "economics.macro",
        "Y = C + I + G + NX",
        ["+"],
    ),
    (
        "Phillips Curve",
        "economics.macro",
        "pi = pi_e - alpha * (u - u_n)",
        ["-", "*"],
    ),
    (
        "IS-LM Equilibrium",
        "economics.macro",
        "Y = C(Y-T) + I(r) + G and M/P = L(Y,r)",
        ["+", "-", "/"],
    ),
    (
        "Cobb-Douglas Production",
        "economics.macro",
        "Y = A * K^alpha * L^(1-alpha)",
        ["*", "exp", "-"],
    ),
    (
        "Present Value",
        "economics.finance",
        "PV = FV / (1 + r)^t",
        ["+", "/", "exp"],
    ),
]


@register_domain("economics")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
