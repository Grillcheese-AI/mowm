from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "sin", "cos", "exp", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Schrodinger Equation",
        "atomic.qm",
        "i*hbar * d(psi)/dt = H * psi",
        ["*", "/"],
    ),
    (
        "Time-Independent Schrodinger",
        "atomic.qm",
        "H * psi = E * psi",
        ["*"],
    ),
    (
        "Heisenberg Uncertainty",
        "atomic.qm",
        "delta_x * delta_p >= hbar / 2",
        ["*", "/"],
    ),
    (
        "Pauli Exclusion",
        "atomic.qm",
        "no two fermions share the same quantum state (n, l, m_l, m_s)",
        [],
    ),
    (
        "Planck-Einstein",
        "atomic.qm",
        "E = h * f = hbar * omega",
        ["*"],
    ),
    (
        "Bohr Model",
        "atomic.qm",
        "E_n = -13.6 eV / n^2",
        ["/", "-"],
    ),
    (
        "Lennard-Jones Potential",
        "atomic.md",
        "V(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)",
        ["*", "-", "/", "exp"],
    ),
    (
        "Coulomb Integral",
        "atomic.qm",
        "J_ij = integral(psi_i*(r1) * psi_j*(r2) * (1/r12) * psi_i(r1) * psi_j(r2))",
        ["*", "/"],
    ),
    (
        "Born-Oppenheimer",
        "atomic.qm",
        "Psi(r,R) approx psi_e(r;R) * chi_n(R)",
        ["*"],
    ),
    (
        "Bragg's Law",
        "atomic.crystal",
        "2 * d * sin(theta) = n * lambda",
        ["*", "sin"],
    ),
    (
        "Bloch's Theorem",
        "atomic.crystal",
        "psi_k(r) = u_k(r) * exp(i*k*r)",
        ["*", "exp"],
    ),
    (
        "Density of States",
        "atomic.crystal",
        "g(E) = (1/(2*pi^2)) * (2*m/hbar^2)^(3/2) * sqrt(E)",
        ["*", "/", "sqrt"],
    ),
]


@register_domain("atomic")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
