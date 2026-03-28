from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "*", "sin", "cos", "sqrt", "exp"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Harmonic Series",
        "music.harmony",
        "f_n = n * f_0 for n = 1, 2, 3, ...",
        ["*"],
    ),
    (
        "Equal Temperament",
        "music.harmony",
        "f_n = f_0 * 2^(n/12)",
        ["*", "exp"],
    ),
    (
        "Consonance Ratio",
        "music.harmony",
        "consonance = simple integer ratio p/q, e.g. 3/2 (fifth)",
        ["/"],
    ),
    (
        "Phasor Representation",
        "music.phasor",
        "x(t) = A * cos(omega*t + phi)",
        ["*", "cos", "+"],
    ),
    (
        "Beat Frequency",
        "music.harmony",
        "f_beat = |f1 - f2|",
        ["-"],
    ),
    (
        "Counterpoint Parallel Fifths",
        "music.counterpoint",
        "avoid consecutive P5 intervals between voices",
        [],
    ),
    (
        "Counterpoint Voice Leading",
        "music.counterpoint",
        "prefer contrary motion: delta_v1 * delta_v2 < 0",
        ["*", "-"],
    ),
    (
        "Distortion Nonlinearity",
        "music.synthesis",
        "y(t) = a1*x + a2*x^2 + a3*x^3 + ...",
        ["+", "*"],
    ),
    (
        "Fourier Decomposition",
        "music.analysis",
        "x(t) = sum(A_n * sin(n*omega*t + phi_n))",
        ["+", "*", "sin"],
    ),
    (
        "ADSR Envelope",
        "music.synthesis",
        "A(t): attack + decay + sustain + release phases",
        ["+"],
    ),
    (
        "Nyquist Theorem",
        "music.sampling",
        "f_s >= 2 * f_max to avoid aliasing",
        ["*"],
    ),
]


@register_domain("music")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
