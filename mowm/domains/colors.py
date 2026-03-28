from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

DOMAIN_OPERATORS: list[str] = ["+", "-", "*", "/", "sin", "cos", "exp"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    (
        "Wavelength-Frequency",
        "colors.physics",
        "c = lambda * f",
        ["*"],
    ),
    (
        "Planck to Perception",
        "colors.physics",
        "L(lambda) = 2*h*c^2 / (lambda^5 * (exp(h*c/(lambda*k_B*T)) - 1))",
        ["*", "/", "exp", "-"],
    ),
    (
        "RGB Additive",
        "colors.space",
        "C = R + G + B",
        ["+"],
    ),
    (
        "CMYK Subtractive",
        "colors.space",
        "C_ink = 1 - R, M_ink = 1 - G, Y_ink = 1 - B",
        ["-"],
    ),
    (
        "HSV to RGB",
        "colors.space",
        "R,G,B = f(H, S, V) with sector index i = floor(H/60)",
        ["+", "-", "*", "/"],
    ),
    (
        "CIE XYZ Tristimulus",
        "colors.space",
        "X = integral(S(lambda)*x_bar(lambda)dlambda)",
        ["*", "+"],
    ),
    (
        "Complementary Colors",
        "colors.harmony",
        "H_complement = (H + 180) mod 360",
        ["+"],
    ),
    (
        "Triadic Harmony",
        "colors.harmony",
        "H_triad = {H, (H+120) mod 360, (H+240) mod 360}",
        ["+"],
    ),
    (
        "Analogous Harmony",
        "colors.harmony",
        "H_analogous = {H-30, H, H+30} mod 360",
        ["+", "-"],
    ),
    (
        "Color Temperature",
        "colors.perception",
        "T_color approx 1.438e7 / lambda_peak (Wien's displacement)",
        ["/"],
    ),
    (
        "Weber-Fechner (color)",
        "colors.perception",
        "delta_L* = k * delta_L / L",
        ["*", "/"],
    ),
    (
        "Metamerism",
        "colors.perception",
        "C1 = C2 perceptually if integral(S1*ck) = integral(S2*ck) for k in {x,y,z}",
        ["*", "+"],
    ),
    (
        "Chromatic Adaptation",
        "colors.perception",
        "R_adapted = R / (R + sigma_R)",
        ["/", "+"],
    ),
    (
        "Gamma Correction",
        "colors.display",
        "V_out = V_in ^ (1/gamma)",
        ["exp", "/"],
    ),
    (
        "sRGB Transfer",
        "colors.display",
        "C_linear = C_sRGB/12.92 if C<=0.04045 else ((C+0.055)/1.055)^2.4",
        ["/", "+", "exp"],
    ),
]


@register_domain("colors")
def seed(library: "AxiomLibrary") -> None:
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else DOMAIN_OPERATORS,
        )
        library.register(axiom)
