# mowm/domains/security.py
"""Security domain — cybersecurity, encryption, access control, threat models.

Registers axioms covering network security, cryptography, authentication,
vulnerability assessment, and compliance frameworks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

SECURITY_OPERATORS: list[str] = [
    "+", "-", "*", "/", "exp", "log", "AND", "OR", "NOT", "IMPLIES",
]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    # Cryptography
    (
        "Shannon Entropy (Key)",
        "security.crypto",
        "H = -sum(p_i * log2(p_i))",
        ["*", "log", "-"],
    ),
    (
        "Brute Force Complexity",
        "security.crypto",
        "attempts = 2^key_length",
        ["exp"],
    ),
    (
        "RSA Key Generation",
        "security.crypto",
        "n = p * q; phi = (p-1)*(q-1)",
        ["*", "-"],
    ),
    (
        "Diffie-Hellman Exchange",
        "security.crypto",
        "shared = g^(a*b) mod p",
        ["exp", "*"],
    ),
    (
        "Hash Collision Probability",
        "security.crypto",
        "P_collision = 1 - exp(-n^2 / (2 * H))",
        ["-", "exp", "/", "*"],
    ),
    # Authentication
    (
        "Multi-Factor Auth",
        "security.auth",
        "auth = factor_know AND factor_have AND factor_are",
        ["AND"],
    ),
    (
        "Password Strength",
        "security.auth",
        "entropy = log2(charset^length)",
        ["log", "exp"],
    ),
    (
        "Session Timeout",
        "security.auth",
        "valid = (now - issued) < max_age",
        ["-"],
    ),
    # Network Security
    (
        "CIA Triad",
        "security.network",
        "security = confidentiality AND integrity AND availability",
        ["AND"],
    ),
    (
        "Defense in Depth",
        "security.network",
        "P_breach = P_layer1 * P_layer2 * P_layer3",
        ["*"],
    ),
    (
        "Firewall Rule",
        "security.network",
        "allow = src_match AND dst_match AND port_match AND NOT blacklisted",
        ["AND", "NOT"],
    ),
    (
        "Risk Assessment",
        "security.risk",
        "risk = likelihood * impact",
        ["*"],
    ),
    # Vulnerability
    (
        "CVSS Score",
        "security.vulnerability",
        "score = base * temporal * environmental",
        ["*"],
    ),
    (
        "Attack Surface",
        "security.vulnerability",
        "surface = sum(entry_points * exposure_factor)",
        ["+", "*"],
    ),
    (
        "Zero Trust Principle",
        "security.network",
        "trust = NOT implicit; verify = always",
        ["NOT"],
    ),
]


@register_domain("security")
def seed(library: "AxiomLibrary") -> None:
    """Seed the axiom library with cybersecurity axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else SECURITY_OPERATORS,
        )
        library.register(axiom)
