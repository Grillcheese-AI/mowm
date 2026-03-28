# mowm/domains/physics.py
"""Physics domain — kinematics, dynamics, energy, fluids, thermo, EM, waves.

Registers ~85 axioms covering the core branches of physics. Each axiom
is encoded as a block-code vector via AxiomLibrary.make_axiom.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

PHYSICS_OPERATORS: list[str] = [
    "+", "-", "*", "/", "sin", "cos", "sqrt", "square", "exp", "log",
]

_KINEMATICS: list[tuple[str, str, str, list[str]]] = [
    ("Position Update", "physics.kinematics", "x = x0 + v*t", ["+", "*"]),
    ("Velocity Update", "physics.kinematics", "v = v0 + a*t", ["+", "*"]),
    (
        "Displacement (constant accel)",
        "physics.kinematics",
        "x = x0 + v0*t + 0.5*a*t^2",
        ["+", "*", "square"],
    ),
    (
        "Velocity-Displacement",
        "physics.kinematics",
        "v^2 = v0^2 + 2*a*(x - x0)",
        ["+", "*", "square", "-"],
    ),
    ("Average Velocity", "physics.kinematics", "v_avg = (v + v0) / 2", ["+", "/"]),
    ("Angular Velocity", "physics.kinematics", "omega = d_theta / dt", ["/"]),
    ("Angular Acceleration", "physics.kinematics", "alpha = d_omega / dt", ["/"]),
    (
        "Centripetal Acceleration",
        "physics.kinematics",
        "a_c = v^2 / r",
        ["square", "/"],
    ),
    (
        "Projectile Range",
        "physics.kinematics",
        "R = v0^2 * sin(2*theta) / g",
        ["square", "*", "sin", "/"],
    ),
    (
        "Projectile Max Height",
        "physics.kinematics",
        "H = v0^2 * sin^2(theta) / (2*g)",
        ["square", "*", "sin", "/"],
    ),
]

_DYNAMICS: list[tuple[str, str, str, list[str]]] = [
    ("Newton's First Law", "physics.dynamics", "F_net = 0 -> v = const", []),
    ("Newton's Second Law", "physics.dynamics", "F = m * a", ["*"]),
    ("Newton's Third Law", "physics.dynamics", "F_12 = -F_21", ["-"]),
    ("Weight", "physics.dynamics", "W = m * g", ["*"]),
    ("Friction", "physics.dynamics", "f = mu * N", ["*"]),
    ("Spring Force", "physics.dynamics", "F = -k * x", ["-", "*"]),
    (
        "Gravitational Force",
        "physics.dynamics",
        "F = G * m1 * m2 / r^2",
        ["*", "/", "square"],
    ),
    ("Momentum", "physics.dynamics", "p = m * v", ["*"]),
    ("Impulse", "physics.dynamics", "J = F * dt = dp", ["*"]),
    (
        "Conservation of Momentum",
        "physics.dynamics",
        "p1 + p2 = p1' + p2'",
        ["+"],
    ),
    ("Torque", "physics.dynamics", "tau = r x F = I * alpha", ["*"]),
    ("Angular Momentum", "physics.dynamics", "L = I * omega", ["*"]),
    (
        "Drag Force",
        "physics.dynamics",
        "F_d = 0.5 * C_d * rho * A * v^2",
        ["*", "square"],
    ),
]

_ENERGY: list[tuple[str, str, str, list[str]]] = [
    ("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2", ["*", "square"]),
    ("Gravitational PE", "physics.energy", "PE = m * g * h", ["*"]),
    ("Elastic PE", "physics.energy", "PE = 0.5 * k * x^2", ["*", "square"]),
    ("Conservation of Energy", "physics.energy", "E_total = KE + PE = const", ["+"]),
    ("Work", "physics.energy", "W = F * d * cos(theta)", ["*", "cos"]),
    ("Power", "physics.energy", "P = W / t = F * v", ["*", "/"]),
    ("Work-Energy Theorem", "physics.energy", "W_net = delta_KE", ["-"]),
    (
        "Rotational KE",
        "physics.energy",
        "KE_rot = 0.5 * I * omega^2",
        ["*", "square"],
    ),
]

_FLUIDS: list[tuple[str, str, str, list[str]]] = [
    ("Pressure", "physics.fluids", "P = F / A", ["/"]),
    ("Hydrostatic Pressure", "physics.fluids", "P = rho * g * h", ["*"]),
    ("Continuity Equation", "physics.fluids", "A1 * v1 = A2 * v2", ["*"]),
    (
        "Bernoulli's Equation",
        "physics.fluids",
        "P + 0.5*rho*v^2 + rho*g*h = const",
        ["+", "*", "square"],
    ),
    ("Buoyancy", "physics.fluids", "F_b = rho_f * V * g", ["*"]),
    (
        "Viscous Flow (Poiseuille)",
        "physics.fluids",
        "Q = pi*r^4*dP / (8*mu*L)",
        ["*", "/", "square"],
    ),
    ("Reynolds Number", "physics.fluids", "Re = rho*v*L / mu", ["*", "/"]),
    (
        "Navier-Stokes (simplified)",
        "physics.fluids",
        "rho*(dv/dt + v.grad(v)) = -grad(P) + mu*laplacian(v)",
        ["+", "-", "*"],
    ),
]

_THERMO: list[tuple[str, str, str, list[str]]] = [
    ("Ideal Gas Law", "physics.thermo", "PV = nRT", ["*"]),
    ("First Law of Thermo", "physics.thermo", "dU = Q - W", ["+", "-"]),
    ("Entropy Change", "physics.thermo", "dS = dQ / T", ["/"]),
    ("Specific Heat", "physics.thermo", "Q = m * c * dT", ["*"]),
    ("Carnot Efficiency", "physics.thermo", "eta = 1 - T_cold / T_hot", ["-", "/"]),
    ("Stefan-Boltzmann", "physics.thermo", "P = sigma * A * T^4", ["*", "square"]),
    ("Fourier's Law", "physics.thermo", "q = -k * dT/dx", ["-", "*", "/"]),
    (
        "Boltzmann Distribution",
        "physics.thermo",
        "P(E) = exp(-E / (k_B * T)) / Z",
        ["exp", "-", "*", "/"],
    ),
    (
        "Maxwell-Boltzmann Speed",
        "physics.thermo",
        "f(v) = 4*pi*(m/(2*pi*k_B*T))^1.5 * v^2 * exp(-m*v^2/(2*k_B*T))",
        ["*", "square", "exp", "-", "/", "sqrt"],
    ),
    ("Clausius Inequality", "physics.thermo", "dS >= dQ / T", ["/"]),
]

_EM: list[tuple[str, str, str, list[str]]] = [
    (
        "Coulomb's Law",
        "physics.em",
        "F = k_e * q1 * q2 / r^2",
        ["*", "/", "square"],
    ),
    (
        "Electric Field",
        "physics.em",
        "E = F / q = k_e * Q / r^2",
        ["/", "*", "square"],
    ),
    ("Gauss's Law", "physics.em", "flux(E) = Q_enc / epsilon_0", ["/"]),
    ("Electric Potential", "physics.em", "V = k_e * Q / r", ["*", "/"]),
    ("Ohm's Law", "physics.em", "V = I * R", ["*"]),
    (
        "Power (electrical)",
        "physics.em",
        "P = I * V = I^2 * R",
        ["*", "square"],
    ),
    (
        "Capacitance",
        "physics.em",
        "C = Q / V = epsilon_0 * A / d",
        ["/", "*"],
    ),
    ("Magnetic Force", "physics.em", "F = q * v x B", ["*"]),
    ("Faraday's Law", "physics.em", "emf = -d(flux_B)/dt", ["-", "/"]),
    (
        "Ampere's Law",
        "physics.em",
        "curl(B) = mu_0 * J + mu_0*epsilon_0 * dE/dt",
        ["+", "*"],
    ),
    ("Lorentz Force", "physics.em", "F = q*(E + v x B)", ["+", "*"]),
    (
        "Biot-Savart",
        "physics.em",
        "dB = mu_0/(4*pi) * I*dl x r_hat / r^2",
        ["*", "/", "square"],
    ),
    ("Inductance", "physics.em", "V_L = -L * dI/dt", ["-", "*", "/"]),
    (
        "Maxwell Displacement Current",
        "physics.em",
        "J_d = epsilon_0 * dE/dt",
        ["*", "/"],
    ),
]

_WAVES: list[tuple[str, str, str, list[str]]] = [
    ("Wave Speed", "physics.waves", "v = f * lambda", ["*"]),
    (
        "Wave Equation",
        "physics.waves",
        "d2y/dx2 = (1/v^2) * d2y/dt2",
        ["/", "square"],
    ),
    (
        "Standing Wave",
        "physics.waves",
        "y = 2*A*sin(kx)*cos(omega*t)",
        ["*", "sin", "cos"],
    ),
    (
        "Doppler Effect",
        "physics.waves",
        "f' = f * (v + v_obs) / (v + v_src)",
        ["*", "+", "/"],
    ),
    ("Intensity", "physics.waves", "I = P / (4*pi*r^2)", ["/", "*", "square"]),
    (
        "Snell's Law",
        "physics.waves",
        "n1*sin(theta1) = n2*sin(theta2)",
        ["*", "sin"],
    ),
    (
        "Diffraction (single slit)",
        "physics.waves",
        "a*sin(theta) = m*lambda",
        ["*", "sin"],
    ),
    (
        "Interference (double slit)",
        "physics.waves",
        "d*sin(theta) = m*lambda",
        ["*", "sin"],
    ),
    ("Photon Energy", "physics.waves", "E = h * f = h*c / lambda", ["*", "/"]),
    ("De Broglie Wavelength", "physics.waves", "lambda = h / p", ["/"]),
]

_RELATIVITY: list[tuple[str, str, str, list[str]]] = [
    (
        "Lorentz Factor",
        "physics.relativity",
        "gamma = 1 / sqrt(1 - v^2/c^2)",
        ["/", "sqrt", "-", "square"],
    ),
    ("Time Dilation", "physics.relativity", "dt' = gamma * dt", ["*"]),
    ("Length Contraction", "physics.relativity", "L = L0 / gamma", ["/"]),
    (
        "Mass-Energy Equivalence",
        "physics.relativity",
        "E = m * c^2",
        ["*", "square"],
    ),
    (
        "Relativistic Momentum",
        "physics.relativity",
        "p = gamma * m * v",
        ["*"],
    ),
    (
        "Relativistic Energy",
        "physics.relativity",
        "E^2 = (pc)^2 + (mc^2)^2",
        ["+", "square", "*"],
    ),
]

_GRAVITY: list[tuple[str, str, str, list[str]]] = [
    (
        "Gravitational PE (orbital)",
        "physics.gravity",
        "U = -G*M*m / r",
        ["-", "*", "/"],
    ),
    (
        "Escape Velocity",
        "physics.gravity",
        "v_esc = sqrt(2*G*M / r)",
        ["sqrt", "*", "/"],
    ),
    (
        "Orbital Velocity",
        "physics.gravity",
        "v_orb = sqrt(G*M / r)",
        ["sqrt", "*", "/"],
    ),
    (
        "Kepler's Third Law",
        "physics.gravity",
        "T^2 = (4*pi^2 / (G*M)) * a^3",
        ["*", "/", "square"],
    ),
    (
        "Schwarzschild Radius",
        "physics.gravity",
        "r_s = 2*G*M / c^2",
        ["*", "/", "square"],
    ),
]

_ALL_PHYSICS = (
    _KINEMATICS + _DYNAMICS + _ENERGY + _FLUIDS + _THERMO
    + _EM + _WAVES + _RELATIVITY + _GRAVITY
)


@register_domain("physics")
def seed(library: "AxiomLibrary") -> None:
    """Seed the axiom library with physics axioms."""
    for name, domain, formula, ops in _ALL_PHYSICS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else PHYSICS_OPERATORS,
        )
        library.register(axiom)
