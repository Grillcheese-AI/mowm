"""Domain registry for MoWM axiom library.

Each domain module exports a seed(library) function via the @register_domain
decorator. Use seed_all(library) to seed all registered domains at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

_REGISTRY: dict[str, Callable[["AxiomLibrary"], None]] = {}


def register_domain(name: str):
    """Decorator to register a domain's seed function."""
    def decorator(fn: Callable[["AxiomLibrary"], None]):
        _REGISTRY[name] = fn
        return fn
    return decorator


def seed_all(library: "AxiomLibrary", domains: list[str] | None = None) -> None:
    """Seed the axiom library with all (or selected) registered domains."""
    targets = domains if domains is not None else list(_REGISTRY.keys())
    for name in targets:
        if name in _REGISTRY:
            _REGISTRY[name](library)


def available_domains() -> list[str]:
    """Return names of all registered domains."""
    return list(_REGISTRY.keys())
