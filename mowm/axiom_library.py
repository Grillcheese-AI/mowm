# mowm/axiom_library.py
"""Global World axiom library — storage, selection, and composition.

The AxiomLibrary is the living Global World: it stores axioms as block-code
vectors with metadata, supports cosine similarity selection, and composes
axioms via VSA binding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes


@dataclass
class Axiom:
    """A single axiom in the Global World.

    Attributes:
        name: Human-readable name (e.g. "Newton's Second Law").
        domain: Dot-separated domain path (e.g. "physics.dynamics").
        formula_str: Symbolic formula string (e.g. "F = m * a").
        vector: Block-code vector of shape (k, l).
        operators: Grammar operators for symbolic regression.
    """

    name: str
    domain: str
    formula_str: str
    vector: np.ndarray
    operators: list[str] = field(default_factory=list)


class AxiomLibrary:
    """Living Global World — axiom storage with similarity-based selection.

    Axioms are block-code vectors constructed via:
        axiom_vec = bind(domain_vec, bind(name_vec, param_vec))

    Selection uses the BlockCodes.similarity_batch for GPU-accelerated
    cosine similarity over the codebook.

    Args:
        k: Number of blocks.
        l: Block length.
        domains: Optional list of domain names to pre-filter.
    """

    def __init__(
        self,
        k: int,
        l: int,  # noqa: E741
        domains: list[str] | None = None,
    ) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self.encoder = WorldEncoder(k=k, l=l)
        self._axioms: dict[int, Axiom] = {}
        self._next_id: int = 0
        self._allowed_domains = set(domains) if domains else None

    def __len__(self) -> int:
        return len(self._axioms)

    def register(self, axiom: Axiom) -> int:
        """Register an axiom and return its unique ID."""
        aid = self._next_id
        self._axioms[aid] = axiom
        self._next_id += 1
        return aid

    def select(self, query_vec: np.ndarray, top_n: int = 5) -> list[Axiom]:
        """Select the top-N most similar axioms to a query vector.

        Uses BlockCodes.similarity_batch for GPU-accelerated comparison.
        """
        if not self._axioms:
            return []

        ids = list(self._axioms.keys())
        axioms = [self._axioms[i] for i in ids]
        codebook = np.stack([a.vector for a in axioms], axis=0)

        sims = self.bc.similarity_batch(query_vec, codebook)
        sorted_indices = np.argsort(-sims)
        top_indices = sorted_indices[:top_n]

        return [axioms[int(i)] for i in top_indices]

    def compose(self, axioms: list[Axiom]) -> np.ndarray:
        """Compose multiple axioms into a single vector via VSA binding."""
        if not axioms:
            raise ValueError("Cannot compose an empty list of axioms")

        result = axioms[0].vector.copy()
        for ax in axioms[1:]:
            result = self.bc.bind(result, ax.vector)
        return result

    def remove(self, axiom_id: int) -> None:
        """Remove an axiom by its ID."""
        if axiom_id not in self._axioms:
            raise KeyError(f"Axiom ID {axiom_id} not found")
        del self._axioms[axiom_id]

    def domains(self) -> dict[str, list[Axiom]]:
        """Group all registered axioms by their top-level domain."""
        result: dict[str, list[Axiom]] = {}
        for ax in self._axioms.values():
            top_domain = ax.domain.split(".")[0]
            result.setdefault(top_domain, []).append(ax)
        return result

    def make_axiom(
        self,
        name: str,
        domain: str,
        formula_str: str,
        operators: list[str] | None = None,
    ) -> Axiom:
        """Create an axiom with its vector auto-generated via WorldEncoder.

        Constructs: axiom_vec = bind(domain_vec, bind(name_vec, formula_vec))
        Deterministic: same inputs always produce the same vector.
        """
        domain_vec = self.encoder.encode_action(domain)
        name_vec = self.encoder.encode_action(name)
        formula_vec = self.encoder.encode_action(formula_str)
        axiom_vec = self.bc.bind(domain_vec, self.bc.bind(name_vec, formula_vec))
        return Axiom(
            name=name,
            domain=domain,
            formula_str=formula_str,
            vector=axiom_vec,
            operators=operators or [],
        )

    def all_axioms(self) -> list[Axiom]:
        """Return all registered axioms as a list."""
        return list(self._axioms.values())
