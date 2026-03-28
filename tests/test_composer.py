# tests/test_composer.py
"""Tests for Composer — template-based text generation from reasoning."""
from __future__ import annotations

import pytest

from mowm.reasoning.causal_tracer import CausalNarrative
from mowm.reasoning.chain_builder import DerivationChain, DerivationStep
from mowm.reasoning.comparator import ComparisonResult
from mowm.reasoning.composer import Composer
from mowm.reasoning.knowledge_search import KnowledgeContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def composer() -> Composer:
    return Composer()


@pytest.fixture
def physics_chain() -> DerivationChain:
    """Chain for: why do objects fall at the same rate?"""
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Weight",
                formula="W = m * g",
                substitutions={},
                result_var="W",
                result_value="m * g",
                explanation="By Weight: W = m * g",
            ),
            DerivationStep(
                axiom_name="Newton's Second Law",
                formula="F = m * a",
                substitutions={},
                result_var="a",
                result_value="F / m = m*g / m = g",
                explanation="By Newton's Second Law: a = F/m = m*g/m = g",
            ),
        ],
        conclusion="Acceleration a = g for all masses; mass cancels.",
        variables_used={"m", "g", "a", "F", "W"},
        axioms_used=["Weight", "Newton's Second Law"],
    )


@pytest.fixture
def physics_causal() -> CausalNarrative:
    return CausalNarrative(
        cause="mass",
        effects=[
            ("grav_force", "increases", 1.0),
            ("inertia", "increases", 1.0),
        ],
        cancellations=[("grav_force", "inertia")],
        net_effect="grav_force and inertia cancel out",
        graph_path=["mass", "grav_force", "inertia", "acceleration"],
    )


@pytest.fixture
def economics_chain() -> DerivationChain:
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Supply-Demand Equilibrium",
                formula="Q_d = Q_s",
                substitutions={},
                result_var="Q_d",
                result_value="Q_s",
                explanation="By Supply-Demand Equilibrium: Q_d = Q_s at equilibrium.",
            ),
        ],
        conclusion="Equilibrium: Q_d = Q_s at price P*.",
        variables_used={"Q_d", "Q_s", "P"},
        axioms_used=["Supply-Demand Equilibrium"],
    )


@pytest.fixture
def logic_chain() -> DerivationChain:
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Modus Ponens",
                formula="(P AND (P IMPLIES Q)) IMPLIES Q",
                substitutions={},
                result_var="Q",
                result_value="True (given P and P->Q)",
                explanation=(
                    "By Modus Ponens: if P is true and P implies Q, "
                    "then Q must be true."
                ),
            ),
        ],
        conclusion="Q follows from P and P->Q by Modus Ponens.",
        variables_used={"P", "Q"},
        axioms_used=["Modus Ponens"],
    )


@pytest.fixture
def bio_chain() -> DerivationChain:
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Hamilton's Rule",
                formula="benefit = r * B - C",
                substitutions={"r": 0.5, "B": 10.0, "C": 3.0},
                result_var="benefit",
                result_value=2.0,
                explanation=(
                    "By Hamilton's Rule: benefit = r*B - C = "
                    "0.5*10 - 3 = 2.0 > 0, altruism favored."
                ),
            ),
        ],
        conclusion="benefit = 2.0 > 0; altruism is favored.",
        variables_used={"r", "B", "C", "benefit"},
        axioms_used=["Hamilton's Rule"],
    )


@pytest.fixture
def music_comparison() -> ComparisonResult:
    return ComparisonResult(
        items=[
            {
                "name": "Major Third",
                "formula": "ratio = 5 / 4",
                "domain": "music.harmony",
                "value": 1.25,
            },
            {
                "name": "Minor Third",
                "formula": "ratio = 6 / 5",
                "domain": "music.harmony",
                "value": 1.2,
            },
        ],
        similarities=["Same domain: music", "Both are intervals"],
        differences=[
            "Major Third ratio: 5/4 = 1.25",
            "Minor Third ratio: 6/5 = 1.2",
        ],
        structural_diff=(
            "Major Third uses 5/4 ratio vs Minor Third uses 6/5 ratio"
        ),
    )


# ---------------------------------------------------------------------------
# TestLevelSelection
# ---------------------------------------------------------------------------


class TestLevelSelection:
    def test_level_1_no_chain(self, composer):
        level = composer._select_level(chain=None, causal=None)
        assert level == 1

    def test_level_1_empty_chain(self, composer):
        chain = DerivationChain()
        level = composer._select_level(chain=chain, causal=None)
        assert level == 1

    def test_level_2_short_chain(self, composer):
        chain = DerivationChain(
            steps=[DerivationStep(
                axiom_name="A", formula="x = 1",
                explanation="By A: x = 1",
            )],
            conclusion="x = 1",
            variables_used={"x"},
            axioms_used=["A"],
        )
        level = composer._select_level(chain=chain, causal=None)
        assert level == 2

    def test_level_3_multiple_steps(self, composer, physics_chain):
        level = composer._select_level(chain=physics_chain, causal=None)
        assert level >= 3

    def test_level_4_chain_plus_causal(self, composer, physics_chain, physics_causal):
        level = composer._select_level(chain=physics_chain, causal=physics_causal)
        assert level == 4


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Physics
# ---------------------------------------------------------------------------


class TestComposeExplanationPhysics:
    def test_explanation_contains_mass_or_cancel(
        self, composer, physics_chain, physics_causal,
    ):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=physics_causal,
            question="Why do heavier objects fall at the same rate?",
        )
        assert "mass" in text.lower() or "cancel" in text.lower()

    def test_explanation_lists_axioms(
        self, composer, physics_chain, physics_causal,
    ):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=physics_causal,
            question="Why do heavier objects fall at the same rate?",
        )
        assert "Weight" in text or "Newton" in text

    def test_explanation_includes_derivation_steps(
        self, composer, physics_chain,
    ):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=None,
            question="How does force relate to mass?",
        )
        assert "W = m * g" in text or "F = m * a" in text

    def test_explanation_nonempty(self, composer, physics_chain):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=None,
            question="Explain force.",
        )
        assert len(text) > 20


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Economics
# ---------------------------------------------------------------------------


class TestComposeExplanationEconomics:
    def test_economics_explanation(self, composer, economics_chain):
        text = composer.compose_explanation(
            chain=economics_chain,
            causal=None,
            question="Explain supply and demand equilibrium.",
        )
        assert "equilibrium" in text.lower() or "Q_d" in text

    def test_economics_lists_axiom(self, composer, economics_chain):
        text = composer.compose_explanation(
            chain=economics_chain,
            causal=None,
            question="Explain supply and demand.",
        )
        assert "Supply-Demand" in text or "Q_s" in text


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Logic
# ---------------------------------------------------------------------------


class TestComposeExplanationLogic:
    def test_logic_explanation(self, composer, logic_chain):
        text = composer.compose_explanation(
            chain=logic_chain,
            causal=None,
            question="Explain modus ponens.",
        )
        assert "Modus Ponens" in text
        assert "P" in text or "Q" in text


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Bio
# ---------------------------------------------------------------------------


class TestComposeExplanationBio:
    def test_bio_explanation(self, composer, bio_chain):
        text = composer.compose_explanation(
            chain=bio_chain,
            causal=None,
            question="Why does natural selection favor altruism in kin?",
        )
        assert "Hamilton" in text
        assert "2.0" in text or "benefit" in text.lower()


# ---------------------------------------------------------------------------
# TestComposeComparison -- Music
# ---------------------------------------------------------------------------


class TestComposeComparisonMusic:
    def test_music_comparison_names(self, composer, music_comparison):
        text = composer.compose_comparison(
            result=music_comparison,
            question="Compare major and minor thirds.",
        )
        assert "Major Third" in text
        assert "Minor Third" in text

    def test_comparison_shows_values(self, composer, music_comparison):
        text = composer.compose_comparison(
            result=music_comparison,
            question="Compare intervals.",
        )
        # Either the numeric values or formulas should appear
        assert "1.25" in text or "5 / 4" in text

    def test_comparison_shows_differences(self, composer, music_comparison):
        text = composer.compose_comparison(
            result=music_comparison,
            question="Compare intervals.",
        )
        assert "differ" in text.lower() or "1.2" in text or "6 / 5" in text


# ---------------------------------------------------------------------------
# TestComposeGeneral
# ---------------------------------------------------------------------------


class TestComposeGeneral:
    def test_general_from_context(self, composer):
        from mowm.axiom_library import AxiomLibrary
        lib = AxiomLibrary(k=4, l=32)
        ax = lib.make_axiom("Entropy Change", "physics.thermo", "dS = dQ / T")
        ctx = KnowledgeContext(
            primary_axioms=[ax],
            supporting_axioms=[],
            concepts=["Entropy Change"],
            domain_chain=["physics.thermo"],
            variables_mentioned={"dS", "dQ", "T"},
        )
        text = composer.compose_general(
            context=ctx,
            chain=None,
            question="Tell me about entropy.",
        )
        assert "Entropy" in text or "dS" in text
        assert len(text) > 10

    def test_general_with_chain(self, composer, physics_chain):
        from mowm.axiom_library import AxiomLibrary
        lib = AxiomLibrary(k=4, l=32)
        ax = lib.make_axiom("Newton's Second Law", "physics.dynamics", "F = m * a")
        ctx = KnowledgeContext(
            primary_axioms=[ax],
            concepts=["Newton's Second Law"],
            domain_chain=["physics.dynamics"],
            variables_mentioned={"F", "m", "a"},
        )
        text = composer.compose_general(
            context=ctx,
            chain=physics_chain,
            question="Tell me about force.",
        )
        assert len(text) > 20


# ---------------------------------------------------------------------------
# TestComposeEdgeCases
# ---------------------------------------------------------------------------


class TestComposeEdgeCases:
    def test_empty_chain_explanation(self, composer):
        chain = DerivationChain()
        text = composer.compose_explanation(
            chain=chain,
            causal=None,
            question="Why?",
        )
        assert isinstance(text, str)
        assert len(text) > 0

    def test_empty_comparison(self, composer):
        result = ComparisonResult()
        text = composer.compose_comparison(
            result=result,
            question="Compare nothing.",
        )
        assert isinstance(text, str)

    def test_empty_context_general(self, composer):
        ctx = KnowledgeContext()
        text = composer.compose_general(
            context=ctx,
            chain=None,
            question="Tell me about nothing.",
        )
        assert isinstance(text, str)
