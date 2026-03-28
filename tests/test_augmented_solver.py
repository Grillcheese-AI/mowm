# tests/test_augmented_solver.py
"""Tests for AugmentedSolver — LLM-assisted Q&A with template fallback."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.qa.augmented_solver import AugmentedSolver
from mowm.qa.direct_solver import QAResponse
from mowm.qa.parser import QuestionParser

K, L = 4, 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom("Newton's Second Law", "physics", "F = m * a"))
    lib.register(lib.make_axiom("Kinetic Energy", "physics", "KE = 0.5 * m * v^2"))
    lib.register(lib.make_axiom("Ohm's Law", "electronics", "V = I * R"))
    return lib


@pytest.fixture
def executor() -> SafeFormulaExecutor:
    return SafeFormulaExecutor()


@pytest.fixture
def parser() -> QuestionParser:
    return QuestionParser()


@pytest.fixture
def solver(lib, executor) -> AugmentedSolver:
    return AugmentedSolver(lib, executor)


@pytest.fixture
def solver_with_endpoint(lib, executor) -> AugmentedSolver:
    return AugmentedSolver(
        lib,
        executor,
        llm_endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key="sk-test-key",
    )


# ---------------------------------------------------------------------------
# TestTemplateFallback
# ---------------------------------------------------------------------------


class TestTemplateFallback:
    def test_template_returns_qa_response(self, solver, parser):
        pq = parser.parse("Explain Newton's Second Law")
        resp = solver.solve(pq, "physics")
        assert isinstance(resp, QAResponse)

    def test_template_mode_is_template(self, solver, parser):
        pq = parser.parse("Explain Newton's Second Law")
        resp = solver.solve(pq, "physics")
        assert resp.mode == "TEMPLATE"

    def test_template_confidence_scales_with_match(self, solver, parser):
        pq = parser.parse("Explain Newton's Second Law")
        resp = solver.solve(pq, "physics")
        # Confidence should be positive and below direct solver level
        assert 0.0 < resp.confidence < 0.90

    def test_template_answer_contains_domain(self, solver, parser):
        pq = parser.parse("Explain Newton's Second Law")
        resp = solver.solve(pq, "physics")
        assert "physics" in resp.answer.lower() or "PHYSICS" in resp.answer

    def test_template_answer_contains_question(self, solver, parser):
        pq = parser.parse("Explain Newton's Second Law")
        resp = solver.solve(pq, "physics")
        assert "Newton" in resp.answer or "Second Law" in resp.answer


# ---------------------------------------------------------------------------
# TestContextBuilding
# ---------------------------------------------------------------------------


class TestContextBuilding:
    def test_context_has_domain(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        ctx = solver._build_context(pq, "physics", {})
        assert ctx["domain"] == "physics"

    def test_context_has_axioms_list(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        ctx = solver._build_context(pq, "physics", {})
        assert "axioms" in ctx
        assert isinstance(ctx["axioms"], list)

    def test_context_axioms_have_name_and_formula(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        ctx = solver._build_context(pq, "physics", {})
        for ax in ctx["axioms"]:
            assert "name" in ax
            assert "formula" in ax

    def test_context_computed_values_populated(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        ctx = solver._build_context(pq, "physics", {})
        # Variables m=10, a=9.8 should allow executor to compute something
        assert "computed_values" in ctx

    def test_context_question_field(self, solver, parser):
        question = "What is F given m=10 and a=9.8?"
        pq = parser.parse(question)
        ctx = solver._build_context(pq, "physics", {})
        assert ctx["question"] == question

    def test_find_relevant_axioms_prefers_domain(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        axioms = solver.find_relevant_axioms(pq, "physics")
        domains = [ax.domain.split(".")[0] for ax in axioms]
        assert "physics" in domains

    def test_find_relevant_axioms_respects_max(self, solver, parser):
        pq = parser.parse("mass velocity force kinetic energy")
        solver_small = AugmentedSolver(solver._lib, solver._executor, max_axioms=2)
        axioms = solver_small.find_relevant_axioms(pq, "physics")
        assert len(axioms) <= 2

    def test_find_relevant_axioms_empty_library(self, executor, parser):
        empty_lib = AxiomLibrary(k=K, l=L)
        s = AugmentedSolver(empty_lib, executor)
        pq = parser.parse("What is force?")
        axioms = s.find_relevant_axioms(pq, "physics")
        assert axioms == []


# ---------------------------------------------------------------------------
# TestLLMCall (mock httpx)
# ---------------------------------------------------------------------------


class TestLLMCall:
    def _make_mock_response(self, content: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_llm_call_returns_qa_response(self, solver_with_endpoint, parser):
        pq = parser.parse("Explain kinetic energy")
        mock_resp = self._make_mock_response("Kinetic energy is KE = 0.5 * m * v^2.")
        with patch("mowm.qa.augmented_solver.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            resp = solver_with_endpoint.solve(pq, "physics")
        assert isinstance(resp, QAResponse)

    def test_llm_call_mode_is_llm(self, solver_with_endpoint, parser):
        pq = parser.parse("Explain kinetic energy")
        mock_resp = self._make_mock_response("KE explanation")
        with patch("mowm.qa.augmented_solver.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            resp = solver_with_endpoint.solve(pq, "physics")
        assert resp.mode == "LLM"

    def test_llm_call_uses_bearer_token(self, solver_with_endpoint, parser):
        pq = parser.parse("Explain force")
        mock_resp = self._make_mock_response("Force explanation")
        with patch("mowm.qa.augmented_solver.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            solver_with_endpoint.solve(pq, "physics")
        call_kwargs = mock_httpx.post.call_args
        headers = call_kwargs[1].get("headers", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {})
        # Check Authorization header was passed
        assert "headers" in call_kwargs.kwargs or call_kwargs[1]
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "Authorization" in all_kwargs.get("headers", {})
        assert all_kwargs["headers"]["Authorization"] == "Bearer sk-test-key"

    def test_llm_fallback_on_http_error(self, solver_with_endpoint, parser):
        pq = parser.parse("Explain kinetic energy")
        with patch("mowm.qa.augmented_solver.httpx") as mock_httpx:
            mock_httpx.post.side_effect = Exception("connection refused")
            resp = solver_with_endpoint.solve(pq, "physics")
        # Should fall back to template
        assert isinstance(resp, QAResponse)
        assert resp.mode == "TEMPLATE"


# ---------------------------------------------------------------------------
# TestLLMPromptFormat
# ---------------------------------------------------------------------------


class TestLLMPromptFormat:
    def test_prompt_contains_domain(self, solver, parser):
        pq = parser.parse("What is F given m=5?")
        ctx = solver._build_context(pq, "physics", {})
        prompt = solver._build_prompt(pq, ctx)
        assert "physics" in prompt

    def test_prompt_contains_question(self, solver, parser):
        question = "What is F given m=5?"
        pq = parser.parse(question)
        ctx = solver._build_context(pq, "physics", {})
        prompt = solver._build_prompt(pq, ctx)
        assert question in prompt

    def test_prompt_contains_axiom_name(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        ctx = solver._build_context(pq, "physics", {})
        prompt = solver._build_prompt(pq, ctx)
        # At least one physics axiom name should appear
        assert "Newton" in prompt or "Kinetic" in prompt or "F" in prompt


# ---------------------------------------------------------------------------
# TestOpenRouterAPIKey
# ---------------------------------------------------------------------------


class TestOpenRouterAPIKey:
    def test_no_api_key_no_auth_header(self, lib, executor, parser):
        s = AugmentedSolver(
            lib,
            executor,
            llm_endpoint="https://openrouter.ai/api/v1/chat/completions",
        )
        pq = parser.parse("Explain force")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("mowm.qa.augmented_solver.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_resp
            s.solve(pq, "physics")
        call_kwargs = mock_httpx.post.call_args.kwargs or mock_httpx.post.call_args[1]
        assert "Authorization" not in call_kwargs.get("headers", {})

    def test_solver_without_endpoint_uses_template(self, lib, executor, parser):
        s = AugmentedSolver(lib, executor)
        pq = parser.parse("Compare Newton vs Euler")
        resp = s.solve(pq, "physics")
        assert resp.mode == "TEMPLATE"
