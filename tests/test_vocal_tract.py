# tests/test_vocal_tract.py
"""Tests for VocalTract — TTS with prosody modulation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mowm.vocal_tract import (
    INTENT_PROSODY,
    ProsodyProfile,
    SpeechOutput,
    VocalTract,
)


class TestProsodyMapping:
    def test_explain_maps_to_instructive(self):
        assert INTENT_PROSODY["EXPLAIN"] == ProsodyProfile.INSTRUCTIVE

    def test_compare_maps_to_analytical(self):
        assert INTENT_PROSODY["COMPARE"] == ProsodyProfile.ANALYTICAL

    def test_urgent_maps_to_urgent(self):
        assert INTENT_PROSODY["BREAKING_NEWS"] == ProsodyProfile.URGENT

    def test_compute_maps_to_confident(self):
        assert INTENT_PROSODY["COMPUTE"] == ProsodyProfile.CONFIDENT

    def test_causal_maps_to_step_by_step(self):
        assert INTENT_PROSODY["CAUSAL"] == ProsodyProfile.STEP_BY_STEP


class TestVocalTractNoEngine:
    def setup_method(self):
        self.vt = VocalTract(tts_engine=None)

    def test_has_no_engine(self):
        assert self.vt.has_engine is False

    def test_speak_returns_speech_output(self):
        result = self.vt.speak("Force equals mass times acceleration.")
        assert isinstance(result, SpeechOutput)
        assert result.text == "Force equals mass times acceleration."
        assert result.audio_bytes is None

    def test_speak_produces_ssml(self):
        result = self.vt.speak("Hello world.", intent="EXPLAIN")
        assert "<speak>" in result.ssml
        assert "</speak>" in result.ssml
        assert "<prosody" in result.ssml

    def test_ssml_contains_rate(self):
        result = self.vt.speak("Test.", intent="BREAKING_NEWS")
        assert 'rate="fast"' in result.ssml

    def test_ssml_contains_pitch(self):
        result = self.vt.speak("Test.", intent="BREAKING_NEWS")
        assert 'pitch="high"' in result.ssml

    def test_speak_explain_slow_rate(self):
        result = self.vt.speak("Slow explanation.", intent="EXPLAIN")
        assert 'rate="slow"' in result.ssml

    def test_speak_unknown_intent_defaults(self):
        result = self.vt.speak("Unknown intent.", intent="FOOBAR")
        assert result.prosody == ProsodyProfile.INFORMATIVE

    def test_speak_multiple_sentences_have_breaks(self):
        result = self.vt.speak(
            "First sentence. Second sentence. Third.",
            intent="STEP_BY_STEP",
        )
        assert "<break" in result.ssml

    def test_params_stored(self):
        result = self.vt.speak("Test.", intent="COMPUTE")
        assert "rate" in result.params
        assert "pitch" in result.params
        assert "volume" in result.params


class TestVocalTractWithEngine:
    def test_has_engine(self):
        mock = MagicMock()
        vt = VocalTract(tts_engine=mock)
        assert vt.has_engine is True

    def test_engine_called(self):
        mock = MagicMock()
        mock.synthesize.return_value = b"audio_data"
        vt = VocalTract(tts_engine=mock)
        result = vt.speak("Hello.", intent="COMPUTE")
        mock.synthesize.assert_called_once()
        assert result.audio_bytes == b"audio_data"

    def test_engine_failure_no_crash(self):
        mock = MagicMock()
        mock.synthesize.side_effect = RuntimeError("TTS failed")
        vt = VocalTract(tts_engine=mock)
        result = vt.speak("Hello.", intent="EXPLAIN")
        assert result.audio_bytes is None
        assert len(result.ssml) > 0

    def test_engine_receives_params(self):
        mock = MagicMock()
        mock.synthesize.return_value = b""
        vt = VocalTract(tts_engine=mock)
        vt.speak("Test.", intent="URGENT")
        call_kwargs = mock.synthesize.call_args
        # Should have rate, pitch, volume in kwargs
        assert "rate" in call_kwargs.kwargs or len(call_kwargs.args) > 1
