# mowm/vocal_tract.py
"""VocalTract — TTS with prosody modulation from reasoning intent.

Maps the logical intent of the reasoning decoder to acoustic parameters
(speed, pitch style) for text-to-speech synthesis. Supports local TTS
engines (VITS, Tacotron2) or API-based services.

When no TTS engine is available, outputs SSML-annotated text that any
standard TTS system can consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProsodyProfile(Enum):
    """Acoustic profiles mapped to reasoning intents."""
    INSTRUCTIVE = "instructive"      # EXPLAIN: slow, clear, measured
    ANALYTICAL = "analytical"        # COMPARE: dynamic, varied pitch
    INFORMATIVE = "informative"      # GENERAL: neutral, steady
    URGENT = "urgent"                # BREAKING_NEWS: fast, high energy
    CAUTIOUS = "cautious"            # EVALUATE (fail): slow, careful
    CONFIDENT = "confident"          # COMPUTE (high conf): assertive
    STEP_BY_STEP = "step_by_step"    # CAUSAL chain: measured, pausing


# Intent -> prosody mapping
INTENT_PROSODY: dict[str, ProsodyProfile] = {
    "EXPLAIN": ProsodyProfile.INSTRUCTIVE,
    "COMPARE": ProsodyProfile.ANALYTICAL,
    "GENERAL": ProsodyProfile.INFORMATIVE,
    "COMPUTE": ProsodyProfile.CONFIDENT,
    "EVALUATE": ProsodyProfile.CAUTIOUS,
    "CAUSAL": ProsodyProfile.STEP_BY_STEP,
    "BREAKING_NEWS": ProsodyProfile.URGENT,
}

# Prosody -> acoustic parameters
PROSODY_PARAMS: dict[ProsodyProfile, dict[str, Any]] = {
    ProsodyProfile.INSTRUCTIVE: {
        "rate": "slow", "pitch": "medium", "volume": "medium",
        "pause_between_sentences": 0.5,
    },
    ProsodyProfile.ANALYTICAL: {
        "rate": "medium", "pitch": "medium-high", "volume": "medium",
        "pause_between_sentences": 0.3,
    },
    ProsodyProfile.INFORMATIVE: {
        "rate": "medium", "pitch": "medium", "volume": "medium",
        "pause_between_sentences": 0.3,
    },
    ProsodyProfile.URGENT: {
        "rate": "fast", "pitch": "high", "volume": "loud",
        "pause_between_sentences": 0.1,
    },
    ProsodyProfile.CAUTIOUS: {
        "rate": "slow", "pitch": "low", "volume": "soft",
        "pause_between_sentences": 0.6,
    },
    ProsodyProfile.CONFIDENT: {
        "rate": "medium", "pitch": "medium", "volume": "loud",
        "pause_between_sentences": 0.2,
    },
    ProsodyProfile.STEP_BY_STEP: {
        "rate": "slow", "pitch": "medium", "volume": "medium",
        "pause_between_sentences": 0.8,
    },
}


@dataclass
class SpeechOutput:
    """Result of TTS synthesis.

    Attributes:
        text: The original text.
        ssml: SSML-annotated version for standard TTS engines.
        prosody: The prosody profile used.
        audio_bytes: Raw audio bytes (None if no TTS engine available).
        params: Acoustic parameters applied.
    """
    text: str
    ssml: str
    prosody: ProsodyProfile
    audio_bytes: bytes | None = None
    params: dict[str, Any] = field(default_factory=dict)


class VocalTract:
    """Text-to-speech with prosody modulated by reasoning intent.

    Maps reasoning intents (EXPLAIN, COMPARE, BREAKING_NEWS, etc.) to
    acoustic profiles that control speed, pitch, volume, and pausing.

    When a TTS engine is provided, synthesizes audio. Otherwise, produces
    SSML markup that any standard TTS service can consume.

    Args:
        tts_engine: Optional TTS engine with a .synthesize(text, **params)
            method that returns audio bytes. If None, only SSML is produced.
    """

    def __init__(self, tts_engine: Any | None = None) -> None:
        self._engine = tts_engine

    def speak(
        self,
        text: str,
        intent: str = "GENERAL",
    ) -> SpeechOutput:
        """Convert text to speech with intent-modulated prosody.

        Args:
            text: Text to speak.
            intent: Reasoning intent (EXPLAIN, COMPARE, COMPUTE, etc.).

        Returns:
            SpeechOutput with text, SSML, prosody, and optional audio.
        """
        prosody = INTENT_PROSODY.get(
            intent.upper(), ProsodyProfile.INFORMATIVE,
        )
        params = dict(PROSODY_PARAMS.get(prosody, {}))
        ssml = self._to_ssml(text, prosody, params)

        audio = None
        if self._engine is not None:
            try:
                audio = self._engine.synthesize(text, **params)
            except Exception:
                pass

        return SpeechOutput(
            text=text,
            ssml=ssml,
            prosody=prosody,
            audio_bytes=audio,
            params=params,
        )

    def _to_ssml(
        self,
        text: str,
        prosody: ProsodyProfile,
        params: dict[str, Any],
    ) -> str:
        """Generate SSML markup from text and prosody parameters.

        Produces W3C-compatible SSML that any TTS engine can consume.
        """
        rate = params.get("rate", "medium")
        pitch = params.get("pitch", "medium")
        volume = params.get("volume", "medium")
        pause_s = params.get("pause_between_sentences", 0.3)
        pause_ms = int(pause_s * 1000)

        # Split into sentences and add pauses
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        parts = ['<speak>']
        parts.append(
            f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
        )

        for i, sentence in enumerate(sentences):
            parts.append(f"  {sentence}.")
            if i < len(sentences) - 1:
                parts.append(f'  <break time="{pause_ms}ms"/>')

        parts.append("</prosody>")
        parts.append("</speak>")
        return "\n".join(parts)

    @property
    def has_engine(self) -> bool:
        """Whether a TTS engine is available."""
        return self._engine is not None
