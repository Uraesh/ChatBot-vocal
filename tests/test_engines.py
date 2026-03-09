"""Tests for NLP engine selection."""

from __future__ import annotations

import pytest

from nosql_project.config import Settings
from nosql_project.engines import (
    PiperTtsEngine,
    RuleBasedNlpEngine,
    SimpleSttEngine,
    SimpleTtsEngine,
    TransformersNlpEngine,
    WhisperSttEngine,
    create_nlp_engine,
    create_stt_engine,
    create_tts_engine,
)


def test_create_nlp_engine_rule_based() -> None:
    """Factory should build the rule-based NLP engine."""
    settings = Settings(nlp_backend="rule_based")
    engine = create_nlp_engine(settings)
    assert isinstance(engine, RuleBasedNlpEngine)


def test_create_nlp_engine_transformers() -> None:
    """Factory should build the transformers NLP engine."""
    settings = Settings(nlp_backend="transformers", nlp_model_name="google/flan-t5-small")
    engine = create_nlp_engine(settings)
    assert isinstance(engine, TransformersNlpEngine)
    assert engine.model_name == "google/flan-t5-small"


def test_create_nlp_engine_invalid_backend() -> None:
    """Invalid backend values must raise ValueError."""
    settings = Settings(nlp_backend="unsupported-backend")
    with pytest.raises(ValueError, match="Unsupported NLP_BACKEND"):
        create_nlp_engine(settings)


def test_create_stt_engine_simple() -> None:
    """Factory should build the simple STT engine."""
    settings = Settings(stt_backend="simple")
    engine = create_stt_engine(settings)
    assert isinstance(engine, SimpleSttEngine)


def test_create_stt_engine_whisper() -> None:
    """Factory should build the whisper STT engine."""
    settings = Settings(stt_backend="whisper", stt_model_size="tiny")
    engine = create_stt_engine(settings)
    assert isinstance(engine, WhisperSttEngine)
    assert engine.model_size == "tiny"


def test_create_stt_engine_invalid_backend() -> None:
    """Invalid STT backend values must raise ValueError."""
    settings = Settings(stt_backend="invalid-stt")
    with pytest.raises(ValueError, match="Unsupported STT_BACKEND"):
        create_stt_engine(settings)


def test_create_tts_engine_simple() -> None:
    """Factory should build the simple TTS engine."""
    settings = Settings(tts_backend="simple")
    engine = create_tts_engine(settings)
    assert isinstance(engine, SimpleTtsEngine)


def test_create_tts_engine_piper() -> None:
    """Factory should build the piper TTS engine."""
    settings = Settings(
        tts_backend="piper",
        tts_piper_executable="piper",
        tts_piper_model_path="voice.onnx",
        tts_piper_speaker_id=0,
    )
    engine = create_tts_engine(settings)
    assert isinstance(engine, PiperTtsEngine)
    assert engine.model_path == "voice.onnx"
    assert engine.speaker_id == 0


def test_create_tts_engine_invalid_backend() -> None:
    """Invalid TTS backend values must raise ValueError."""
    settings = Settings(tts_backend="invalid-tts")
    with pytest.raises(ValueError, match="Unsupported TTS_BACKEND"):
        create_tts_engine(settings)
