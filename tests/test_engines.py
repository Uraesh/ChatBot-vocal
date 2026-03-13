"""Tests for engine selection helpers."""

from __future__ import annotations

import pytest

from nosql_project.config import Settings
from nosql_project.engines import (
    HybridNlpEngine,
    Phi3NlpEngine,
    PiperTtsEngine,
    RuleBasedNlpEngine,
    SimpleSttEngine,
    SimpleTtsEngine,
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


def test_create_nlp_engine_hybrid() -> None:
    """Factory should build the hybrid NLP engine."""
    settings = Settings(
        nlp_backend="hybrid",
        nlp_model_name="google/flan-t5-small",
        nlp_model_path="D:/models/Phi-3-mini-4k-instruct-q4.gguf",
        nlp_max_new_tokens=120,
        nlp_temperature=0.5,
        nlp_n_threads=2,
    )
    engine = create_nlp_engine(settings)
    assert isinstance(engine, HybridNlpEngine)
    assert engine.flan.model_name == "google/flan-t5-small"
    assert engine.phi3.model_path == "D:/models/Phi-3-mini-4k-instruct-q4.gguf"
    assert engine.phi3.max_new_tokens == 120
    assert engine.phi3.temperature == 0.5
    assert engine.phi3.n_threads == 2


def test_create_nlp_engine_phi3() -> None:
    """Factory should build the Phi-3 NLP engine."""
    settings = Settings(
        nlp_backend="phi3",
        nlp_model_path="D:/models/Phi-3-mini-4k-instruct-q4.gguf",
        nlp_max_new_tokens=80,
        nlp_temperature=0.4,
        nlp_n_threads=4,
    )
    engine = create_nlp_engine(settings)
    assert isinstance(engine, Phi3NlpEngine)
    assert engine.model_path == "D:/models/Phi-3-mini-4k-instruct-q4.gguf"
    assert engine.max_new_tokens == 80
    assert engine.temperature == 0.4
    assert engine.n_threads == 4


def test_create_nlp_engine_invalid_backend() -> None:
    """Invalid backend values must raise ValueError."""
    settings = Settings(nlp_backend="unsupported-backend")
    with pytest.raises(ValueError, match="non reconnu"):
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
    with pytest.raises(ValueError, match="non reconnu"):
        create_stt_engine(settings)


def test_create_tts_engine_simple() -> None:
    """Factory should build the simple TTS engine."""
    settings = Settings(tts_backend="simple")
    engine = create_tts_engine(settings)
    assert isinstance(engine, SimpleTtsEngine)


def test_create_tts_engine_piper_with_speaker() -> None:
    """Factory should build the piper TTS engine with a speaker id."""
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


def test_create_tts_engine_piper_without_speaker() -> None:
    """Negative speaker id should disable the Piper speaker selection."""
    settings = Settings(
        tts_backend="piper",
        tts_piper_executable="piper",
        tts_piper_model_path="voice.onnx",
        tts_piper_speaker_id=-1,
    )
    engine = create_tts_engine(settings)
    assert isinstance(engine, PiperTtsEngine)
    assert engine.speaker_id is None


def test_create_tts_engine_invalid_backend() -> None:
    """Invalid TTS backend values must raise ValueError."""
    settings = Settings(tts_backend="invalid-tts")
    with pytest.raises(ValueError, match="non reconnu"):
        create_tts_engine(settings)
