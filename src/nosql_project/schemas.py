"""Data models used by the API and pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class AudioTask:
    """Input audio task waiting for STT."""

    request_id: str
    session_id: str
    audio_bytes: bytes
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class TranscriptTask:
    """Text produced by STT and waiting for NLP."""

    request_id: str
    session_id: str
    transcript: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ResponseTask:
    """Text produced by NLP and waiting for TTS."""

    request_id: str
    session_id: str
    transcript: str
    response_text: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class VoiceResult:
    """Final result produced by TTS."""

    request_id: str
    session_id: str
    transcript: str
    response_text: str
    audio_bytes: bytes
    created_at: datetime
    completed_at: datetime = field(default_factory=utc_now)


class ChatRequest(BaseModel):
    """Incoming text chat request."""

    session_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    """Text chat response."""

    request_id: str
    response: str
    message_limit: int
    messages_used: int
    messages_remaining: int
    preferred_title: str | None = None


class VoiceRequest(BaseModel):
    """Incoming voice request encoded in base64."""

    session_id: str = Field(min_length=1, max_length=128)
    audio_base64: str = Field(min_length=1)


class VoiceResponse(BaseModel):
    """Voice response including text and synthesized audio."""

    request_id: str
    text: str
    audio_base64: str
    transcript: str | None = None
    message_limit: int
    messages_used: int
    messages_remaining: int
    preferred_title: str | None = None


class HealthResponse(BaseModel):
    """API health and queue metrics."""

    status: str
    pending_requests: int
    audio_queue_size: int
    transcript_queue_size: int
    response_queue_size: int


class SessionStateResponse(BaseModel):
    """Session quota and personalization state exposed to the UI."""

    session_id: str
    message_limit: int
    messages_used: int
    messages_remaining: int
    preferred_title: str | None = None
    suggested_greeting: str


class ConversationMessageResponse(BaseModel):
    """Single message restored in the UI conversation history."""

    role: str
    content: str
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    """Recent session messages restored after page reload."""

    session_id: str
    messages: list[ConversationMessageResponse]
