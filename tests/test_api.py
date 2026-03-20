"""API endpoint tests."""

from __future__ import annotations

import base64
import csv
from collections.abc import Iterator
from io import StringIO

from fastapi.testclient import TestClient
import pytest

from nosql_project.api import app


@pytest.fixture(name="client")
def fixture_client() -> Iterator[TestClient]:
    """Build a test client with application lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Health endpoint should return queue metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pending_requests"] >= 0
    assert body["audio_queue_size"] >= 0
    assert body["transcript_queue_size"] >= 0
    assert body["response_queue_size"] >= 0


def test_root_web_interface(client: TestClient) -> None:
    """Root endpoint should serve the local web interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Vocal IA" in response.text


def test_chat_endpoint(client: TestClient) -> None:
    """Chat endpoint should return a request id and a text response."""
    response = client.post(
        "/chat",
        json={"session_id": "demo-session", "text": "bonjour"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["response"]
    assert body["message_limit"] >= 1
    assert body["messages_used"] == 1
    assert body["messages_remaining"] == body["message_limit"] - 1


def test_chat_stream_endpoint(client: TestClient) -> None:
    """Chat stream endpoint should return server-sent events."""
    response = client.post(
        "/chat/stream",
        json={"session_id": "demo-session", "text": "bonjour"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: meta" in response.text


def test_voice_endpoint_success(client: TestClient) -> None:
    """Voice endpoint should process a valid base64 audio payload."""
    audio_base64 = base64.b64encode(b"bonjour").decode("ascii")
    response = client.post(
        "/voice",
        json={"session_id": "demo-session", "audio_base64": audio_base64},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["text"]
    assert body["transcript"]
    assert body["messages_used"] == 1
    decoded_audio = base64.b64decode(body["audio_base64"], validate=True)
    assert decoded_audio


def test_session_state_and_history(client: TestClient) -> None:
    """Session endpoints should expose quota and replay recent messages."""
    session_id = "memory-session"

    state_before = client.get(f"/session/{session_id}")
    assert state_before.status_code == 200
    assert state_before.json()["messages_used"] == 0

    response = client.post(
        "/chat",
        json={"session_id": session_id, "text": "je suis une femme, bonsoir"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["preferred_title"] == "madame"

    state_after = client.get(f"/session/{session_id}")
    assert state_after.status_code == 200
    state_payload = state_after.json()
    assert state_payload["messages_used"] == 1
    assert state_payload["preferred_title"] == "madame"
    assert state_payload["suggested_greeting"]

    history_response = client.get(f"/session/{session_id}/history")
    assert history_response.status_code == 200
    messages = history_response.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_voice_endpoint_invalid_base64(client: TestClient) -> None:
    """Voice endpoint should return 400 for invalid base64 payload."""
    response = client.post(
        "/voice",
        json={"session_id": "demo-session", "audio_base64": "invalid-base64%%%"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "audio_base64 invalide."


def test_export_csv_for_session(client: TestClient) -> None:
    """CSV export should include chat and voice interactions for a session."""
    session_id = "csv-session"
    chat_response = client.post(
        "/chat",
        json={"session_id": session_id, "text": "bonjour"},
    )
    assert chat_response.status_code == 200

    audio_base64 = base64.b64encode(b"salut").decode("ascii")
    voice_response = client.post(
        "/voice",
        json={"session_id": session_id, "audio_base64": audio_base64},
    )
    assert voice_response.status_code == 200

    export_response = client.get(f"/exports/conversations.csv?session_id={session_id}")
    assert export_response.status_code == 200
    assert "text/csv" in export_response.headers["content-type"]
    assert "attachment" in export_response.headers["content-disposition"]

    rows = list(csv.DictReader(StringIO(export_response.text)))
    assert rows
    assert any(
        row["channel"] == "chat"
        and row["session_id"] == session_id
        and row["user_input"] == "bonjour"
        and row["assistant_output"]
        for row in rows
    )
    assert any(
        row["channel"] == "voice"
        and row["session_id"] == session_id
        and row["user_input"] == "salut"
        and row["assistant_output"]
        for row in rows
    )
