"""FastAPI application exposing chat and voice endpoints."""
# pylint: disable=duplicate-code

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import binascii
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
from io import StringIO
import logging
from pathlib import Path
import re
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .config import Settings
from .engines import (
    NlpEngine,
    PiperTtsEngine,
    RuleBasedNlpEngine,
    SimpleSttEngine,
    SimpleTtsEngine,
    SttEngine,
    TtsEngine,
    TransformersNlpEngine,
    create_nlp_engine,
    create_stt_engine,
    create_tts_engine,
    WhisperSttEngine,
)
from .mongo_utils import create_interaction_collection
from .pipeline import AsyncVoicePipeline, PipelineError
from .schemas import ChatRequest, ChatResponse, HealthResponse, VoiceRequest, VoiceResponse

LOGGER = logging.getLogger(__name__)
SETTINGS = Settings.from_env()
WEB_DIR = Path(__file__).resolve().parent / "web"


@dataclass(slots=True)
class InteractionRecord:
    """Single conversation sample for offline analysis export."""

    recorded_at: datetime
    request_id: str
    session_id: str
    channel: str
    user_input: str
    assistant_output: str


class InteractionStore:
    """In-memory bounded store used to export interactions as CSV."""

    def __init__(self, max_records: int = 10_000) -> None:
        self._records: deque[InteractionRecord] = deque(maxlen=max_records)
        self._lock = asyncio.Lock()

    async def append(
        self,
        *,
        request_id: str,
        session_id: str,
        channel: str,
        user_input: str,
        assistant_output: str,
    ) -> None:
        """Add a new interaction row."""
        record = InteractionRecord(
            recorded_at=datetime.now(timezone.utc),
            request_id=request_id,
            session_id=session_id.strip(),
            channel=channel.strip(),
            user_input=user_input.strip(),
            assistant_output=assistant_output.strip(),
        )
        async with self._lock:
            self._records.append(record)

    async def snapshot(self, session_id: str | None = None) -> list[InteractionRecord]:
        """Return a stable copy of interactions, optionally filtered by session."""
        normalized = (session_id or "").strip()
        async with self._lock:
            if not normalized:
                return list(self._records)
            return [record for record in self._records if record.session_id == normalized]


class MongoInteractionStore:
    """MongoDB-backed interaction store for long-term analysis."""

    def __init__(
        self,
        *,
        mongo_uri: str,
        mongo_database: str,
        mongo_collection: str,
        server_selection_timeout_ms: int,
    ) -> None:
        self._client, self._collection = create_interaction_collection(
            mongo_uri=mongo_uri,
            mongo_database=mongo_database,
            mongo_collection=mongo_collection,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )

    def warmup(self) -> None:
        """Ping MongoDB and ensure indexes."""
        self._client.admin.command("ping")
        self._collection.create_index([("session_id", 1), ("recorded_at", 1)])
        self._collection.create_index("request_id", unique=True)

    async def append(
        self,
        *,
        request_id: str,
        session_id: str,
        channel: str,
        user_input: str,
        assistant_output: str,
    ) -> None:
        """Insert interaction document into MongoDB."""
        document = {
            "recorded_at": datetime.now(timezone.utc),
            "request_id": request_id.strip(),
            "session_id": session_id.strip(),
            "channel": channel.strip(),
            "user_input": user_input.strip(),
            "assistant_output": assistant_output.strip(),
        }
        await asyncio.to_thread(self._collection.insert_one, document)

    async def snapshot(
        self,
        *,
        session_id: str | None,
        limit: int,
    ) -> list[InteractionRecord]:
        """Read interaction documents from MongoDB sorted by time."""
        return await asyncio.to_thread(
            self._snapshot_sync,
            session_id=session_id,
            limit=limit,
        )

    def _snapshot_sync(self, *, session_id: str | None, limit: int) -> list[InteractionRecord]:
        query: dict[str, str] = {}
        normalized_session = (session_id or "").strip()
        if normalized_session:
            query["session_id"] = normalized_session

        cursor = self._collection.find(query).sort("recorded_at", 1).limit(limit)
        records: list[InteractionRecord] = []
        for document in cursor:
            recorded_at = document.get("recorded_at")
            if isinstance(recorded_at, datetime):
                if recorded_at.tzinfo is None:
                    recorded_at = recorded_at.replace(tzinfo=timezone.utc)
            else:
                recorded_at = datetime.now(timezone.utc)
            records.append(
                InteractionRecord(
                    recorded_at=recorded_at,
                    request_id=str(document.get("request_id", "")),
                    session_id=str(document.get("session_id", "")),
                    channel=str(document.get("channel", "")),
                    user_input=str(document.get("user_input", "")),
                    assistant_output=str(document.get("assistant_output", "")),
                )
            )
        return records

    async def close(self) -> None:
        """Close MongoDB client."""
        await asyncio.to_thread(self._client.close)


def _build_interactions_csv(records: list[InteractionRecord]) -> str:
    """Build CSV content from interaction records."""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "recorded_at_utc",
            "request_id",
            "session_id",
            "channel",
            "user_input",
            "assistant_output",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record.recorded_at.isoformat(),
                record.request_id,
                record.session_id,
                record.channel,
                record.user_input,
                record.assistant_output,
            ]
        )
    return buffer.getvalue()


def _safe_filename_token(value: str) -> str:
    """Normalize arbitrary text to a short filename-safe token."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    if not normalized:
        return "session"
    return normalized[:64]


def _build_nlp_engine(settings: Settings) -> NlpEngine:
    try:
        engine = create_nlp_engine(settings)
        if isinstance(engine, TransformersNlpEngine):
            engine.warmup()
        LOGGER.info(
            "NLP backend selected: %s (model=%s)",
            settings.nlp_backend,
            settings.nlp_model_name,
        )
        return engine
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if settings.nlp_fallback_to_rule_based:
            message = str(exc).splitlines()[0] if str(exc) else "unknown error"
            LOGGER.warning(
                "NLP backend '%s' failed (%s: %s). Falling back to rule_based.",
                settings.nlp_backend,
                type(exc).__name__,
                message,
            )
            return RuleBasedNlpEngine()
        raise


def _build_stt_engine(settings: Settings) -> SttEngine:
    try:
        engine = create_stt_engine(settings)
        if isinstance(engine, WhisperSttEngine):
            engine.warmup()
        LOGGER.info(
            "STT backend selected: %s (model=%s)",
            settings.stt_backend,
            settings.stt_model_size,
        )
        return engine
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if settings.stt_fallback_to_simple:
            message = str(exc).splitlines()[0] if str(exc) else "unknown error"
            LOGGER.warning(
                "STT backend '%s' failed (%s: %s). Falling back to simple.",
                settings.stt_backend,
                type(exc).__name__,
                message,
            )
            return SimpleSttEngine()
        raise


def _build_tts_engine(settings: Settings) -> TtsEngine:
    try:
        engine = create_tts_engine(settings)
        if isinstance(engine, PiperTtsEngine):
            engine.warmup()
        LOGGER.info(
            "TTS backend selected: %s",
            settings.tts_backend,
        )
        return engine
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if settings.tts_fallback_to_simple:
            message = str(exc).splitlines()[0] if str(exc) else "unknown error"
            LOGGER.warning(
                "TTS backend '%s' failed (%s: %s). Falling back to simple.",
                settings.tts_backend,
                type(exc).__name__,
                message,
            )
            return SimpleTtsEngine()
        raise


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create and dispose runtime components for each app lifecycle."""
    application.state.settings = SETTINGS
    application.state.nlp_engine = _build_nlp_engine(SETTINGS)
    application.state.interactions = InteractionStore(
        max_records=SETTINGS.interactions_memory_max_records
    )
    application.state.mongo_interactions = None
    if SETTINGS.interactions_use_mongo:
        try:
            mongo_store = MongoInteractionStore(
                mongo_uri=SETTINGS.mongo_uri,
                mongo_database=SETTINGS.mongo_database,
                mongo_collection=SETTINGS.interactions_mongo_collection,
                server_selection_timeout_ms=SETTINGS.interactions_mongo_timeout_ms,
            )
            mongo_store.warmup()
            application.state.mongo_interactions = mongo_store
            LOGGER.info(
                "Interaction Mongo store enabled: %s.%s",
                SETTINGS.mongo_database,
                SETTINGS.interactions_mongo_collection,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "Interaction Mongo store disabled (%s: %s). Falling back to memory only.",
                type(exc).__name__,
                str(exc).splitlines()[0] if str(exc) else "unknown error",
            )
    stt_engine = _build_stt_engine(SETTINGS)
    tts_engine = _build_tts_engine(SETTINGS)
    application.state.pipeline = AsyncVoicePipeline(
        stt_engine=stt_engine,
        nlp_engine=application.state.nlp_engine,
        tts_engine=tts_engine,
        language=SETTINGS.default_language,
        queue_max_size=SETTINGS.queue_max_size,
    )
    pipeline: AsyncVoicePipeline = application.state.pipeline
    try:
        await pipeline.start()
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Service startup failed.")
        raise
    LOGGER.info("Service started.")
    try:
        yield
    finally:
        mongo_store = getattr(application.state, "mongo_interactions", None)
        if isinstance(mongo_store, MongoInteractionStore):
            try:
                await mongo_store.close()
            except Exception:  # pylint: disable=broad-exception-caught
                LOGGER.exception("Interaction Mongo store shutdown failed.")
                raise
        try:
            await pipeline.stop()
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.exception("Service shutdown failed.")
            raise
        LOGGER.info("Service stopped.")


app = FastAPI(title=SETTINGS.app_name, lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def web_interface() -> FileResponse:
    """Serve the local single-page UI used for manual chat/voice tests."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Interface web introuvable.")
    return FileResponse(index_path)


def _get_settings(request: Request) -> Settings:
    """Read validated settings from app state."""
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    raise HTTPException(status_code=500, detail="Configuration indisponible.")


def _get_pipeline(request: Request) -> AsyncVoicePipeline:
    """Read pipeline instance from app state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if isinstance(pipeline, AsyncVoicePipeline):
        return pipeline
    raise HTTPException(status_code=500, detail="Pipeline indisponible.")


def _get_nlp_engine(request: Request) -> NlpEngine:
    """Read NLP engine instance from app state."""
    nlp_engine = getattr(request.app.state, "nlp_engine", None)
    if nlp_engine is None or not hasattr(nlp_engine, "generate_reply"):
        raise HTTPException(status_code=500, detail="Moteur NLP indisponible.")
    return nlp_engine


def _get_interaction_store(request: Request) -> InteractionStore:
    """Read interaction store instance from app state."""
    interaction_store = getattr(request.app.state, "interactions", None)
    if isinstance(interaction_store, InteractionStore):
        return interaction_store
    raise HTTPException(status_code=500, detail="Journal des interactions indisponible.")


def _get_mongo_interaction_store(request: Request) -> MongoInteractionStore | None:
    """Read optional Mongo interaction store instance from app state."""
    mongo_store = getattr(request.app.state, "mongo_interactions", None)
    if isinstance(mongo_store, MongoInteractionStore):
        return mongo_store
    return None


async def _store_interaction(
    *,
    interaction_store: InteractionStore,
    mongo_store: MongoInteractionStore | None,
    request_id: str,
    session_id: str,
    channel: str,
    user_input: str,
    assistant_output: str,
) -> None:
    """Persist interaction rows without failing user requests on storage errors."""
    try:
        await interaction_store.append(
            request_id=request_id,
            session_id=session_id,
            channel=channel,
            user_input=user_input,
            assistant_output=assistant_output,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Failed to store interaction row request_id=%s", request_id)
    if mongo_store is not None:
        try:
            await mongo_store.append(
                request_id=request_id,
                session_id=session_id,
                channel=channel,
                user_input=user_input,
                assistant_output=assistant_output,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.exception("Failed to persist interaction in Mongo request_id=%s", request_id)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health and queue depth indicators."""
    pipeline = _get_pipeline(request)
    try:
        response = HealthResponse(
            status="ok",
            pending_requests=pipeline.pending_requests,
            audio_queue_size=pipeline.audio_queue_size,
            transcript_queue_size=pipeline.transcript_queue_size,
            response_queue_size=pipeline.response_queue_size,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Health endpoint failed.")
        raise HTTPException(status_code=500, detail="Erreur interne health.") from exc
    LOGGER.debug(
        "Health checked pending=%s stt_queue=%s nlp_queue=%s tts_queue=%s",
        response.pending_requests,
        response.audio_queue_size,
        response.transcript_queue_size,
        response.response_queue_size,
    )
    return response


@app.get("/exports/conversations.csv")
async def export_conversations_csv(
    request: Request,
    session_id: str | None = None,
) -> Response:
    """Download interactions as CSV for hallucination and quality analysis."""
    interaction_store = _get_interaction_store(request)
    settings = _get_settings(request)
    mongo_store = _get_mongo_interaction_store(request)
    if mongo_store is not None:
        records = await mongo_store.snapshot(
            session_id=session_id,
            limit=settings.interactions_export_limit,
        )
    else:
        records = await interaction_store.snapshot(session_id=session_id)
    csv_payload = _build_interactions_csv(records)
    filename_suffix = _safe_filename_token(session_id or "all_sessions")
    headers = {
        "Content-Disposition": f'attachment; filename="conversations_{filename_suffix}.csv"'
    }
    return Response(content=csv_payload, media_type="text/csv; charset=utf-8", headers=headers)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Run a text-only inference request."""
    settings = _get_settings(request)
    nlp_engine = _get_nlp_engine(request)
    interaction_store = _get_interaction_store(request)
    mongo_store = _get_mongo_interaction_store(request)
    request_id = str(uuid4())
    LOGGER.info(
        "Chat request received request_id=%s session_id=%s text_length=%s",
        request_id,
        payload.session_id,
        len(payload.text),
    )
    try:
        reply = await nlp_engine.generate_reply(
            prompt=payload.text,
            language=settings.default_language,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Chat generation failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Erreur interne chat.") from exc

    LOGGER.info(
        "Chat request completed request_id=%s response_length=%s",
        request_id,
        len(reply),
    )
    await _store_interaction(
        interaction_store=interaction_store,
        mongo_store=mongo_store,
        request_id=request_id,
        session_id=payload.session_id,
        channel="chat",
        user_input=payload.text,
        assistant_output=reply,
    )
    return ChatResponse(request_id=request_id, response=reply)


@app.post("/voice", response_model=VoiceResponse)
async def voice(request: Request, payload: VoiceRequest) -> VoiceResponse:
    """Run full asynchronous STT -> NLP -> TTS inference."""
    settings = _get_settings(request)
    pipeline = _get_pipeline(request)
    interaction_store = _get_interaction_store(request)
    mongo_store = _get_mongo_interaction_store(request)
    LOGGER.info(
        "Voice request received session_id=%s audio_base64_length=%s",
        payload.session_id,
        len(payload.audio_base64),
    )
    try:
        audio_bytes = base64.b64decode(payload.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        LOGGER.warning("Voice request invalid base64 session_id=%s", payload.session_id)
        raise HTTPException(status_code=400, detail="audio_base64 invalide.") from exc

    if not audio_bytes:
        LOGGER.warning("Voice request empty decoded audio session_id=%s", payload.session_id)
        raise HTTPException(status_code=400, detail="audio vide.")

    request_id = "n/a"
    try:
        request_id = await pipeline.submit_audio(
            session_id=payload.session_id,
            audio_bytes=audio_bytes,
        )
        result = await pipeline.wait_for_result(
            request_id=request_id,
            timeout_seconds=settings.voice_timeout_seconds,
        )
    except TimeoutError as exc:
        LOGGER.warning("Voice request timeout request_id=%s", request_id)
        raise HTTPException(status_code=504, detail="Traitement vocal timeout.") from exc
    except PipelineError as exc:
        LOGGER.error("Voice pipeline error request_id=%s error=%s", request_id, exc)
        raise HTTPException(status_code=500, detail="Erreur interne pipeline vocal.") from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Voice endpoint unexpected failure request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Erreur interne voice.") from exc

    encoded_audio = base64.b64encode(result.audio_bytes).decode("ascii")
    LOGGER.info(
        "Voice request completed request_id=%s response_length=%s audio_length=%s",
        result.request_id,
        len(result.response_text),
        len(result.audio_bytes),
    )
    await _store_interaction(
        interaction_store=interaction_store,
        mongo_store=mongo_store,
        request_id=result.request_id,
        session_id=result.session_id,
        channel="voice",
        user_input=result.transcript,
        assistant_output=result.response_text,
    )
    return VoiceResponse(
        request_id=result.request_id,
        text=result.response_text,
        audio_base64=encoded_audio,
        transcript=result.transcript,
    )
