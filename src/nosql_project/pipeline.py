"""Asynchronous pipeline orchestrating STT, NLP and TTS stages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Final
from uuid import uuid4

from .engines import NlpEngine, SttEngine, TtsEngine
from .schemas import AudioTask, ResponseTask, TranscriptTask, VoiceResult

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineMetrics:
    """Runtime counters useful for health checks."""

    completed_requests: int = 0
    failed_requests: int = 0


class PipelineError(RuntimeError):
    """Raised when one stage of the asynchronous pipeline fails."""


class AsyncVoicePipeline:
    """Concurrent three-stage voice pipeline using asyncio queues."""

    def __init__(
        self,
        stt_engine: SttEngine,
        nlp_engine: NlpEngine,
        tts_engine: TtsEngine,
        *,
        language: str = "fr",
        queue_max_size: int = 128,
    ) -> None:
        self._stt_engine = stt_engine
        self._nlp_engine = nlp_engine
        self._tts_engine = tts_engine
        self._language = language
        self._queue_max_size = queue_max_size

        self._audio_queue: asyncio.Queue[AudioTask] = asyncio.Queue(maxsize=queue_max_size)
        self._transcript_queue: asyncio.Queue[TranscriptTask] = asyncio.Queue(
            maxsize=queue_max_size
        )
        self._response_queue: asyncio.Queue[ResponseTask] = asyncio.Queue(maxsize=queue_max_size)
        self._result_futures: dict[str, asyncio.Future[VoiceResult]] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._metrics = PipelineMetrics()

    @property
    def metrics(self) -> PipelineMetrics:
        """Expose current pipeline metrics."""
        return self._metrics

    @property
    def audio_queue_size(self) -> int:
        """Return current size of the STT input queue."""
        return self._audio_queue.qsize()

    @property
    def transcript_queue_size(self) -> int:
        """Return current size of the NLP input queue."""
        return self._transcript_queue.qsize()

    @property
    def response_queue_size(self) -> int:
        """Return current size of the TTS input queue."""
        return self._response_queue.qsize()

    @property
    def pending_requests(self) -> int:
        """Return number of requests waiting for completion."""
        return len(self._result_futures)

    async def start(self) -> None:
        """Start background worker tasks."""
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._stt_worker(), name="stt-worker"),
            asyncio.create_task(self._nlp_worker(), name="nlp-worker"),
            asyncio.create_task(self._tts_worker(), name="tts-worker"),
        ]
        LOGGER.info(
            "Async pipeline started with %s workers (queue_max_size=%s).",
            len(self._workers),
            self._queue_max_size,
        )

    async def stop(self) -> None:
        """Stop background worker tasks and fail pending requests."""
        if not self._running:
            return
        self._running = False
        for worker in self._workers:
            worker.cancel()
        outcomes = await asyncio.gather(*self._workers, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, Exception) and not isinstance(outcome, asyncio.CancelledError):
                LOGGER.error("Worker exited with unexpected exception: %s", outcome)
        self._workers.clear()
        for future in self._result_futures.values():
            if not future.done():
                future.set_exception(PipelineError("Pipeline stopped before completion."))
                self._metrics.failed_requests += 1
        self._result_futures.clear()
        LOGGER.info(
            "Async pipeline stopped (completed=%s failed=%s).",
            self._metrics.completed_requests,
            self._metrics.failed_requests,
        )

    async def submit_audio(self, session_id: str, audio_bytes: bytes) -> str:
        """Submit audio payload and return a request identifier."""
        if not self._running:
            raise PipelineError("Pipeline is not running.")
        if not session_id.strip():
            raise PipelineError("Session identifier cannot be empty.")
        if not audio_bytes:
            raise PipelineError("Audio payload cannot be empty.")
        request_id = str(uuid4())
        loop = asyncio.get_running_loop()
        self._result_futures[request_id] = loop.create_future()
        await self._audio_queue.put(
            AudioTask(
                request_id=request_id,
                session_id=session_id,
                audio_bytes=audio_bytes,
            )
        )
        LOGGER.debug(
            "Audio task submitted request_id=%s session_id=%s queue_size=%s",
            request_id,
            session_id,
            self._audio_queue.qsize(),
        )
        return request_id

    async def wait_for_result(self, request_id: str, timeout_seconds: float) -> VoiceResult:
        """Wait for a completed result associated to a request id."""
        future = self._result_futures.get(request_id)
        if future is None:
            raise PipelineError(f"Unknown request_id: {request_id}")
        started_at = perf_counter()
        try:
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
            elapsed = perf_counter() - started_at
            LOGGER.info("Request completed request_id=%s elapsed=%.3fs", request_id, elapsed)
            return result
        except TimeoutError:
            elapsed = perf_counter() - started_at
            LOGGER.warning(
                "Request timeout request_id=%s timeout=%.3fs elapsed=%.3fs",
                request_id,
                timeout_seconds,
                elapsed,
            )
            raise
        finally:
            self._result_futures.pop(request_id, None)

    async def _stt_worker(self) -> None:
        while True:
            task = await self._audio_queue.get()
            try:
                transcript = await self._stt_engine.transcribe(task.audio_bytes, self._language)
                await self._transcript_queue.put(
                    TranscriptTask(
                        request_id=task.request_id,
                        session_id=task.session_id,
                        transcript=transcript,
                        created_at=task.created_at,
                    )
                )
            except asyncio.CancelledError:
                LOGGER.info("STT worker cancellation received.")
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._metrics.failed_requests += 1
                self._set_failure(task.request_id, stage="STT", error=exc)
            finally:
                self._audio_queue.task_done()

    async def _nlp_worker(self) -> None:
        while True:
            task = await self._transcript_queue.get()
            try:
                response_text = await self._nlp_engine.generate_reply(
                    task.transcript,
                    self._language,
                )
                await self._response_queue.put(
                    ResponseTask(
                        request_id=task.request_id,
                        session_id=task.session_id,
                        transcript=task.transcript,
                        response_text=response_text,
                        created_at=task.created_at,
                    )
                )
            except asyncio.CancelledError:
                LOGGER.info("NLP worker cancellation received.")
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._metrics.failed_requests += 1
                self._set_failure(task.request_id, stage="NLP", error=exc)
            finally:
                self._transcript_queue.task_done()

    async def _tts_worker(self) -> None:
        while True:
            task = await self._response_queue.get()
            try:
                audio_bytes = await self._tts_engine.synthesize(task.response_text, self._language)
                result = VoiceResult(
                    request_id=task.request_id,
                    session_id=task.session_id,
                    transcript=task.transcript,
                    response_text=task.response_text,
                    audio_bytes=audio_bytes,
                    created_at=task.created_at,
                )
                self._metrics.completed_requests += 1
                future = self._result_futures.get(task.request_id)
                if future is not None and not future.done():
                    future.set_result(result)
            except asyncio.CancelledError:
                LOGGER.info("TTS worker cancellation received.")
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._metrics.failed_requests += 1
                self._set_failure(task.request_id, stage="TTS", error=exc)
            finally:
                self._response_queue.task_done()

    def _set_failure(self, request_id: str, stage: str, error: Exception) -> None:
        """Set request exception without crashing workers."""
        message = f"{stage} stage failed for request {request_id}: {error}"
        LOGGER.exception(message)
        future = self._result_futures.get(request_id)
        if future is not None and not future.done():
            future.set_exception(PipelineError(message))
