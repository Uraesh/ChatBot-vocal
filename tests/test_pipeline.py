"""Tests for the asynchronous voice pipeline."""

from __future__ import annotations

import asyncio

from nosql_project.engines import RuleBasedNlpEngine, SimpleSttEngine, SimpleTtsEngine
from nosql_project.pipeline import AsyncVoicePipeline, PipelineError


class FailingSttEngine:
    """STT engine that always fails."""

    async def transcribe(self, audio_bytes: bytes, language: str) -> str:
        del audio_bytes, language
        raise RuntimeError("stt failure")


class FailingNlpEngine:
    """NLP engine that always fails."""

    async def generate_reply(self, prompt: str, language: str) -> str:
        del prompt, language
        raise RuntimeError("nlp failure")


class FailingTtsEngine:
    """TTS engine that always fails."""

    async def synthesize(self, text: str, language: str) -> bytes:
        del text, language
        raise RuntimeError("tts failure")


class SlowTtsEngine:
    """TTS engine that is intentionally slow."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds

    async def synthesize(self, text: str, language: str) -> bytes:
        del text, language
        await asyncio.sleep(self.delay_seconds)
        return b"audio"


def test_pipeline_roundtrip() -> None:
    """The pipeline should produce a non-empty response and audio payload."""

    async def scenario() -> None:
        pipeline = AsyncVoicePipeline(
            stt_engine=SimpleSttEngine(delay_seconds=0.0),
            nlp_engine=RuleBasedNlpEngine(delay_seconds=0.0),
            tts_engine=SimpleTtsEngine(delay_seconds=0.0),
        )
        await pipeline.start()
        try:
            request_id = await pipeline.submit_audio("session-1", b"bonjour")
            result = await pipeline.wait_for_result(request_id, timeout_seconds=2.0)

            assert result.request_id == request_id
            assert result.response_text
            assert result.audio_bytes
        finally:
            await pipeline.stop()

    asyncio.run(scenario())


def test_pipeline_uses_prompt_builder() -> None:
    """The pipeline should allow injecting session-aware prompt building."""

    class RecordingNlpEngine:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def generate_reply(self, prompt: str, language: str) -> str:
            del language
            self.prompts.append(prompt)
            return "ok"

    async def scenario() -> None:
        recorder = RecordingNlpEngine()

        async def prompt_builder(session_id: str, transcript: str) -> str:
            return f"Utilisateur: {session_id}\nAssistant: memo\nUtilisateur: {transcript}\nAssistant:"

        pipeline = AsyncVoicePipeline(
            stt_engine=SimpleSttEngine(delay_seconds=0.0),
            nlp_engine=recorder,
            tts_engine=SimpleTtsEngine(delay_seconds=0.0),
            prompt_builder=prompt_builder,
        )
        await pipeline.start()
        try:
            request_id = await pipeline.submit_audio("session-42", b"bonjour")
            await pipeline.wait_for_result(request_id, timeout_seconds=2.0)
            assert recorder.prompts
            assert "session-42" in recorder.prompts[0]
        finally:
            await pipeline.stop()

    asyncio.run(scenario())


def test_pipeline_timeout() -> None:
    """wait_for_result should timeout if a stage is too slow."""

    async def scenario() -> None:
        pipeline = AsyncVoicePipeline(
            stt_engine=SimpleSttEngine(delay_seconds=0.0),
            nlp_engine=RuleBasedNlpEngine(delay_seconds=0.0),
            tts_engine=SlowTtsEngine(delay_seconds=0.2),
        )
        await pipeline.start()
        try:
            request_id = await pipeline.submit_audio("session-1", b"bonjour")
            try:
                await pipeline.wait_for_result(request_id, timeout_seconds=0.05)
            except TimeoutError:
                pass
            else:
                raise AssertionError("TimeoutError was not raised.")
        finally:
            await pipeline.stop()

    asyncio.run(scenario())


def test_pipeline_stt_failure() -> None:
    """STT failure should surface as PipelineError."""

    async def scenario() -> None:
        pipeline = AsyncVoicePipeline(
            stt_engine=FailingSttEngine(),
            nlp_engine=RuleBasedNlpEngine(delay_seconds=0.0),
            tts_engine=SimpleTtsEngine(delay_seconds=0.0),
        )
        await pipeline.start()
        try:
            request_id = await pipeline.submit_audio("session-1", b"bonjour")
            try:
                await pipeline.wait_for_result(request_id, timeout_seconds=1.0)
            except PipelineError:
                pass
            else:
                raise AssertionError("PipelineError was not raised.")
        finally:
            await pipeline.stop()

    asyncio.run(scenario())


def test_pipeline_nlp_failure() -> None:
    """NLP failure should surface as PipelineError."""

    async def scenario() -> None:
        pipeline = AsyncVoicePipeline(
            stt_engine=SimpleSttEngine(delay_seconds=0.0),
            nlp_engine=FailingNlpEngine(),
            tts_engine=SimpleTtsEngine(delay_seconds=0.0),
        )
        await pipeline.start()
        try:
            request_id = await pipeline.submit_audio("session-1", b"bonjour")
            try:
                await pipeline.wait_for_result(request_id, timeout_seconds=1.0)
            except PipelineError:
                pass
            else:
                raise AssertionError("PipelineError was not raised.")
        finally:
            await pipeline.stop()

    asyncio.run(scenario())


def test_pipeline_tts_failure() -> None:
    """TTS failure should surface as PipelineError."""

    async def scenario() -> None:
        pipeline = AsyncVoicePipeline(
            stt_engine=SimpleSttEngine(delay_seconds=0.0),
            nlp_engine=RuleBasedNlpEngine(delay_seconds=0.0),
            tts_engine=FailingTtsEngine(),
        )
        await pipeline.start()
        try:
            request_id = await pipeline.submit_audio("session-1", b"bonjour")
            try:
                await pipeline.wait_for_result(request_id, timeout_seconds=1.0)
            except PipelineError:
                pass
            else:
                raise AssertionError("PipelineError was not raised.")
        finally:
            await pipeline.stop()

    asyncio.run(scenario())
