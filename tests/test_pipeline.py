"""Tests for the asynchronous voice pipeline."""

from __future__ import annotations

import asyncio

from nosql_project.engines import RuleBasedNlpEngine, SimpleSttEngine, SimpleTtsEngine
from nosql_project.pipeline import AsyncVoicePipeline


def test_pipeline_roundtrip() -> None:
    """The pipeline should produce a non-empty response and audio payload."""

    async def scenario() -> None:
        pipeline = AsyncVoicePipeline(
            stt_engine=SimpleSttEngine(delay_seconds=0.0),
            nlp_engine=RuleBasedNlpEngine(delay_seconds=0.0),
            tts_engine=SimpleTtsEngine(delay_seconds=0.0),
        )
        await pipeline.start()
        request_id = await pipeline.submit_audio("session-1", b"bonjour")
        result = await pipeline.wait_for_result(request_id, timeout_seconds=2.0)
        await pipeline.stop()

        assert result.request_id == request_id
        assert result.response_text
        assert result.audio_bytes

    asyncio.run(scenario())
