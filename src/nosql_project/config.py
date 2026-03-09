"""Runtime settings for the asynchronous voice chatbot service."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _as_bool(value: str, default: bool) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True, slots=True)
# pylint: disable=too-many-instance-attributes
class Settings:
    """Service configuration loaded from environment variables."""

    app_name: str = "NoSQL Async Voice Chatbot"
    queue_max_size: int = 128
    voice_timeout_seconds: float = 20.0
    default_language: str = "fr"
    debug: bool = False
    subtitles_file_path: str = "Fr/fr.txt"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_database: str = "nosql_project"
    mongo_collection: str = "dialogues"
    mongo_batch_size: int = 1000
    interactions_use_mongo: bool = False
    interactions_mongo_collection: str = "interaction_logs"
    interactions_mongo_timeout_ms: int = 5000
    interactions_memory_max_records: int = 10000
    interactions_export_limit: int = 50000
    nlp_backend: str = "rule_based"
    nlp_model_name: str = "google/flan-t5-small"
    nlp_max_new_tokens: int = 80
    nlp_num_beams: int = 2
    nlp_temperature: float = 0.7
    nlp_local_files_only: bool = False
    nlp_fallback_to_rule_based: bool = True
    stt_backend: str = "simple"
    stt_model_size: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_fallback_to_simple: bool = True
    tts_backend: str = "simple"
    tts_piper_executable: str = "piper"
    tts_piper_model_path: str = ""
    tts_piper_speaker_id: int = -1
    tts_fallback_to_simple: bool = True

    @staticmethod
    def from_env() -> "Settings":
        """Build settings from environment variables."""
        queue_raw = os.getenv("QUEUE_MAX_SIZE", "128")
        timeout_raw = os.getenv("VOICE_TIMEOUT_SECONDS", "20.0")
        mongo_batch_raw = os.getenv("MONGO_BATCH_SIZE", "1000")
        interactions_timeout_raw = os.getenv("INTERACTIONS_MONGO_TIMEOUT_MS", "5000")
        interactions_memory_max_records_raw = os.getenv(
            "INTERACTIONS_MEMORY_MAX_RECORDS",
            "10000",
        )
        interactions_export_limit_raw = os.getenv("INTERACTIONS_EXPORT_LIMIT", "50000")
        nlp_max_tokens_raw = os.getenv("NLP_MAX_NEW_TOKENS", "80")
        nlp_num_beams_raw = os.getenv("NLP_NUM_BEAMS", "2")
        nlp_temperature_raw = os.getenv("NLP_TEMPERATURE", "0.7")
        stt_beam_size_raw = os.getenv("STT_BEAM_SIZE", "1")
        tts_piper_speaker_id_raw = os.getenv("TTS_PIPER_SPEAKER_ID", "-1")
        return Settings(
            app_name=os.getenv("APP_NAME", "NoSQL Async Voice Chatbot"),
            queue_max_size=int(queue_raw),
            voice_timeout_seconds=float(timeout_raw),
            default_language=os.getenv("DEFAULT_LANGUAGE", "fr"),
            debug=_as_bool(os.getenv("DEBUG", "false"), default=False),
            subtitles_file_path=os.getenv("SUBTITLES_FILE_PATH", "Fr/fr.txt"),
            mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            mongo_database=os.getenv("MONGO_DATABASE", "nosql_project"),
            mongo_collection=os.getenv("MONGO_COLLECTION", "dialogues"),
            mongo_batch_size=int(mongo_batch_raw),
            interactions_use_mongo=_as_bool(
                os.getenv("INTERACTIONS_USE_MONGO", "false"),
                default=False,
            ),
            interactions_mongo_collection=os.getenv(
                "INTERACTIONS_MONGO_COLLECTION",
                "interaction_logs",
            ),
            interactions_mongo_timeout_ms=int(interactions_timeout_raw),
            interactions_memory_max_records=int(interactions_memory_max_records_raw),
            interactions_export_limit=int(interactions_export_limit_raw),
            nlp_backend=os.getenv("NLP_BACKEND", "rule_based"),
            nlp_model_name=os.getenv("NLP_MODEL_NAME", "google/flan-t5-small"),
            nlp_max_new_tokens=int(nlp_max_tokens_raw),
            nlp_num_beams=int(nlp_num_beams_raw),
            nlp_temperature=float(nlp_temperature_raw),
            nlp_local_files_only=_as_bool(
                os.getenv("NLP_LOCAL_FILES_ONLY", "false"),
                default=False,
            ),
            nlp_fallback_to_rule_based=_as_bool(
                os.getenv("NLP_FALLBACK_TO_RULE_BASED", "true"),
                default=True,
            ),
            stt_backend=os.getenv("STT_BACKEND", "simple"),
            stt_model_size=os.getenv("STT_MODEL_SIZE", "small"),
            stt_device=os.getenv("STT_DEVICE", "cpu"),
            stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "int8"),
            stt_beam_size=int(stt_beam_size_raw),
            stt_fallback_to_simple=_as_bool(
                os.getenv("STT_FALLBACK_TO_SIMPLE", "true"),
                default=True,
            ),
            tts_backend=os.getenv("TTS_BACKEND", "simple"),
            tts_piper_executable=os.getenv("TTS_PIPER_EXECUTABLE", "piper"),
            tts_piper_model_path=os.getenv("TTS_PIPER_MODEL_PATH", ""),
            tts_piper_speaker_id=int(tts_piper_speaker_id_raw),
            tts_fallback_to_simple=_as_bool(
                os.getenv("TTS_FALLBACK_TO_SIMPLE", "true"),
                default=True,
            ),
        )
