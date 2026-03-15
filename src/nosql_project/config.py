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
    voice_timeout_seconds: float = 120.0      # augmenté : Phi-3 est plus lent que Flan-T5
    default_language: str = "fr"
    debug: bool = False
    subtitles_file_path: str = "Fr/fr.txt"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_database: str = "nosql_project"
    mongo_collection: str = "dialogues"
    mongo_batch_size: int = 1000
    chat_history_max_turns: int = 6
    chat_stream_chunk_chars: int = 24
    interactions_use_mongo: bool = False
    interactions_mongo_collection: str = "interaction_logs"
    interactions_mongo_timeout_ms: int = 5000
    interactions_memory_max_records: int = 10000
    interactions_export_limit: int = 50000

    # --- NLP (Phi-3 Mini GGUF) ---
    nlp_backend: str = "rule_based"
    nlp_model_name: str = "google/flan-t5-small"   # conservé pour backend transformers
    nlp_model_path: str = ""                        # NOUVEAU : chemin vers le .gguf Phi-3
    nlp_max_new_tokens: int = 400
    nlp_num_beams: int = 4                          # utilisé uniquement par transformers
    nlp_temperature: float = 0.7
    nlp_n_threads: int = 3                          # NOUVEAU : threads CPU pour llama.cpp
    nlp_local_files_only: bool = False
    nlp_fallback_to_rule_based: bool = True
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    openrouter_site_url: str = ""
    openrouter_app_name: str = ""
    openrouter_timeout_seconds: float = 30.0

    # --- STT (Whisper) ---
    stt_backend: str = "simple"
    stt_model_size: str = "base"                    # "base" recommandé pour le français
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 5                          # 5 pour meilleure précision (était 1)
    stt_fallback_to_simple: bool = True

    # --- TTS (Piper) ---
    tts_backend: str = "simple"
    tts_piper_executable: str = r"D:\Licience 3 IA-BD\No sql\NoSql Project\tools\piper\piper\piper.exe"
    tts_piper_model_path: str = ""
    tts_piper_speaker_id: int = -1
    tts_fallback_to_simple: bool = True

    @staticmethod
    def from_env() -> "Settings":
        """Build settings from environment variables."""
        queue_raw = os.getenv("QUEUE_MAX_SIZE", "128")
        timeout_raw = os.getenv("VOICE_TIMEOUT_SECONDS", "120.0")
        mongo_batch_raw = os.getenv("MONGO_BATCH_SIZE", "1000")
        chat_history_turns_raw = os.getenv("CHAT_HISTORY_MAX_TURNS", "6")
        chat_stream_chunk_chars_raw = os.getenv("CHAT_STREAM_CHUNK_CHARS", "24")
        interactions_timeout_raw = os.getenv("INTERACTIONS_MONGO_TIMEOUT_MS", "5000")
        interactions_memory_max_records_raw = os.getenv(
            "INTERACTIONS_MEMORY_MAX_RECORDS", "10000",
        )
        interactions_export_limit_raw = os.getenv("INTERACTIONS_EXPORT_LIMIT", "50000")
        nlp_max_tokens_raw = os.getenv("NLP_MAX_NEW_TOKENS", "400")
        nlp_num_beams_raw = os.getenv("NLP_NUM_BEAMS", "4")
        nlp_temperature_raw = os.getenv("NLP_TEMPERATURE", "0.7")
        nlp_n_threads_raw = os.getenv("NLP_N_THREADS", "3")
        stt_beam_size_raw = os.getenv("STT_BEAM_SIZE", "5")
        tts_piper_speaker_id_raw = os.getenv("TTS_PIPER_SPEAKER_ID", "-1")

        app_name = os.getenv("APP_NAME", "NoSQL Async Voice Chatbot")
        openrouter_timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30.0")
        return Settings(
            app_name=app_name,
            queue_max_size=int(queue_raw),
            voice_timeout_seconds=float(timeout_raw),
            default_language=os.getenv("DEFAULT_LANGUAGE", "fr"),
            debug=_as_bool(os.getenv("DEBUG", "false"), default=False),
            subtitles_file_path=os.getenv("SUBTITLES_FILE_PATH", "Fr/fr.txt"),
            mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            mongo_database=os.getenv("MONGO_DATABASE", "nosql_project"),
            mongo_collection=os.getenv("MONGO_COLLECTION", "dialogues"),
            mongo_batch_size=int(mongo_batch_raw),
            chat_history_max_turns=int(chat_history_turns_raw),
            chat_stream_chunk_chars=int(chat_stream_chunk_chars_raw),
            interactions_use_mongo=_as_bool(
                os.getenv("INTERACTIONS_USE_MONGO", "false"), default=False,
            ),
            interactions_mongo_collection=os.getenv(
                "INTERACTIONS_MONGO_COLLECTION", "interaction_logs",
            ),
            interactions_mongo_timeout_ms=int(interactions_timeout_raw),
            interactions_memory_max_records=int(interactions_memory_max_records_raw),
            interactions_export_limit=int(interactions_export_limit_raw),
            nlp_backend=os.getenv("NLP_BACKEND", "rule_based"),
            nlp_model_name=os.getenv("NLP_MODEL_NAME", "google/flan-t5-small"),
            nlp_model_path=os.getenv("NLP_MODEL_PATH", ""),
            nlp_max_new_tokens=int(nlp_max_tokens_raw),
            nlp_num_beams=int(nlp_num_beams_raw),
            nlp_temperature=float(nlp_temperature_raw),
            nlp_n_threads=int(nlp_n_threads_raw),
            nlp_local_files_only=_as_bool(
                os.getenv("NLP_LOCAL_FILES_ONLY", "false"), default=False,
            ),
            nlp_fallback_to_rule_based=_as_bool(
                os.getenv("NLP_FALLBACK_TO_RULE_BASED", "true"), default=True,
            ),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_model=os.getenv("OPENROUTER_MODEL", ""),
            openrouter_site_url=os.getenv("OPENROUTER_SITE_URL", ""),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", app_name),
            openrouter_timeout_seconds=float(openrouter_timeout_raw),
            stt_backend=os.getenv("STT_BACKEND", "simple"),
            stt_model_size=os.getenv("STT_MODEL_SIZE", "base"),
            stt_device=os.getenv("STT_DEVICE", "cpu"),
            stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "int8"),
            stt_beam_size=int(stt_beam_size_raw),
            stt_fallback_to_simple=_as_bool(
                os.getenv("STT_FALLBACK_TO_SIMPLE", "true"), default=True,
            ),
            tts_backend=os.getenv("TTS_BACKEND", "simple"),
            tts_piper_executable=os.getenv("TTS_PIPER_EXECUTABLE", r"D:\Licience 3 IA-BD\No sql\NoSql Project\tools\piper\piper\piper.exe"),
            tts_piper_model_path=os.getenv("TTS_PIPER_MODEL_PATH", ""),
            tts_piper_speaker_id=int(tts_piper_speaker_id_raw),
            tts_fallback_to_simple=_as_bool(
                os.getenv("TTS_FALLBACK_TO_SIMPLE", "true"), default=True,
            ),
        )
    
