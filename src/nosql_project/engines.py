"""Engine protocols and baseline/real NLP implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import Settings

LOGGER = logging.getLogger(__name__)


class SttEngine(Protocol):
    """Contract for speech-to-text engines."""

    async def transcribe(self, audio_bytes: bytes, language: str) -> str:
        """Convert audio bytes to text."""
        raise NotImplementedError


class NlpEngine(Protocol):
    """Contract for text generation engines."""

    async def generate_reply(self, prompt: str, language: str) -> str:
        """Generate a reply from text input."""
        raise NotImplementedError


class TtsEngine(Protocol):
    """Contract for text-to-speech engines."""

    async def synthesize(self, text: str, language: str) -> bytes:
        """Convert text to audio bytes."""
        raise NotImplementedError


@dataclass(slots=True)
class SimpleSttEngine:
    """Small mock STT engine for local MVP tests."""

    delay_seconds: float = 0.02

    async def transcribe(self, audio_bytes: bytes, language: str) -> str:
        """Decode bytes as UTF-8 and fallback to a canned sentence."""
        del language
        await asyncio.sleep(self.delay_seconds)
        transcript = audio_bytes.decode("utf-8", errors="ignore").strip()
        if transcript:
            return transcript
        return "Je n'ai pas bien entendu."


@dataclass(slots=True)
class WhisperSttEngine:
    """Speech-to-text engine backed by faster-whisper."""

    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    _model: Any | None = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            whisper_module: Any = importlib.import_module("faster_whisper")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with "
                "`python -m pip install faster-whisper`."
            ) from exc

        whisper_model_class: Any = getattr(whisper_module, "WhisperModel", None)
        if whisper_model_class is None:
            raise RuntimeError("WhisperModel is unavailable in faster-whisper.")

        self._model = whisper_model_class(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        LOGGER.info(
            "Loaded STT model: faster-whisper %s (device=%s, compute_type=%s)",
            self.model_size,
            self.device,
            self.compute_type,
        )
        return self._model

    def warmup(self) -> None:
        """Eagerly load model resources."""
        _ = self._ensure_model()

    async def transcribe(self, audio_bytes: bytes, language: str) -> str:
        """Convert audio bytes to text using faster-whisper."""
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, language)

    def _transcribe_sync(self, audio_bytes: bytes, language: str) -> str:
        model = self._ensure_model()
        if not audio_bytes:
            return "Je n'ai pas bien entendu."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = Path(tmp_file.name)

        try:
            language_hint = language.strip().lower() or None
            segments, _ = model.transcribe(
                str(temp_path),
                language=language_hint,
                beam_size=self.beam_size,
                vad_filter=True,
            )
            text_parts: list[str] = []
            for segment in segments:
                segment_text = str(getattr(segment, "text", "")).strip()
                if segment_text:
                    text_parts.append(segment_text)
            transcript = " ".join(text_parts).strip()
            if transcript:
                return transcript
            return "Je n'ai pas bien entendu."
        finally:
            temp_path.unlink(missing_ok=True)


@dataclass(slots=True)
class RuleBasedNlpEngine:
    """Simple rule-based NLP baseline compatible with CPU-only runtime."""

    delay_seconds: float = 0.02

    async def generate_reply(self, prompt: str, language: str) -> str:
        """Generate a deterministic answer for MVP validation."""
        del language
        await asyncio.sleep(self.delay_seconds)
        normalized = prompt.lower().strip()
        if "bonjour" in normalized:
            return "Bonjour, je suis pret a t'aider."
        if "merci" in normalized:
            return "Avec plaisir."
        if normalized.endswith("?"):
            return "Question recue. Je traite ta demande en mode asynchrone."
        return f"Message recu: {prompt}"


@dataclass(slots=True)
class TransformersNlpEngine:
    """NLP engine backed by Hugging Face Transformers text2text generation."""

    model_name: str = "google/flan-t5-small"
    max_new_tokens: int = 80
    num_beams: int = 2
    temperature: float = 0.7
    local_files_only: bool = False
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)

    def _ensure_runtime(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            transformers_module: Any = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformers is not installed. Install with "
                "`python -m pip install transformers`."
            ) from exc

        tokenizer_class: Any = getattr(transformers_module, "AutoTokenizer", None)
        tf_model_class: Any = getattr(transformers_module, "TFAutoModelForSeq2SeqLM", None)
        if tokenizer_class is None or tf_model_class is None:
            raise RuntimeError("Transformers TensorFlow seq2seq API is unavailable.")

        try:
            self._tokenizer = tokenizer_class.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model = tf_model_class.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                use_safetensors=False,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError(
                f"Unable to load TensorFlow model '{self.model_name}': {exc}"
            ) from exc
        LOGGER.info("Loaded NLP model: %s", self.model_name)
        return self._tokenizer, self._model

    def warmup(self) -> None:
        """Eagerly load model resources."""
        _ = self._ensure_runtime()

    async def generate_reply(self, prompt: str, language: str) -> str:
        """Generate a response using an actual sequence-to-sequence model."""
        return await asyncio.to_thread(self._generate_sync, prompt, language)

    # pylint: disable=too-many-locals
    def _generate_sync(self, prompt: str, language: str) -> str:
        tokenizer, model = self._ensure_runtime()
        framed_prompt = self._build_prompt(prompt, language)
        do_sample = self.temperature > 0
        inputs = tokenizer(
            framed_prompt,
            return_tensors="tf",
            truncation=True,
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": do_sample,
            "no_repeat_ngram_size": 3,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(self.temperature, 1e-5)
        output_ids: Any = model.generate(**inputs, **generation_kwargs)
        first_pass = self._decode_generated_text(tokenizer, output_ids)
        cleaned_first_pass = self._sanitize_generated_text(first_pass, prompt)
        if cleaned_first_pass:
            return cleaned_first_pass

        retry_prompt = self._build_retry_prompt(prompt, language)
        retry_inputs = tokenizer(
            retry_prompt,
            return_tensors="tf",
            truncation=True,
        )
        retry_ids: Any = model.generate(**retry_inputs, **generation_kwargs)
        second_pass = self._decode_generated_text(tokenizer, retry_ids)
        cleaned_second_pass = self._sanitize_generated_text(second_pass, prompt)
        if cleaned_second_pass:
            return cleaned_second_pass

        prompt_normalized = prompt.strip().lower()
        if "comment vas" in prompt_normalized:
            return "Je vais bien, merci. Et toi ?"
        if "salut" in prompt_normalized or "bonjour" in prompt_normalized:
            return "Salut, je vais bien. Tu veux parler de quoi ?"
        return "Je suis la pour aider. Peux-tu preciser ta demande ?"

    @staticmethod
    def _build_prompt(prompt: str, language: str) -> str:
        normalized_lang = language.strip().lower()
        if normalized_lang == "fr":
            return f"Question: {prompt}\nReponse en francais courte:"
        return f"Answer naturally to this user message: {prompt}"

    @staticmethod
    def _build_retry_prompt(prompt: str, language: str) -> str:
        normalized_lang = language.strip().lower()
        if normalized_lang == "fr":
            return f"Utilisateur: {prompt}\nAssistant (une phrase):"
        return f"Question: {prompt}\nAnswer:"

    @staticmethod
    def _decode_generated_text(tokenizer: Any, output_ids: Any) -> str:
        if getattr(output_ids, "shape", (0,))[0] > 0:
            return tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()
        return ""

    @staticmethod
    # pylint: disable=too-many-return-statements,too-many-branches
    def _sanitize_generated_text(generated_text: str, prompt: str) -> str:
        cleaned = generated_text.strip()
        if not cleaned:
            return ""
        prompt_clean = prompt.strip()
        prompt_lower = prompt_clean.lower()
        lower = cleaned.lower()

        markers = (
            "assistant:",
            "reponse:",
            "response:",
            "answer:",
        )
        for marker in markers:
            index = lower.rfind(marker)
            if index >= 0:
                candidate = cleaned[index + len(marker) :].strip()
                if candidate:
                    cleaned = candidate
                    lower = cleaned.lower()
                break

        if prompt_lower and lower == prompt_lower:
            return ""
        if lower.startswith("reponds "):
            return ""
        if "message utilisateur" in lower and prompt_lower in lower:
            return ""
        if "question:" in lower and prompt_lower in lower:
            return ""
        if "assistant" in lower and prompt_lower in lower:
            return ""
        if "utilisateur" in lower and prompt_lower in lower:
            return ""
        if "useateur" in lower and prompt_lower in lower:
            return ""
        if "francophone" in lower and "repond" in lower:
            return ""
        if "reponse en francais" in lower:
            return ""
        if "francais courte" in lower:
            return ""
        if prompt_lower and lower.startswith(prompt_lower):
            stripped = cleaned[len(prompt_clean) :].strip(" :-")
            if not stripped:
                return ""
            cleaned = stripped

        return cleaned


@dataclass(slots=True)
class SimpleTtsEngine:
    """Small mock TTS engine serializing text to bytes."""

    delay_seconds: float = 0.02

    async def synthesize(self, text: str, language: str) -> bytes:
        """Encode the output text as UTF-8 bytes."""
        await asyncio.sleep(self.delay_seconds)
        payload = f"[{language}] {text}"
        return payload.encode("utf-8")


@dataclass(slots=True)
class PiperTtsEngine:
    """Text-to-speech engine backed by the Piper CLI."""

    executable: str = "piper"
    model_path: str = ""
    speaker_id: int | None = None

    def warmup(self) -> None:
        """Validate executable and model path."""
        resolved_exec = shutil.which(self.executable)
        if resolved_exec is None:
            raise RuntimeError(
                f"Piper executable not found: {self.executable}. "
                "Set TTS_PIPER_EXECUTABLE or add piper to PATH."
            )
        if not self.model_path.strip():
            raise RuntimeError("TTS_PIPER_MODEL_PATH must be configured.")
        if not Path(self.model_path).exists():
            raise RuntimeError(f"Piper model file not found: {self.model_path}")

    async def synthesize(self, text: str, language: str) -> bytes:
        """Generate a wav payload from text using Piper."""
        return await asyncio.to_thread(self._synthesize_sync, text, language)

    def _synthesize_sync(self, text: str, language: str) -> bytes:
        del language
        self.warmup()
        normalized_text = text.strip()
        if not normalized_text:
            return b""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = Path(tmp_file.name)

        command = [
            self.executable,
            "--model",
            self.model_path,
            "--output_file",
            str(output_path),
        ]
        if self.speaker_id is not None:
            command.extend(["--speaker", str(self.speaker_id)])

        try:
            completed = subprocess.run(
                command,
                input=normalized_text,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(f"Piper synthesis failed: {stderr}")
            audio_bytes = output_path.read_bytes()
            if not audio_bytes:
                raise RuntimeError("Piper did not produce any audio output.")
            return audio_bytes
        finally:
            output_path.unlink(missing_ok=True)


def create_nlp_engine(settings: "Settings") -> NlpEngine:
    """Create NLP backend from settings."""
    backend = settings.nlp_backend.strip().lower()
    if backend == "rule_based":
        return RuleBasedNlpEngine()
    if backend == "transformers":
        return TransformersNlpEngine(
            model_name=settings.nlp_model_name,
            max_new_tokens=settings.nlp_max_new_tokens,
            num_beams=settings.nlp_num_beams,
            temperature=settings.nlp_temperature,
            local_files_only=settings.nlp_local_files_only,
        )
    raise ValueError(f"Unsupported NLP_BACKEND value: {settings.nlp_backend}")


def create_stt_engine(settings: "Settings") -> SttEngine:
    """Create STT backend from settings."""
    backend = settings.stt_backend.strip().lower()
    if backend == "simple":
        return SimpleSttEngine()
    if backend == "whisper":
        return WhisperSttEngine(
            model_size=settings.stt_model_size,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            beam_size=settings.stt_beam_size,
        )
    raise ValueError(f"Unsupported STT_BACKEND value: {settings.stt_backend}")


def create_tts_engine(settings: "Settings") -> TtsEngine:
    """Create TTS backend from settings."""
    backend = settings.tts_backend.strip().lower()
    if backend == "simple":
        return SimpleTtsEngine()
    if backend == "piper":
        speaker = settings.tts_piper_speaker_id
        speaker_id = speaker if speaker >= 0 else None
        return PiperTtsEngine(
            executable=settings.tts_piper_executable,
            model_path=settings.tts_piper_model_path,
            speaker_id=speaker_id,
        )
    raise ValueError(f"Unsupported TTS_BACKEND value: {settings.tts_backend}")
