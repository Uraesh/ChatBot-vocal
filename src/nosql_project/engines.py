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

_PHI3_SYSTEM_PROMPT = (
    "Tu es un assistant vocal conversationnel francophone sympathique, "
    "comme Siri ou Google Assistant. "
    "Reponds TOUJOURS en francais, de facon naturelle et concise (1 a 3 phrases maximum). "
    "Ne repete jamais la question de l'utilisateur. "
    "Sois chaleureux et direct."
)

_KNOWN_INTENTS = {"salutation", "question", "commande", "remerciement", "inconnu"}


class SttEngine(Protocol):
    """Contract for speech-to-text engines."""
    async def transcribe(self, audio_bytes: bytes, language: str) -> str:
        raise NotImplementedError

class NlpEngine(Protocol):
    """Contract for text generation engines."""
    async def generate_reply(self, prompt: str, language: str) -> str:
        raise NotImplementedError

class TtsEngine(Protocol):
    """Contract for text-to-speech engines."""
    async def synthesize(self, text: str, language: str) -> bytes:
        raise NotImplementedError


def _extract_last_user_message(prompt: str) -> str:
    lines = prompt.strip().splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        for prefix in ("utilisateur:", "user:"):
            if stripped.lower().startswith(prefix):
                candidate = stripped[len(prefix):].strip()
                if candidate:
                    return candidate
    return prompt.strip()

def _echo_ratio(response: str, user_input: str) -> float:
    r_words = set(response.split())
    u_words = set(user_input.split())
    if not u_words:
        return 0.0
    return len(r_words & u_words) / len(u_words)

def _sanitize_output(text: str, user_message: str) -> str:
    cleaned = text.strip()
    for token in ("<|end|>", "<|assistant|>", "<|user|>", "<|system|>"):
        cleaned = cleaned.replace(token, "").strip()
    if not cleaned:
        return ""
    user_lower = user_message.strip().lower()
    lower = cleaned.lower()
    if user_lower and (lower == user_lower or _echo_ratio(lower, user_lower) > 0.75):
        return ""
    return cleaned

def _is_whisper_hallucination(text: str) -> bool:
    patterns = {
        "sous-titres réalisés par", "sous-titres par", "merci d'avoir regardé",
        "abonnez-vous", "transcription automatique", "sous-titrage",
        "[musique]", "[music]", "(musique)",
    }
    return any(p in text.lower().strip() for p in patterns)

def _french_fallback_response(user_message: str) -> str:
    lower = user_message.lower().strip()
    if any(w in lower for w in ("bonjour", "salut", "bonsoir", "coucou")):
        return "Bonjour ! Comment puis-je vous aider ?"
    if any(w in lower for w in ("comment vas", "ca va", "comment tu vas")):
        return "Je vais tres bien, merci ! Et vous ?"
    if "merci" in lower:
        return "Avec plaisir ! N'hesitez pas si vous avez besoin d'autre chose."
    if any(w in lower for w in ("qui es-tu", "qui es tu", "qui etes-vous")):
        return "Je suis un assistant vocal IA construit avec FastAPI, Whisper et Piper."
    if lower.endswith("?"):
        return "Bonne question. Pouvez-vous me donner plus de details ?"
    return "Je vous ecoute. N'hesitez pas a preciser votre demande."

def _intent_to_hint(intent: str) -> str:
    hints: dict[str, str] = {
        "salutation":   "L'utilisateur te salue, reponds chaleureusement.",
        "question":     "L'utilisateur pose une question, reponds precisement.",
        "commande":     "L'utilisateur donne une instruction, execute-la ou explique.",
        "remerciement": "L'utilisateur te remercie, reponds avec simplicite.",
        "inconnu":      "Reponds de facon naturelle et demande des precisions si besoin.",
    }
    return hints.get(intent, "")


@dataclass(slots=True)
class SimpleSttEngine:
    """Small mock STT engine for local MVP tests."""
    delay_seconds: float = 0.02

    async def transcribe(self, audio_bytes: bytes, language: str) -> str:
        del language
        await asyncio.sleep(self.delay_seconds)
        transcript = audio_bytes.decode("utf-8", errors="ignore").strip()
        return transcript if transcript else "Je n'ai pas bien entendu."


@dataclass(slots=True)
class WhisperSttEngine:
    """Speech-to-text engine backed by faster-whisper."""
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    _model: Any | None = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            whisper_module: Any = importlib.import_module("faster_whisper")
        except ModuleNotFoundError as exc:
            raise RuntimeError("pip install faster-whisper") from exc
        whisper_class: Any = getattr(whisper_module, "WhisperModel", None)
        if whisper_class is None:
            raise RuntimeError("WhisperModel introuvable.")
        self._model = whisper_class(self.model_size, device=self.device, compute_type=self.compute_type)
        LOGGER.info("STT charge : faster-whisper %s (device=%s)", self.model_size, self.device)
        return self._model

    def warmup(self) -> None:
        _ = self._ensure_model()

    async def transcribe(self, audio_bytes: bytes, language: str) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, language)

    def _transcribe_sync(self, audio_bytes: bytes, language: str) -> str:
        model = self._ensure_model()
        if not audio_bytes:
            return "Je n'ai pas bien entendu."
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = Path(tmp.name)
        try:
            lang_code = language.strip().lower()[:2] if language.strip() else "fr"
            segments, info = model.transcribe(
                str(temp_path), language=lang_code, beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                condition_on_previous_text=False, temperature=0.0,
            )
            LOGGER.debug("STT langue=%s prob=%.2f", info.language, info.language_probability)
            parts: list[str] = [
                str(getattr(seg, "text", "")).strip() for seg in segments
                if str(getattr(seg, "text", "")).strip()
                and not _is_whisper_hallucination(str(getattr(seg, "text", "")))
            ]
            transcript = " ".join(parts).strip()
            return transcript if transcript else "Je n'ai pas bien entendu."
        finally:
            temp_path.unlink(missing_ok=True)


@dataclass(slots=True)
class RuleBasedNlpEngine:
    """Fallback NLP — réponses déterministes sans modèle IA."""
    delay_seconds: float = 0.02

    async def generate_reply(self, prompt: str, language: str) -> str:
        del language
        await asyncio.sleep(self.delay_seconds)
        normalized = _extract_last_user_message(prompt).lower().strip()
        if any(w in normalized for w in ("bonjour", "salut", "bonsoir", "coucou")):
            return "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
        if any(w in normalized for w in ("comment vas", "ca va", "comment tu vas")):
            return "Je vais tres bien, merci ! Et vous ?"
        if "merci" in normalized:
            return "Avec plaisir ! N'hesitez pas si vous avez d'autres questions."
        if any(w in normalized for w in ("qui es-tu", "qui es tu")):
            return "Je suis un assistant vocal IA construit avec FastAPI, Whisper et Piper."
        if normalized.endswith("?"):
            return "Bonne question ! Pouvez-vous me donner plus de details ?"
        return "Message bien recu. Comment puis-je vous aider ?"


@dataclass(slots=True)
class FlanT5NluEngine:
    """
    Module NLU Flan-T5 Small (TensorFlow).
    Etape 1 du pipeline hybride : classification d'intention + reformulation.
    NE genere PAS la reponse finale — prepare le contexte pour Phi-3.

    pip install "tensorflow>=2.15.0" "transformers>=4.44.0,<5.0.0" sentencepiece tf-keras
    """
    model_name: str = "google/flan-t5-small"
    max_new_tokens: int = 60
    local_files_only: bool = False
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)

    def _ensure_runtime(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            tf_module: Any = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError("pip install 'transformers>=4.44.0,<5.0.0'") from exc
        tok_class: Any = getattr(tf_module, "AutoTokenizer", None)
        model_class: Any = getattr(tf_module, "TFAutoModelForSeq2SeqLM", None)
        if tok_class is None or model_class is None:
            raise RuntimeError("API TensorFlow seq2seq introuvable dans transformers.")
        try:
            self._tokenizer = tok_class.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            self._model = model_class.from_pretrained(self.model_name, local_files_only=self.local_files_only, use_safetensors=False)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError(f"Impossible de charger Flan-T5 '{self.model_name}': {exc}") from exc
        LOGGER.info("Flan-T5 NLU charge : %s", self.model_name)
        return self._tokenizer, self._model

    def warmup(self) -> None:
        _ = self._ensure_runtime()

    def classify_intent(self, user_message: str) -> str:
        """Classifier l'intention : salutation/question/commande/remerciement/inconnu."""
        tokenizer, model = self._ensure_runtime()
        prompt = (
            f"Classifie l'intention de ce message en UN seul mot parmi : "
            f"salutation, question, commande, remerciement, inconnu.\n"
            f"Message: {user_message}\nIntention:"
        )
        inputs = tokenizer(prompt, return_tensors="tf", truncation=True, max_length=256)
        output_ids: Any = model.generate(**inputs, max_new_tokens=10, num_beams=2, do_sample=False)
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        for intent in _KNOWN_INTENTS:
            if intent in raw:
                return intent
        return "inconnu"

    def reformulate(self, user_message: str) -> str:
        """Reformuler le message en une phrase claire pour Phi-3."""
        tokenizer, model = self._ensure_runtime()
        prompt = (
            f"Reformule ce message en une phrase francaise claire et complete, "
            f"sans changer le sens : {user_message}"
        )
        inputs = tokenizer(prompt, return_tensors="tf", truncation=True, max_length=256)
        output_ids: Any = model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            num_beams=4, do_sample=False, no_repeat_ngram_size=3,
        )
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        cleaned = _sanitize_output(raw, user_message)
        return cleaned if cleaned else user_message

    def enrich_context(self, user_message: str) -> dict[str, str]:
        """Retourner intention + message reformulé — input de Phi-3."""
        intent = self.classify_intent(user_message)
        reformulated = self.reformulate(user_message)
        LOGGER.debug("Flan-T5 : intent=%s reformulated='%s'", intent, reformulated)
        return {"intent": intent, "reformulated": reformulated}


@dataclass(slots=True)
class Phi3NlpEngine:
    """
    Moteur de generation conversationnelle Phi-3 Mini (GGUF / llama.cpp).
    Etape 2 du pipeline hybride : reçoit le contexte enrichi par Flan-T5.

    pip install llama-cpp-python
    Modele : https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
    Fichier : Phi-3-mini-4k-instruct-q4.gguf (~2.2 Go)
    Variable : NLP_MODEL_PATH=C:/models/Phi-3-mini-4k-instruct-q4.gguf
    """
    model_path: str = ""
    max_new_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.92
    top_k: int = 50
    repeat_penalty: float = 1.1
    n_ctx: int = 4096
    n_threads: int = 3
    _llm: Any | None = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> Any:
        if self._llm is not None:
            return self._llm
        try:
            llama_module: Any = importlib.import_module("llama_cpp")
        except ModuleNotFoundError as exc:
            raise RuntimeError("pip install llama-cpp-python") from exc
        llama_class: Any = getattr(llama_module, "Llama", None)
        if llama_class is None:
            raise RuntimeError("Classe Llama introuvable dans llama_cpp.")
        if self.model_path.strip() and Path(self.model_path).exists():
            LOGGER.info("Chargement Phi-3 Mini (local) depuis %s ...", self.model_path)
            self._llm = llama_class(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        else:
            LOGGER.info("Telechargement Phi-3 Mini depuis Hugging Face (~2.2 Go)...")
            self._llm = llama_class.from_pretrained(
                repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
                filename="Phi-3-mini-4k-instruct-q4.gguf",
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            LOGGER.info("Phi-3 Mini telecharge et mis en cache.")
        LOGGER.info("Phi-3 Mini pret (n_ctx=%s n_threads=%s)", self.n_ctx, self.n_threads)
        return self._llm

    def warmup(self) -> None:
        _ = self._ensure_model()

    def generate(self, intent: str, reformulated_message: str) -> str:
        """Generer une reponse avec contexte enrichi par Flan-T5."""
        llm = self._ensure_model()
        intent_hint = _intent_to_hint(intent)
        formatted_prompt = (
            f"<|system|>\n{_PHI3_SYSTEM_PROMPT} {intent_hint}<|end|>\n"
            f"<|user|>\n{reformulated_message}<|end|>\n"
            f"<|assistant|>\n"
        )
        LOGGER.debug("Phi-3 prompt : %s chars (intent=%s)", len(formatted_prompt), intent)
        output: Any = llm(
            formatted_prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            stop=["<|end|>", "<|user|>", "<|system|>", "\n\n\n"],
            echo=False,
        )
        raw: str = output["choices"][0]["text"].strip()
        cleaned = _sanitize_output(raw, reformulated_message)
        return cleaned if cleaned else _french_fallback_response(reformulated_message)


@dataclass(slots=True)
class HybridNlpEngine:
    """
    Moteur NLP hybride : Flan-T5 (TensorFlow) + Phi-3 Mini (GGUF).

    Flan-T5 comprend bien les instructions → classe l'intention + reformule.
    Phi-3 parle bien → genere une reponse naturelle et conversationnelle.
    Resultat : un chatbot qui comprend bien ET qui parle bien.
    """
    flan: FlanT5NluEngine = field(default_factory=FlanT5NluEngine)
    phi3: Phi3NlpEngine = field(default_factory=Phi3NlpEngine)

    def warmup(self) -> None:
        self.flan.warmup()
        self.phi3.warmup()

    async def generate_reply(self, prompt: str, language: str) -> str:
        del language
        return await asyncio.to_thread(self._generate_sync, prompt)

    def _generate_sync(self, prompt: str) -> str:
        user_message = _extract_last_user_message(prompt)
        context = self.flan.enrich_context(user_message)
        return self.phi3.generate(intent=context["intent"], reformulated_message=context["reformulated"])


@dataclass(slots=True)
class SimpleTtsEngine:
    """Small mock TTS engine serializing text to bytes."""
    delay_seconds: float = 0.02

    async def synthesize(self, text: str, language: str) -> bytes:
        await asyncio.sleep(self.delay_seconds)
        return f"[{language}] {text}".encode("utf-8")


@dataclass(slots=True)
class PiperTtsEngine:
    """Text-to-speech engine backed by the Piper CLI."""
    executable: str = "piper"
    model_path: str = ""
    speaker_id: int | None = None

    def warmup(self) -> None:
        if shutil.which(self.executable) is None:
            raise RuntimeError(f"Piper introuvable : {self.executable}.")
        if not self.model_path.strip():
            raise RuntimeError("TTS_PIPER_MODEL_PATH doit etre configure.")
        if not Path(self.model_path).exists():
            raise RuntimeError(f"Modele Piper introuvable : {self.model_path}")

    async def synthesize(self, text: str, language: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text, language)

    def _synthesize_sync(self, text: str, language: str) -> bytes:
        del language
        self.warmup()
        normalized = text.strip()
        if not normalized:
            return b""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)
        command = [self.executable, "--model", self.model_path, "--output_file", str(output_path)]
        if self.speaker_id is not None:
            command.extend(["--speaker", str(self.speaker_id)])
        try:
            result = subprocess.run(command, input=normalized, capture_output=True, text=True, check=False, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"Piper a echoue : {result.stderr.strip()}")
            audio_bytes = output_path.read_bytes()
            if not audio_bytes:
                raise RuntimeError("Piper n'a produit aucun audio.")
            return audio_bytes
        finally:
            output_path.unlink(missing_ok=True)


def create_nlp_engine(settings: "Settings") -> NlpEngine:
    """
    NLP_BACKEND valides :
      - rule_based : fallback deterministe (aucune dependance)
      - hybrid     : Flan-T5 (TF) + Phi-3 Mini GGUF  <- recommande
      - phi3       : Phi-3 Mini seul (sans Flan-T5)
    """
    backend = settings.nlp_backend.strip().lower()
    if backend == "rule_based":
        return RuleBasedNlpEngine()
    if backend == "hybrid":
        flan = FlanT5NluEngine(model_name=settings.nlp_model_name, local_files_only=settings.nlp_local_files_only)
        phi3 = Phi3NlpEngine(model_path=settings.nlp_model_path, max_new_tokens=settings.nlp_max_new_tokens, temperature=settings.nlp_temperature, n_threads=settings.nlp_n_threads)
        return HybridNlpEngine(flan=flan, phi3=phi3)
    if backend == "phi3":
        return Phi3NlpEngine(model_path=settings.nlp_model_path, max_new_tokens=settings.nlp_max_new_tokens, temperature=settings.nlp_temperature, n_threads=settings.nlp_n_threads)
    raise ValueError(f"NLP_BACKEND '{settings.nlp_backend}' non reconnu. Valeurs valides : rule_based, hybrid, phi3")


def create_stt_engine(settings: "Settings") -> SttEngine:
    backend = settings.stt_backend.strip().lower()
    if backend == "simple":
        return SimpleSttEngine()
    if backend == "whisper":
        return WhisperSttEngine(model_size=settings.stt_model_size, device=settings.stt_device, compute_type=settings.stt_compute_type, beam_size=settings.stt_beam_size)
    raise ValueError(f"STT_BACKEND '{settings.stt_backend}' non reconnu. Valeurs valides : simple, whisper")


def create_tts_engine(settings: "Settings") -> TtsEngine:
    backend = settings.tts_backend.strip().lower()
    if backend == "simple":
        return SimpleTtsEngine()
    if backend == "piper":
        speaker = settings.tts_piper_speaker_id
        return PiperTtsEngine(executable=settings.tts_piper_executable, model_path=settings.tts_piper_model_path, speaker_id=speaker if speaker >= 0 else None)
    raise ValueError(f"TTS_BACKEND '{settings.tts_backend}' non reconnu. Valeurs valides : simple, piper")


if __name__ == "__main__":
    pass
    # Exemple de test rapide :
    # settings = Settings.from_env()
    # nlp = create_nlp_engine(settings)
    # nlp.warmup()
    # prompt = "Utilisateur: Qui es-tu ?"
    # response = asyncio.run(nlp.generate_reply(prompt, language="fr"))
    # print("NLP response:", response)  