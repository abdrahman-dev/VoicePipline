from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class GeneralSettings:
    log_level: str = os.getenv("ROBOT_LOG_LEVEL", "INFO")
    student_name: str = os.getenv("ROBOT_STUDENT_NAME", "Student")
    default_session_language: str = os.getenv("ROBOT_DEFAULT_SESSION_LANGUAGE", "ar")  # "ar" or "en"


@dataclass(frozen=True)
class ASRSettings:
    provider: str = os.getenv("ROBOT_ASR_PROVIDER", "google")  # "google" or future options
    sample_rate: int = int(os.getenv("ROBOT_ASR_SAMPLE_RATE", "16000"))
    language_mode: str = os.getenv("ROBOT_ASR_LANGUAGE_MODE", "auto")  # "auto", "en", "ar"
    default_record_duration_seconds: float = float(os.getenv("ROBOT_ASR_DEFAULT_DURATION_SEC", "5.0"))
    supported_languages: Dict[str, str] = field(default_factory=dict)  # populated in get_settings()


@dataclass(frozen=True)
class VADSettings:
    sample_rate: int = int(os.getenv("ROBOT_VAD_SAMPLE_RATE", "16000"))
    initial_threshold: float = float(os.getenv("ROBOT_VAD_THRESHOLD", "0.60"))
    chunk_duration_ms: int = int(os.getenv("ROBOT_VAD_CHUNK_MS", "32"))  # try 30 - 40 if not working
    pre_speech_buffer_seconds: float = float(os.getenv("ROBOT_VAD_PRE_ROLL_SEC", "0.30"))
    min_speech_seconds: float = float(os.getenv("ROBOT_VAD_MIN_SPEECH_SEC", "0.50"))
    silence_timeout_seconds: float = float(os.getenv("ROBOT_VAD_SILENCE_TIMEOUT_SEC", "0.80"))
    max_abs_amplitude: float = float(os.getenv("ROBOT_VAD_MAX_ABS_AMP", "1.0"))
    torch_threads: int = int(os.getenv("ROBOT_VAD_TORCH_THREADS", "1"))
    model_local_path: str = os.getenv("ROBOT_VAD_MODEL_LOCAL_PATH", "")  # optional
    model_hub_repo: str = os.getenv("ROBOT_VAD_HUB_REPO", "snakers4/silero-vad")
    model_hub_name: str = os.getenv("ROBOT_VAD_HUB_NAME", "silero_vad")
    model_trust_repo: bool = os.getenv("ROBOT_VAD_TRUST_REPO", "true").lower() in ("1", "true", "yes")


"""Updated LLMSettings dataclass for openrouter API"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class LLMSettings:
    """LLM configuration for openrouter API integration."""
    
    provider: str = os.getenv("ROBOT_LLM_PROVIDER", "openrouter")
    
    # OpenRouter API credentials
    openrouter_api_key: str = field(default=os.getenv("ROBOT_OPENROUTER_API_KEY", "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"))
    openrouter_model: str = os.getenv("ROBOT_OPENROUTER_MODEL", "openrouter/free")
    
    # Timeouts
    openrouter_availability_timeout_seconds: int = int(
        os.getenv("ROBOT_OPENROUTER_AVAILABILITY_TIMEOUT_SEC", "5")
    )
    request_timeout_seconds: int = int(
        os.getenv("ROBOT_LLM_REQUEST_TIMEOUT_SEC", "90")
    )
    summarization_timeout_seconds: int = int(
        os.getenv("ROBOT_LLM_SUMMARIZE_TIMEOUT_SEC", "60")
    )
    
    # Database & Memory
    db_path: str = os.getenv("ROBOT_LLM_DB_PATH", "robot_sessions.db")
    sliding_window_size: int = int(os.getenv("ROBOT_LLM_WINDOW_SIZE", "10"))
    
    # System Prompts
    system_prompt_arabic: str = (
        "أنت روبوت تعليمي ذكي مساعد للطلاب في رحلتهم التعليمية.\n\n"
        "**أهدافك الرئيسية:**\n"
        "1. شرح المفاهيم بطريقة بسيطة وممتعة\n"
        "2. تشجيع الفضول والأسئلة\n"
        "3. مساعدة الطالب على فهم الدروس\n"
        "4. توفير أمثلة عملية ذات صلة\n\n"
        "**قواعد التفاعل:**\n"
        "- تحدث بلغة عربية فصحى مع لمسات من اللهجة المصرية (غير رسمي وودود)\n"
        "- اجعل الإجابات قصيرة وسهلة الفهم (جملتين لثلاث جمل كحد أقصى)\n"
        "- إذا لم تفهم السؤال، اطلب توضيح برفق\n"
        "- استخدم تشبيهات وأمثلة من الحياة اليومية\n"
        "- كن متحمساً وإيجابياً دائماً\n\n"
        "**ممنوع:**\n"
        "- إعطاء الإجابة الكاملة مباشرة (ساعد الطالب يفكر بنفسه)\n"
        "- الإجابات الطويلة جداً\n"
        "- استخدام مصطلحات معقدة بدون شرح"
    )
    
    system_prompt_english: str = (
        "You are an intelligent educational robot assistant helping students in their learning journey.\n\n"
        "**Your main goals:**\n"
        "1. Explain concepts in a simple and engaging way\n"
        "2. Encourage curiosity and questions\n"
        "3. Help students understand lessons\n"
        "4. Provide practical relevant examples\n\n"
        "**Interaction rules:**\n"
        "- Speak in clear, simple English\n"
        "- Keep answers short and easy to understand (2-3 sentences max)\n"
        "- If you don't understand, ask for clarification politely\n"
        "- Use analogies and examples from daily life\n"
        "- Be enthusiastic and positive always\n\n"
        "**Forbidden:**\n"
        "- Giving complete answers directly (help students think)\n"
        "- Very long responses\n"
        "- Using complex terms without explanation"
    )


@dataclass(frozen=True)
class TTSSettings:
    engine: str = os.getenv("ROBOT_TTS_ENGINE", "edge_tts")  # future: "coqui", "piper", etc
    audio_temp_dir: str = os.getenv("ROBOT_TTS_TEMP_DIR", tempfile.gettempdir())
    audio_filename_template: str = os.getenv("ROBOT_TTS_AUDIO_TEMPLATE", "tts_{turn_id}.mp3")
    voice_map: Dict[str, str] = field(default_factory=dict)  # populated in get_settings()
    pygame_poll_interval_seconds: float = float(os.getenv("ROBOT_TTS_POLL_SEC", "0.05"))


@dataclass(frozen=True)
class Settings:
    general: GeneralSettings
    asr: ASRSettings
    vad: VADSettings
    llm: LLMSettings
    tts: TTSSettings


def get_settings() -> Settings:
    # These are defined in settings (not modules) so providers are swappable/configurable.
    asr_supported = {"en": "en-US", "ar": "ar-EG"}
    voice_map = {"en": "en-US-GuyNeural", "ar": "ar-SA-HamedNeural"}

    general = GeneralSettings()
    asr = ASRSettings(supported_languages=asr_supported)
    vad = VADSettings()
    llm = LLMSettings()
    tts = TTSSettings(voice_map=voice_map)
    return Settings(general=general, asr=asr, vad=vad, llm=llm, tts=tts)

