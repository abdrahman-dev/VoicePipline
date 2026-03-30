"""ASR Module: Speech-to-Text using Google Speech Recognition.

Supports Arabic (ar-EG) and English (en-US) with automatic language detection.
Requires internet connection for Google Speech API.
"""

import logging
from typing import Optional, Tuple

import speech_recognition as sr
import sounddevice as sd

from config.settings import get_settings

logger = logging.getLogger(__name__)

_SETTINGS = get_settings()
_ASR = _SETTINGS.asr

TARGET_SAMPLE_RATE = _ASR.sample_rate
SUPPORTED_LANGUAGES = _ASR.supported_languages


class ASRModuleError(RuntimeError):
    """Raised when audio recording or transcription fails."""


def record_audio(duration: float = _ASR.default_record_duration_seconds, samplerate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds.
        samplerate: Sample rate in Hz (default: 16000).
    
    Returns:
        Audio as bytes (mono, int16).
    
    Raises:
        ASRModuleError: If duration/samplerate invalid or recording fails.
    """
    if duration <= 0:
        raise ASRModuleError(
            "[asr_module.record_audio] Invalid duration. "
            f"Expected positive number, got {duration}. "
            "Fix: pass duration > 0 (example: 5)."
        )
    
    logger.info(f"[ASR] Recording {duration}s at {samplerate}Hz...")
    
    try:
        recording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        audio_bytes = recording.astype('int16').tobytes()
        logger.debug(f"[ASR] Recorded {len(audio_bytes)} bytes")
        return audio_bytes
    
    except Exception as e:
        raise ASRModuleError(
            "[asr_module.record_audio] Recording failed. "
            f"Reason: {e}. "
            "Fix: check microphone connection and sounddevice installation."
        ) from e


def transcribe(
    audio_bytes: bytes,
    samplerate: int = TARGET_SAMPLE_RATE,
    language: str = _ASR.language_mode,
) -> Tuple[Optional[str], Optional[str]]:
    """Convert audio to text.
    
    Args:
        audio_bytes: Audio data as bytes (mono, 16kHz).
        samplerate: Sample rate of audio.
        language: "en", "ar", or "auto" (try ar first, then en).
    
    Returns:
        Tuple of (transcribed_text, detected_language).
        Returns (None, None) if transcription failed.
    
    Raises:
        ASRModuleError: If input invalid or API fails.
    """
    if audio_bytes is None or len(audio_bytes) == 0:
        raise ASRModuleError(
            "[asr_module.transcribe] No audio data provided. "
            "Fix: ensure record_audio() returned valid bytes."
        )

    if _ASR.provider != "google":
        raise ASRModuleError(
            f"[asr_module.transcribe] Unsupported ASR provider '{_ASR.provider}'. "
            "Fix: set ROBOT_ASR_PROVIDER=google or implement an offline ASR backend."
        )
    
    audio_data = sr.AudioData(audio_bytes, samplerate, sample_width=2)
    recognizer = sr.Recognizer()
    
    # Determine languages to try
    if language == "auto":
        langs_to_try = ["ar", "en"]  # Prefer Arabic first (Egypt context)
    elif language in SUPPORTED_LANGUAGES:
        langs_to_try = [language]
    else:
        raise ASRModuleError(
            f"[asr_module.transcribe] Unsupported language '{language}'. "
            f"Fix: use one of {list(SUPPORTED_LANGUAGES.keys())} or 'auto'."
        )
    
    for lang in langs_to_try:
        try:
            lang_code = SUPPORTED_LANGUAGES[lang]
            logger.info(f"[ASR] Attempting recognition with {lang} ({lang_code})...")
            text = recognizer.recognize_google(audio_data, language=lang_code)
            logger.info(f"[ASR] Success: recognized {lang}")
            return text, lang
        
        except sr.UnknownValueError:
            logger.debug(f"[ASR] No speech detected or unclear audio in {lang}")
            continue
        
        except sr.RequestError as e:
            raise ASRModuleError(
                "[asr_module.transcribe] Google Speech API request failed. "
                f"Reason: {e}. "
                "Fix: check internet connection and API quota. "
                "For offline operation, consider Whisper model."
            ) from e
        
        except Exception as e:
            raise ASRModuleError(
                "[asr_module.transcribe] Unexpected error. "
                f"Reason: {e}. "
                "Fix: verify audio format (16kHz, mono, 16-bit)."
            ) from e
    
    logger.warning("[ASR] Speech not detected in any language")
    return None, None


"""
This module intentionally contains no inline test runners.
Use `d:\\AI_Robot\\VoicePipline\\tests` for unit tests and integration tests.
"""