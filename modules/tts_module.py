"""
TTS Module

Provides a thread-safe, interruptible text-to-speech pipeline using `edge_tts` + `pygame`.

Key properties:
- No import-time pygame initialization (safer for testing/headless environments)
- `stop()` halts currently playing audio immediately
- Concurrent `speak()` calls are serialized via a request id; interrupted requests become no-ops
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

from config.settings import get_settings

_SETTINGS = get_settings()
_TTS = _SETTINGS.tts

logger = logging.getLogger(__name__)


class TTSModuleError(RuntimeError):
    pass


def detect_language(text: str) -> str:
    # Simple language heuristic: Arabic Unicode block check.
    for char in text:
        if "\u0600" <= char <= "\u06FF":
            return "ar"
    return "en"


class TTSModule:
    def __init__(self, settings=_TTS):
        self._settings = settings
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._request_id = 0
        self._play_thread: Optional[threading.Thread] = None

        self._pygame = None
        self._edge_tts = None
        self._pygame_ready = False

        if getattr(self._settings, "engine", "edge_tts") != "edge_tts":
            raise TTSModuleError(
                f"[tts_module] Unsupported TTS engine '{self._settings.engine}'. "
                "Fix: set ROBOT_TTS_ENGINE=edge_tts or implement a new engine backend."
            )

        self._init_dependencies()

    def _init_dependencies(self) -> None:
        try:
            import pygame  # type: ignore

            self._pygame = pygame
        except Exception as exc:
            raise TTSModuleError(
                "[tts_module] pygame is required for playback. "
                "Fix: install pygame and ensure an audio device is available."
            ) from exc

        try:
            import edge_tts  # type: ignore

            self._edge_tts = edge_tts
        except Exception as exc:
            raise TTSModuleError(
                "[tts_module] edge_tts is required for speech synthesis. "
                "Fix: install edge-tts."
            ) from exc

        try:
            # Initialize mixer once.
            self._pygame.mixer.init()
            self._pygame_ready = True
        except Exception as exc:
            raise TTSModuleError(
                "[tts_module] Failed to initialize pygame mixer. "
                "Fix: ensure audio backend exists on Windows/Raspberry Pi."
            ) from exc

    def is_playing(self) -> bool:
        if not self._pygame_ready:
            return False
        try:
            return bool(self._pygame.mixer.music.get_busy())
        except Exception:
            return False

    def stop(self) -> None:
        """Immediately stop currently playing audio and invalidate pending requests."""
        with self._lock:
            self._stop_event.set()
            self._request_id += 1  # invalidate current/queued requests
            current_thread = self._play_thread

        # Stop playback outside the lock.
        try:
            if self._pygame_ready:
                self._pygame.mixer.music.stop()
                self._pygame.mixer.music.unload()
        except Exception:
            # Stop should be best-effort; nothing else depends on it succeeding.
            pass

        # We intentionally don't join here; `stop()` must be quick.
        _ = current_thread

    def _get_voice(self, language: str) -> str:
        voice_map = self._settings.voice_map or {}
        return voice_map.get(language, voice_map.get("en", "en-US-GuyNeural"))

    def _temp_path_for_request(self, request_id: int) -> str:
        os.makedirs(self._settings.audio_temp_dir, exist_ok=True)
        filename = self._settings.audio_filename_template.format(turn_id=request_id)
        return os.path.join(self._settings.audio_temp_dir, filename)

    async def _generate_audio_async(self, text: str, voice: str, path: str) -> None:
        assert self._edge_tts is not None
        communicate = self._edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(path)

    def _play_audio(self, path: str, request_id: int) -> None:
        assert self._pygame is not None
        assert self._pygame_ready is True

        try:
            # If we've been interrupted, do not play.
            if self._stop_event.is_set():
                return

            # Load/play.
            self._pygame.mixer.music.load(path)
            self._pygame.mixer.music.play()

            # Poll until finished or interrupted.
            while self._pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    return
                time.sleep(self._settings.pygame_poll_interval_seconds)

        except Exception as exc:
            # Playback errors should not crash the pipeline.
            logger.error("Playback failed: %s", exc, exc_info=True)
        finally:
            # Unload to release file lock before deletion.
            try:
                self._pygame.mixer.music.unload()
            except Exception:
                pass

            # Delete file best-effort.
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

            # Clear stop_event if this request is the latest.
            with self._lock:
                if request_id == self._request_id and self._stop_event.is_set():
                    # Keep stop_event set only until next speak() resets it.
                    pass

    def speak(self, text: str, language: Optional[str] = None) -> None:
        """
        Start (generate + play) speech in a background thread.

        The function returns quickly; playback can be interrupted via `stop()`.
        """
        if not text or text.strip() == "":
            raise TTSModuleError("[tts_module.speak] Empty text")

        if language is None:
            language = detect_language(text)

        voice = self._get_voice(language)

        with self._lock:
            self._stop_event.clear()
            self._request_id += 1
            request_id = self._request_id

        path = self._temp_path_for_request(request_id)

        def worker() -> None:
            # Generate audio (blocking) then play.
            try:
                if self._stop_event.is_set():
                    return

                asyncio.run(self._generate_audio_async(text=text, voice=voice, path=path))

                # If a newer request arrived, do not play/delete races.
                with self._lock:
                    if request_id != self._request_id or self._stop_event.is_set():
                        try:
                            if os.path.exists(path):
                                os.remove(path)
                        except Exception:
                            pass
                        return

                self._play_audio(path=path, request_id=request_id)
            except Exception as exc:
                # Generation/playback errors should not crash the whole pipeline.
                logger.error("speak() worker failed: %s", exc, exc_info=True)
                return

        self._play_thread = threading.Thread(target=worker, daemon=True)
        self._play_thread.start()