from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from config.settings import get_settings
from modules import asr_module, llm_module, tts_module, vad_module


_SETTINGS = get_settings()


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, _SETTINGS.general.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def float32_chunk_to_int16_bytes(chunk: np.ndarray) -> bytes:
    # Convert [-1, 1] float PCM into signed int16 PCM bytes.
    chunk_clipped = np.clip(chunk, -1.0, 1.0)
    chunk_i16 = (chunk_clipped * 32767.0).astype(np.int16, copy=False)
    return chunk_i16.tobytes()


@dataclass
class Segment:
    turn_id: int
    audio_chunks: List[bytes]  # list of int16-PCM chunks to avoid heavy work in callback


class VoicePipeline:
    def __init__(self) -> None:
        setup_logging()
        self._logger = logging.getLogger("voice_pipeline")

        # Keep sample rates aligned across VAD capture and ASR decoding.
        if _SETTINGS.vad.sample_rate != _SETTINGS.asr.sample_rate:
            raise RuntimeError(
                "VAD sample_rate and ASR sample_rate must match. "
                f"Got vad={_SETTINGS.vad.sample_rate}, asr={_SETTINGS.asr.sample_rate}."
            )

        self._sample_rate = _SETTINGS.vad.sample_rate
        self._chunk_size = int(self._sample_rate * _SETTINGS.vad.chunk_duration_ms / 1000)
        self._chunk_seconds = self._chunk_size / float(self._sample_rate)

        if self._chunk_size <= 0:
            raise RuntimeError("Invalid VAD chunk size computed.")

        self._pre_chunks = max(1, int(_SETTINGS.vad.pre_speech_buffer_seconds / self._chunk_seconds))
        self._min_speech_chunks = max(1, int(_SETTINGS.vad.min_speech_seconds / self._chunk_seconds))
        self._silence_timeout_chunks = max(1, int(_SETTINGS.vad.silence_timeout_seconds / self._chunk_seconds))

        self._logger.info(
            "Config: sample_rate=%s chunk_size=%s pre_chunks=%s min_speech_chunks=%s silence_timeout_chunks=%s",
            self._sample_rate,
            self._chunk_size,
            self._pre_chunks,
            self._min_speech_chunks,
            self._silence_timeout_chunks,
        )

        vad_module.set_threshold(_SETTINGS.vad.initial_threshold)

        self._tts = tts_module.TTSModule(_SETTINGS.tts)
        self._llm = llm_module.LLMModule(_SETTINGS.llm)

        # Create a conversation session once.
        self._session_id = self._llm.session_manager.create_session(
            student_name=_SETTINGS.general.student_name,
            language=_SETTINGS.general.default_session_language,
        )

        self._latest_turn_id = 0
        self._turn_lock = threading.Lock()

        self._active_segment_lock = threading.Lock()
        self._active_segment_turn_id: Optional[int] = None

    def _next_turn_id(self) -> int:
        with self._turn_lock:
            self._latest_turn_id += 1
            return self._latest_turn_id

    def _get_latest_turn_id(self) -> int:
        with self._turn_lock:
            return self._latest_turn_id

    def _maybe_stop_tts_on_interrupt(self) -> None:
        if self._tts.is_playing():
            self._logger.info("[interrupt] Speech started while TTS playing; stopping playback.")
            self._tts.stop()

    def _process_segment(self, segment: Segment) -> None:
        # Drop stale segments if a newer utterance started.
        if segment.turn_id != self._get_latest_turn_id():
            return

        try:
            self._logger.info("[ASR] Transcribing speech segment...")
            audio_bytes = b"".join(segment.audio_chunks)
            text, detected_lang = asr_module.transcribe(
                audio_bytes,
                samplerate=self._sample_rate,
                language=_SETTINGS.asr.language_mode,
            )

            self._logger.info("[ASR] result=%r lang=%r", text, detected_lang)
            if not text:
                return

            if segment.turn_id != self._get_latest_turn_id():
                return

            t0 = time.monotonic()
            response = self._llm.chat(self._session_id, text)
            t1 = time.monotonic()
            self._logger.info("[LLM] response_time=%.2fs", t1 - t0)

            if segment.turn_id != self._get_latest_turn_id():
                return

            # Start TTS playback (interruptible).
            self._logger.info("[TTS] Playback start (lang=%s)", detected_lang)
            self._tts.speak(response, language=detected_lang)
        except (asr_module.ASRModuleError, llm_module.LLMModuleError, tts_module.TTSModuleError) as exc:
            self._logger.error("Segment processing failed: %s", exc, exc_info=True)
        except Exception as exc:
            self._logger.error("Unexpected segment processing error: %s", exc, exc_info=True)

    def run_forever(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "Failed to import sounddevice. "
                "Fix: install it with `pip install sounddevice` and ensure PortAudio is available."
            ) from exc

        # VAD segment state.
        pre_buffer: Deque[bytes] = deque(maxlen=self._pre_chunks)
        segment_chunks: Optional[Deque[bytes]] = None
        silence_chunks = 0
        speech_chunks = 0
        current_turn_id: Optional[int] = None

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            nonlocal segment_chunks, silence_chunks, speech_chunks, current_turn_id, pre_buffer

            if status:
                self._logger.debug("Audio stream status: %s", status)

            if indata is None or len(indata) == 0:
                return

            # `indata` comes as shape (frames, channels); enforce 1D mono.
            chunk = np.asarray(indata[:, 0], dtype=np.float32)
            speech_now = vad_module.is_speech(chunk)
            chunk_bytes = float32_chunk_to_int16_bytes(chunk)

            if segment_chunks is None:
                # Not currently recording a speech segment.
                if speech_now:
                    current_turn_id = self._next_turn_id()
                    self._active_segment_turn_id = current_turn_id

                    # CRITICAL: interrupt any TTS currently playing.
                    self._maybe_stop_tts_on_interrupt()

                    self._logger.info("[VAD] Speech detected; starting segment (turn=%s).", current_turn_id)
                    segment_chunks = deque(pre_buffer)
                    segment_chunks.append(chunk_bytes)
                    speech_chunks = 1
                    silence_chunks = 0
                    pre_buffer.clear()
                else:
                    pre_buffer.append(chunk_bytes)
                return

            # Currently recording a segment.
            assert current_turn_id is not None
            segment_chunks.append(chunk_bytes)
            speech_chunks += 1

            if speech_now:
                silence_chunks = 0
            else:
                silence_chunks += 1

            if silence_chunks >= self._silence_timeout_chunks and speech_chunks >= self._min_speech_chunks:
                # Finalize segment.
                audio_chunks = list(segment_chunks)
                finished_turn_id = current_turn_id

                segment_chunks = None
                silence_chunks = 0
                speech_chunks = 0
                current_turn_id = None
                self._active_segment_turn_id = None

                self._logger.info(
                    "[VAD] Segment ended; enqueuing ASR/LLM/TTS processing (turn=%s).",
                    finished_turn_id,
                )

                threading.Thread(
                    target=self._process_segment,
                    args=(Segment(turn_id=finished_turn_id, audio_chunks=audio_chunks),),
                    daemon=True,
                ).start()

        self._logger.info("Listening... Press Ctrl+C to stop.")
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_size,
            callback=callback,
        ):
            while True:
                time.sleep(0.5)


def main() -> None:
    pipeline = VoicePipeline()
    pipeline.run_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Keep this silent; logs already show it was running.
        pass

