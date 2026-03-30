"""Simple real-time VAD test script using sounddevice InputStream.

Records ~5 seconds at 16kHz, chunks audio in ~30ms blocks, and prints:
- speech chunk count
- silence chunk count
- speech percentage
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    raise RuntimeError(
        "[test_vad] Failed to import sounddevice. "
        f"Reason: {exc}. "
        "Fix: install it with `pip install sounddevice` and ensure PortAudio is available."
    ) from exc

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.vad_module import VADModuleError, is_speech, set_threshold, threshold_tuning_guide


SAMPLE_RATE = 16000
CHUNK_SIZE = 480  # ~30 ms at 16kHz
DURATION_SEC = 5.0
DEFAULT_THRESHOLD = 0.60


@dataclass
class VADStats:
    speech_chunks: int = 0
    silence_chunks: int = 0
    callback_errors: int = 0

    @property
    def total(self) -> int:
        return self.speech_chunks + self.silence_chunks

    @property
    def speech_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.speech_chunks / self.total


def run_stream_test(duration_sec: float = DURATION_SEC) -> VADStats:
    """Run VAD on live microphone stream and return collected stats."""
    stats = VADStats()
    callback_error_messages: list[str] = []

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        del frames, time_info
        if status:
            # Non-fatal stream status indicators (over/underflow etc.)
            print(f"[test_vad.callback] Stream status: {status}", file=sys.stderr)

        try:
            # indata shape is (frames, channels). Convert to 1D mono view/copy once.
            chunk = indata[:, 0]
            if is_speech(chunk):
                stats.speech_chunks += 1
            else:
                stats.silence_chunks += 1
        except VADModuleError as exc:
            stats.callback_errors += 1
            callback_error_messages.append(str(exc))
        except Exception as exc:
            stats.callback_errors += 1
            callback_error_messages.append(
                "[test_vad.callback] Unexpected callback failure. "
                f"Reason: {exc}. Fix: verify audio device, dtype float32, and chunk size."
            )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            print(f"[test_vad] Listening for {duration_sec:.1f}s... speak now.")
            time.sleep(duration_sec)
    except Exception as exc:
        raise RuntimeError(
            "[test_vad.run_stream_test] Audio stream failed to start/run. "
            f"Reason: {exc}. "
            "Fix: check microphone permissions/device, and ensure sample rate 16000 is supported."
        ) from exc

    if callback_error_messages:
        print("[test_vad] Callback errors detected:", file=sys.stderr)
        for msg in callback_error_messages[:3]:
            print(f"  - {msg}", file=sys.stderr)
        if len(callback_error_messages) > 3:
            print(f"  - ... {len(callback_error_messages) - 3} more", file=sys.stderr)

    return stats


def main() -> None:
    """Entrypoint for the VAD streaming test script."""
    try:
        set_threshold(DEFAULT_THRESHOLD)
        print(f"[test_vad] Using threshold: {DEFAULT_THRESHOLD:.2f}")
        print("[test_vad] " + threshold_tuning_guide().replace("\n", " | "))

        stats = run_stream_test(duration_sec=DURATION_SEC)
        print(f"[test_vad] Speech chunks  : {stats.speech_chunks}")
        print(f"[test_vad] Silence chunks : {stats.silence_chunks}")
        print(f"[test_vad] Speech percent : {stats.speech_pct:.2f}%")
        if stats.callback_errors:
            print(f"[test_vad] Callback errors: {stats.callback_errors}")
    except Exception as exc:
        print(
            "[test_vad.main] Test run failed. "
            f"Reason: {exc}. "
            "Fix: verify dependencies and audio device configuration.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()

