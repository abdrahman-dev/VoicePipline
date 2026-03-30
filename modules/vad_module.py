"""Minimal Silero VAD wrapper optimized for low-latency CPU inference.

This module is designed for embedded environments such as Raspberry Pi 4.
It keeps runtime overhead low by:
1) loading the model only once (singleton style),
2) forcing single-thread CPU execution,
3) minimizing work inside the hot path (`is_speech`).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch

from config.settings import get_settings

_SETTINGS = get_settings()
_VAD = _SETTINGS.vad


TARGET_SAMPLE_RATE = _VAD.sample_rate
_MAX_ABS_AMPLITUDE = _VAD.max_abs_amplitude

# Single-thread CPU execution is usually best for low-latency small-chunk inference.
torch.set_num_threads(_VAD.torch_threads)

_MODEL = None
_MODEL_LOCK = threading.Lock()
_THRESHOLD = _VAD.initial_threshold


class VADModuleError(RuntimeError):
    """Raised when input preparation or Silero inference fails."""


def set_threshold(threshold: float) -> None:
    """Set the speech probability threshold used by `is_speech`.

    A lower threshold catches weaker speech but increases false positives.
    A higher threshold is more conservative in noisy rooms.
    """
    if not (0.0 <= threshold <= 1.0):
        raise VADModuleError(
            "[vad_module.set_threshold] Invalid threshold. "
            f"Expected range [0.0, 1.0], got {threshold}. "
            "Fix: pass a float between 0.0 and 1.0 (example: 0.6)."
        )
    global _THRESHOLD
    _THRESHOLD = float(threshold)


def get_threshold() -> float:
    """Return current speech probability threshold."""
    return _THRESHOLD


def _load_model_once():
    """Load Silero VAD once and cache it for all subsequent calls."""
    global _MODEL, _UTILS

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        try:
            local_path = _VAD.model_local_path.strip()

            if local_path:
                try:
                    model = torch.jit.load(local_path, map_location="cpu")
                except Exception:
                    model = torch.load(local_path, map_location="cpu")

                utils = None  # local model may not provide utils

            else:
                # 🔥 FIX هنا
                model, utils = torch.hub.load(
                    _VAD.model_hub_repo,
                    _VAD.model_hub_name,
                    trust_repo=_VAD.model_trust_repo,
                )

            model.eval()

            _MODEL = model
            _UTILS = utils  # optional

            return _MODEL

        except Exception as exc:
            raise VADModuleError(
                "[vad_module._load_model_once] Failed to load Silero VAD model. "
                f"Reason: {exc}. "
                "Fix: set ROBOT_VAD_MODEL_LOCAL_PATH if you have a local model, "
                "or ensure internet access for torch.hub first download."
            ) from exc


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono float32 with clipping to [-1, 1]."""
    if audio is None:
        raise VADModuleError(
            "[vad_module.to_mono_float32] audio is None. "
            "Fix: pass a valid numpy array from your audio capture pipeline."
        )

    arr = np.asarray(audio)
    if arr.size == 0:
        raise VADModuleError(
            "[vad_module.to_mono_float32] audio chunk is empty. "
            "Fix: check stream blocksize and callback data flow."
        )

    # If shape is (N, C), average channels to mono.
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim != 1:
        raise VADModuleError(
            "[vad_module.to_mono_float32] Invalid audio shape. "
            f"Expected 1D mono or 2D (samples, channels), got {arr.shape}. "
            "Fix: provide mono array or standard interleaved channel array."
        )

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    # Clip unexpected ranges to keep inference stable.
    np.clip(arr, -_MAX_ABS_AMPLITUDE, _MAX_ABS_AMPLITUDE, out=arr)
    return arr


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Lightweight linear resampling for optional pre-processing.

    This helper is intentionally simple and dependency-light. For best quality,
    use high-quality offline resamplers when available.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise VADModuleError(
            "[vad_module.resample_linear] Invalid sample rate. "
            f"orig_sr={orig_sr}, target_sr={target_sr}. "
            "Fix: use positive integer sample rates."
        )

    if orig_sr == target_sr:
        return audio

    if audio.size == 0:
        return audio

    duration = audio.shape[0] / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))

    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, duration, num=target_len, endpoint=False, dtype=np.float64)
    return np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)


def prepare_audio_chunk(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Optional helper: normalize input format to 16kHz mono float32."""
    mono = to_mono_float32(audio)
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = resample_linear(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return mono


def is_speech(audio_chunk: np.ndarray) -> bool:
    """Return True if speech is detected for a 16kHz mono float32 chunk.

    Input expectation (fast path):
    - sample rate: 16kHz
    - shape: (N,)
    - dtype: float32
    - chunk size around 480 samples (~30 ms) for real-time streaming
    """
    try:
        if audio_chunk is None:
            raise VADModuleError(
                "[vad_module.is_speech] audio_chunk is None. "
                "Fix: pass a valid 1D numpy array with 16kHz audio."
            )

        # Fast path checks: avoid unnecessary allocations if chunk already matches expectations.
        if not isinstance(audio_chunk, np.ndarray):
            chunk = np.asarray(audio_chunk, dtype=np.float32)
        else:
            chunk = audio_chunk
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32, copy=False)

        if chunk.ndim != 1:
            raise VADModuleError(
                "[vad_module.is_speech] Invalid chunk dimensionality. "
                f"Expected 1D mono array, got shape {chunk.shape}. "
                "Fix: convert multi-channel input using prepare_audio_chunk()."
            )

        if chunk.size == 0:
            return False

        if not chunk.flags["C_CONTIGUOUS"]:
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)

        model = _load_model_once()

        # from_numpy shares memory when possible, keeping overhead minimal.
        chunk_tensor = torch.from_numpy(chunk)
        speech_prob_tensor = model(chunk_tensor.unsqueeze(0), TARGET_SAMPLE_RATE)
        # Some Silero wrappers return shape [1] or scalar tensors.
        speech_prob = float(speech_prob_tensor.detach().cpu().numpy().flatten()[0])
        return speech_prob >= _THRESHOLD

    except VADModuleError:
        raise
    except Exception as exc:
        raise VADModuleError(
            "[vad_module.is_speech] Silero inference failed. "
            f"Reason: {exc}. "
            "Fix: verify 16kHz mono float32 chunks, and that model loading succeeded."
        ) from exc


def threshold_tuning_guide() -> str:
    """Return practical threshold tuning advice for different environments."""
    return (
        "Threshold tuning guide (Silero speech probability):\n"
        "- Quiet room: 0.45-0.55 to catch softer voice.\n"
        "- Normal home/office: 0.55-0.65 (recommended start: 0.60).\n"
        "- Noisy environment: 0.65-0.80 to reduce false positives.\n"
        "If you get too many false triggers, increase threshold by 0.05 steps.\n"
        "If speech is missed, decrease threshold by 0.05 steps."
    )
