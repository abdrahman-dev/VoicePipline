# Mini AI Educational Robot – Voice Pipeline

## What it does
Real-time voice pipeline:
`VAD (speech detection) -> ASR (transcription) -> LLM (Ollama) -> TTS (Edge TTS)`

Interrupt behavior:
If the user starts speaking while TTS is playing, the pipeline immediately calls `tts.stop()` and begins listening for the next utterance.

## Folder layout
- `modules/`: `vad_module.py`, `asr_module.py`, `llm_module.py`, `tts_module.py`
- `config/settings.py`: centralized configuration (no provider hardcoding inside modules)
- `main.py`: orchestration layer
- `tests/`: unit tests for each module

## Setup
Install dependencies (example):

```powershell
pip install numpy torch requests sounddevice SpeechRecognition edge-tts pygame
```

Silero VAD:
- `vad_module` will try to load a local model if `ROBOT_VAD_MODEL_LOCAL_PATH` is set
- otherwise it uses `torch.hub` from `snakers4/silero-vad`

## Configure (optional)
Use environment variables to override defaults. Common ones:

- `ROBOT_OLLAMA_HOST`, `ROBOT_OLLAMA_MODEL`
- `ROBOT_LLM_REQUEST_TIMEOUT_SEC`
- `ROBOT_TTS_TEMP_DIR`
- `ROBOT_VAD_THRESHOLD`, `ROBOT_VAD_SILENCE_TIMEOUT_SEC`

## Run
```powershell
python main.py
```

Stop with `Ctrl+C`.

## Tests
```powershell
pytest -q
```

