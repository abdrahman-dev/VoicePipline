# AI Educational Robot - Voice Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/yourusername/VoicePipline)

A real-time, modular voice interaction pipeline for educational robots. Supports Arabic and English with intelligent interruption handling for natural conversation flow.

---

## Architecture

```
Microphone → VAD (Silero) → ASR (Google) → LLM (OpenRouter) → TTS (Edge) → Speaker
```

### Pipeline Flow
1. **VAD**: Voice Activity Detection using Silero VAD (CPU-optimized)
2. **ASR**: Speech-to-Text using Google Speech Recognition
3. **LLM**: Language understanding via OpenRouter API
4. **TTS**: Text-to-Speech using Edge TTS (Microsoft Azure)

---

## Quick Start

### Prerequisites
- Python 3.8+
- OpenRouter API key

### Installation

```powershell
# Navigate to project
cd D:/AI_Robot/VoicePipline

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install numpy torch sounddevice SpeechRecognition edge-tts pygame requests pytest
```

### Configuration

Set environment variables or create `.env` file:

```powershell
$env:ROBOT_OPENROUTER_API_KEY = "your-api-key"
$env:ROBOT_OPENROUTER_MODEL = "openrouter/free"
$env:ROBOT_LOG_LEVEL = "INFO"
```

### Run

```powershell
python main.py
```

---

## Project Structure

```
VoicePipline/
├── main.py                 # Main orchestration (VoicePipeline class)
├── conftest.py             # Pytest configuration
├── pytest.ini              # Pytest settings
├── README.md               # This file
│
├── config/
│   ├── settings.py         # Centralized configuration (dataclasses)
│   ├── llm_client.py       # OpenRouter client utilities
│   ├── ropo_test.py        # Testing utilities
│   └── __init__.py
│
├── modules/
│   ├── vad_module.py       # Voice Activity Detection (Silero VAD)
│   ├── asr_module.py       # Speech-to-Text (Google)
│   ├── llm_module.py       # LLM with session management (OpenRouter)
│   ├── tts_module.py       # Text-to-Speech (Edge TTS)
│   └── __init__.py
│
└── test_modules/
    ├── test_vad.py         # VAD unit tests
    ├── test_asr.py         # ASR unit tests
    ├── test_llm.py         # LLM unit tests
    ├── test_tts.py         # TTS unit tests
    └── __init__.py
```

---

## Configuration

All settings via environment variables in `config/settings.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOT_LOG_LEVEL` | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `ROBOT_VAD_THRESHOLD` | 0.60 | Speech detection sensitivity (0.0-1.0) |
| `ROBOT_VAD_SAMPLE_RATE` | 16000 | Audio sample rate in Hz |
| `ROBOT_LLM_PROVIDER` | openrouter | LLM provider |
| `ROBOT_OPENROUTER_API_KEY` | - | OpenRouter API key |
| `ROBOT_OPENROUTER_MODEL` | openrouter/free | Model name |
| `ROBOT_LLM_REQUEST_TIMEOUT_SEC` | 90 | Request timeout |
| `ROBOT_LLM_WINDOW_SIZE` | 10 | Conversation memory window |
| `ROBOT_TTS_ENGINE` | edge_tts | TTS engine |
| `ROBOT_TTS_POLL_SEC` | 0.05 | Pygame polling interval |

---

## Key Features

- **Real-time Processing**: Low-latency pipeline optimized for responsive interaction
- **Intelligent Interruption**: Natural turn-taking with instant interrupt handling
- **Multi-language Support**: Arabic and English with auto-detection
- **SQLite Persistence**: Session and message storage
- **Modular Architecture**: Swappable components (VAD, ASR, LLM, TTS)
- **Environment Configuration**: No hardcoded settings
- **Embedded-Ready**: CPU-optimized for Raspberry Pi

---

## Testing

```powershell
# Run all tests
pytest -v

# Run specific module tests
pytest test_modules/test_vad.py -v
pytest test_modules/test_llm.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=modules test_modules/
```

---

## Dependencies

- **numpy**: Audio data processing
- **torch**: Silero VAD model
- **sounddevice**: Audio input capture
- **SpeechRecognition**: Google ASR
- **edge-tts**: Microsoft Edge TTS
- **pygame**: Audio playback
- **requests**: HTTP client for OpenRouter
- **pytest**: Testing framework

---

## License

MIT License - See LICENSE file

### Third-party Components
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection
- [Google Speech Recognition](https://pypi.org/project/SpeechRecognition/) - ASR
- [OpenRouter](https://openrouter.ai/) - LLM API
- [Edge TTS](https://pypi.org/project/edge-tts/) - Text-to-Speech
