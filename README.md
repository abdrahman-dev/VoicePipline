# 🤖 AI Educational Robot – Voice Pipeline

A real-time, modular voice interaction pipeline for intelligent robots that listen, understand, and respond to students in real-time. Built with a focus on low-latency embedded performance, multi-language support, and flexible configuration.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Project Structure](#project-structure)
8. [Architecture Deep Dive](#architecture-deep-dive)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The **Voice Pipeline** is a multi-stage speech interaction system designed for educational robots. It orchestrates four core components working in harmony:

```
🎤 Microphone Input
    ↓
[VAD] Speech Detection (Silero)
    ↓
[ASR] Speech-to-Text (Google Speech Recognition)
    ↓
[LLM] Language Understanding (Ollama + LLaMA 3.2)
    ↓
[TTS] Text-to-Speech (Edge TTS)
    ↓
🔊 Speaker Output
```

**Real-time interaction**: If the user starts speaking while the robot is still responding (TTS playing), the pipeline intelligently interrupts the current response and begins processing the new input immediately.

---

## How It Works

### 1. **Voice Activity Detection (VAD) – Listening Phase**

Using the lightweight **Silero VAD** model optimized for embedded systems:

- Continuously monitors incoming audio in real-time
- Detects speech presence with configurable sensitivity
- Maintains a pre-speech buffer to capture the beginning of utterances
- Waits for a silent timeout period to confirm speech has ended
- **Single-threaded CPU execution** for minimal latency (ideal for Raspberry Pi)

**Flow**:

```
Audio chunks arrive → Is there speech? →
  Yes → Start recording → Accumulate chunks →
    Silence for N seconds? → Yes → Send to ASR
```

### 2. **Automatic Speech Recognition (ASR) – Transcription Phase**

Using **Google Speech Recognition** with multi-language support:

- Converts audio bytes to text
- Auto-detects language (English, Arabic, or specified language)
- Returns both transcribed text and detected language
- Sample rate synchronized with VAD (must match: 16kHz default)

**Supported Languages**:

- English (en-US)
- Arabic (ar-EG)
- Auto-detection enabled by default

### 3. **Large Language Model (LLM) – Understanding Phase**

Using **Ollama** backend with **LLaMA 3.2** or compatible models:

- Maintains conversation sessions with memory
- Implements sliding-window history (last 10 messages by default)
- Auto-summarization every 10 messages to keep context relevant
- Stores all conversations in SQLite for future reference
- System prompts configured for educational interaction

**Session Management**:

- Each student gets a unique session ID
- Messages are tracked with timestamps
- Full conversation history persisted to database

### 4. **Text-to-Speech (TTS) – Response Phase**

Using **Edge TTS** (Microsoft Azure) with pygame audio playback:

- Converts LLM response to MP3 audio
- Supports Arabic and English voices
- Thread-safe, with interrupt capability
- Audio files cached in temp directory

**Interrupt Behavior** ⚡:
When speech is detected while TTS is playing:

1. Stop current playback immediately
2. Discard the in-progress audio response
3. Reset to listening mode
4. Process the new user input

---

## Key Features

✅ **Real-time Processing**: Low-latency pipeline optimized for responsive interaction  
✅ **Intelligent Interruption**: Natural turn-taking behavior with interrupt handling  
✅ **Modular Architecture**: Each component is independent and swappable  
✅ **Multi-language Support**: Arabic and English with auto-detection  
✅ **Persistent Memory**: SQLite-backed session and message storage  
✅ **Configurable Settings**: Environment variable-based configuration  
✅ **Educational AI**: System prompts designed for student learning  
✅ **Embedded-Ready**: Optimized for Raspberry Pi and low-resource environments  
✅ **Zero Hardcoding**: All provider/model selection via configuration  
✅ **Comprehensive Logging**: Full pipeline visibility for debugging

---

## Installation

### Prerequisites

- **Python 3.8+**
- **PortAudio** (for sounddevice; see platform-specific instructions below)
- **Ollama** running locally on `http://localhost:11434` (or configure `ROBOT_OLLAMA_HOST`)

### Step 1: Clone and Navigate

```bash
cd d:\AI_Robot\VoicePipline
```

### Step 2: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install --upgrade pip

pip install \
  numpy \
  torch \
  requests \
  sounddevice \
  SpeechRecognition \
  edge-tts \
  pygame \
  pytest
```

### Step 4: Install Ollama

Download and install from [ollama.ai](https://ollama.ai), then pull a model:

```bash
ollama pull glm-4.6:cloud    # or llama2, mistral, neural-chat, etc.
ollama serve                  # Start Ollama server (runs on port 11434)
```

### Step 5: Verify Installation

```powershell
# Test all imports work
python -c "import torch, sounddevice, speech_recognition, edge_tts, pygame; print('All dependencies OK')"

# Verify Ollama is running
python -c "import requests; print(requests.get('http://localhost:11434/api/tags').status_code)"
```

---

## Configuration

The pipeline uses a **centralized, environment-variable-based configuration system**. No hardcoding of settings inside modules.

### Configuration File Structure

Settings are loaded from `config/settings.py` using `dataclass` definitions:

- **GeneralSettings**: Student name, language, logging level
- **VADSettings**: Speech detection thresholds, silence timeout, buffer sizes
- **ASRSettings**: Language mode, sample rate, provider selection
- **LLMSettings**: Ollama connection, model selection, timeouts, system prompts
- **TTSSettings**: Voice selection, audio temp directory, polling interval

### Environment Variables Reference

#### **General Settings**

| Variable                         | Default   | Description                                                        |
| -------------------------------- | --------- | ------------------------------------------------------------------ |
| `ROBOT_LOG_LEVEL`                | `INFO`    | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`             |
| `ROBOT_STUDENT_NAME`             | `Student` | Name of the student (used in LLM context)                          |
| `ROBOT_DEFAULT_SESSION_LANGUAGE` | `ar`      | Default language for new sessions: `ar` (Arabic) or `en` (English) |

#### **VAD (Voice Activity Detection)**

| Variable                        | Default               | Description                                                    |
| ------------------------------- | --------------------- | -------------------------------------------------------------- |
| `ROBOT_VAD_SAMPLE_RATE`         | `16000`               | Audio sample rate in Hz (must match ASR)                       |
| `ROBOT_VAD_THRESHOLD`           | `0.60`                | Speech detection sensitivity (0.0–1.0; lower = more sensitive) |
| `ROBOT_VAD_CHUNK_MS`            | `32`                  | Audio chunk duration in milliseconds (30–40 typical)           |
| `ROBOT_VAD_PRE_ROLL_SEC`        | `0.30`                | Pre-speech buffer to capture utterance start                   |
| `ROBOT_VAD_MIN_SPEECH_SEC`      | `0.50`                | Minimum speech duration to be considered valid                 |
| `ROBOT_VAD_SILENCE_TIMEOUT_SEC` | `0.80`                | Silence duration to trigger end-of-speech                      |
| `ROBOT_VAD_TORCH_THREADS`       | `1`                   | PyTorch CPU threads (1 = single-threaded for low latency)      |
| `ROBOT_VAD_MODEL_LOCAL_PATH`    | ``                    | (Optional) Local path to Silero VAD model file                 |
| `ROBOT_VAD_HUB_REPO`            | `snakers4/silero-vad` | PyTorch Hub repository                                         |
| `ROBOT_VAD_HUB_NAME`            | `silero_vad`          | Model name on PyTorch Hub                                      |
| `ROBOT_VAD_TRUST_REPO`          | `true`                | Trust the repository (set to `false` to review code first)     |

#### **ASR (Automatic Speech Recognition)**

| Variable                         | Default  | Description                                                   |
| -------------------------------- | -------- | ------------------------------------------------------------- |
| `ROBOT_ASR_PROVIDER`             | `google` | ASR provider: `google` (currently only option)                |
| `ROBOT_ASR_SAMPLE_RATE`          | `16000`  | Audio sample rate (must match VAD)                            |
| `ROBOT_ASR_LANGUAGE_MODE`        | `auto`   | Language mode: `auto` (detect), `en` (English), `ar` (Arabic) |
| `ROBOT_ASR_DEFAULT_DURATION_SEC` | `5.0`    | Default recording duration for standalone ASR                 |

#### **LLM (Large Language Model)**

| Variable                          | Default                  | Description                                           |
| --------------------------------- | ------------------------ | ----------------------------------------------------- |
| `ROBOT_LLM_PROVIDER`              | `ollama`                 | LLM provider: `ollama` (currently only option)        |
| `ROBOT_OLLAMA_HOST`               | `http://localhost:11434` | Ollama server URL                                     |
| `ROBOT_OLLAMA_MODEL`              | `glm-4.6:cloud`          | Model name on Ollama (`ollama list` to see available) |
| `ROBOT_OLLAMA_AVAIL_TIMEOUT_SEC`  | `5`                      | Timeout to check if Ollama is available               |
| `ROBOT_LLM_REQUEST_TIMEOUT_SEC`   | `90`                     | Timeout for chat requests (seconds)                   |
| `ROBOT_LLM_SUMMARIZE_TIMEOUT_SEC` | `15`                     | Timeout for message summarization                     |
| `ROBOT_LLM_DB_PATH`               | `robot_sessions.db`      | Path to SQLite session database                       |
| `ROBOT_LLM_WINDOW_SIZE`           | `10`                     | Sliding window size for conversation memory           |

#### **TTS (Text-to-Speech)**

| Variable                   | Default             | Description                                     |
| -------------------------- | ------------------- | ----------------------------------------------- |
| `ROBOT_TTS_ENGINE`         | `edge_tts`          | TTS engine: `edge_tts` (Microsoft Azure)        |
| `ROBOT_TTS_TEMP_DIR`       | `[system temp]`     | Directory for temporary MP3 files               |
| `ROBOT_TTS_AUDIO_TEMPLATE` | `tts_{turn_id}.mp3` | Filename template for audio                     |
| `ROBOT_TTS_POLL_SEC`       | `0.05`              | Pygame polling interval (affects response time) |

### Example Configuration

Create a `.env` file or set environment variables in PowerShell:

```powershell
# .env file or environment variables
$env:ROBOT_STUDENT_NAME = "Ahmed"
$env:ROBOT_DEFAULT_SESSION_LANGUAGE = "ar"
$env:ROBOT_VAD_THRESHOLD = "0.55"
$env:ROBOT_OLLAMA_MODEL = "llama2"
$env:ROBOT_LLM_REQUEST_TIMEOUT_SEC = "120"
$env:ROBOT_LOG_LEVEL = "DEBUG"
```

Load in PowerShell:

```powershell
# Load .env file (basic example)
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)') {
            $name, $value = $matches[1].Trim(), $matches[2].Trim()
            Set-Item "env:$name" $value
        }
    }
}

python main.py
```

---

## Running the Pipeline

### Quick Start

```powershell
# Start Ollama server in another terminal
ollama serve

# In your main terminal, ensure virtualenv is active
.\venv\Scripts\Activate.ps1

# Run the pipeline
python main.py
```

After startup, you'll see:

```
2026-03-30 10:15:22 - voice_pipeline - INFO - Listening... Press Ctrl+C to stop.
```

### Interaction Example

1. **User speaks**: "What is photosynthesis?"
2. **VAD detects** speech, accumulates chunks
3. **VAD finishes** (silence timeout), sends to ASR
4. **ASR transcribes**: "What is photosynthesis?"
5. **LLM generates** educational response
6. **TTS synthesizes** response in detected language
7. **Robot speaks** the response
8. **Listening resumes**

### Interruption Example

1. Robot is speaking (TTS playing)
2. User starts speaking
3. VAD detects speech → calls `tts.stop()` immediately
4. Current response is discarded
5. New utterance is captured and processed
6. Robot responds to new input

### Stop the Pipeline

Press `Ctrl+C` at any time. The pipeline catches `KeyboardInterrupt` and exits cleanly.

---

## Project Structure

```
d:\AI_Robot\VoicePipline/
├── main.py                      # Main orchestration loop
├── conftest.py                  # Pytest configuration
├── pytest.ini                   # Pytest settings
├── README.md                    # This file
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Centralized configuration (dataclasses)
│   └── snakers4-silero-vad/     # Silero VAD model repo
│
├── modules/
│   ├── __init__.py
│   ├── vad_module.py            # Voice Activity Detection
│   ├── asr_module.py            # Automatic Speech Recognition
│   ├── llm_module.py            # LLM with session management
│   ├── tts_module.py            # Text-to-Speech
│   └── snakers4-silero-vad/     # Silero VAD model repo (backup)
│
├── test_modules/
│   ├── __init__.py
│   ├── test_vad.py              # VAD unit tests
│   ├── test_asr.py              # ASR unit tests
│   ├── test_llm.py              # LLM unit tests
│   └── test_tts.py              # TTS unit tests
│
└── robot_sessions.db            # SQLite database (auto-created)
```

### Key Files Explained

| File                    | Purpose                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| `main.py`               | Entry point; contains `VoicePipeline` class that orchestrates all four modules |
| `config/settings.py`    | Dataclass-based configuration loader; reads from environment variables         |
| `modules/vad_module.py` | Silero VAD wrapper; low-latency speech detection for embedded systems          |
| `modules/asr_module.py` | Google Speech Recognition wrapper; converts audio → text                       |
| `modules/llm_module.py` | Ollama integration; manages sessions, memory, summarization                    |
| `modules/tts_module.py` | Edge TTS wrapper; thread-safe, interruptible speech synthesis                  |
| `test_modules/*.py`     | Unit tests for each module (run with pytest)                                   |

---

## Architecture Deep Dive

### 1. VAD Module (`vad_module.py`)

**Purpose**: Detect speech in real-time with minimal CPU overhead.

**Key Design Decisions**:

- **Single model instance**: Model loaded once at startup (singleton pattern)
- **Single-threaded inference**: `torch.set_num_threads(1)` for low latency
- **Minimal hot-path work**: `is_speech()` just runs inference + threshold check
- **Configurable threshold**: `set_threshold(0.0–1.0)` to tune sensitivity

**Why Silero?**

- Lightweight (~50MB)
- Runs on CPU (no GPU required)
- Trained on 40+ languages (good multilingual support)
- Open-source from [GitHub](https://github.com/snakers4/silero-vad)

### 2. ASR Module (`asr_module.py`)

**Purpose**: Convert audio bytes to text with language detection.

**Features**:

- Google Speech Recognition API (free, with quota limits)
- Auto-language detection
- Supports flexible language modes: `auto`, `en`, `ar`
- Returns both text and detected language

**Dependencies**:

- `SpeechRecognition` library
- Internet connection required

### 3. LLM Module (`llm_module.py`)

**Purpose**: Intelligent conversational responses with memory management.

**Architecture**:

```
┌─────────────────────┐
│  OllamaConnection   │  ← Low-level HTTP communication
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ SessionManager      │  ← Session lifecycle, memory persistence
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ LLMModule           │  ← High-level chat interface
└─────────────────────┘
```

**Session Memory Strategy**:

- **Sliding Window**: Keep last N messages (default: 10)
- **Auto-summarization**: Every 10 messages, summarize earlier context
- **SQLite Persistence**: All sessions and messages stored in `robot_sessions.db`
- **Turn-based tracking**: Each conversation turn has a unique ID

**System Prompts**:
Pre-configured prompts for Arabic and English that guide the LLM to:

- Explain concepts simply
- Use relatable examples
- Encourage student thinking (not giving direct answers)
- Keep responses concise (2–3 sentences)
- Match language formality

### 4. TTS Module (`tts_module.py`)

**Purpose**: Convert text to speech with interruption support.

**Features**:

- Uses Edge TTS (Microsoft Azure) for high-quality voices
- Thread-safe with request-based interrupt handling
- Language auto-detection (Arabic vs. English)
- Pygame for audio playback
- MP3 files cached in temp directory

**Interrupt Mechanism**:
Each `speak()` call gets a unique request ID. If `stop()` is called, the playback thread checks the request ID; if it doesn't match, the request is ignored.

### 5. Main Orchestration (`main.py`)

**Purpose**: Coordinate all four modules in a real-time loop.

**Key Components**:

- **VoicePipeline class**: Main orchestrator
- **Audio callback**: Runs in sounddevice's real-time thread
  - Captures chunks
  - Feeds to VAD
  - Accumulates segment data
  - Spawns ASR/LLM/TTS worker threads
- **Segment deque**: Buffers pre-speech audio (configurable duration)
- **Threading**: Worker threads process segments; main thread handles audio capture

**Turn-based Processing**:

- Each user utterance gets a `turn_id`
- If user speaks while TTS is playing, a new turn starts
- Old turns are discarded (see `_get_latest_turn_id()`)
- This prevents delayed ASR/LLM responses from interrupting new input

---

## Testing

The project includes unit tests for each module.

### Run All Tests

```powershell
pytest -v
```

### Run Specific Test File

```powershell
pytest test_modules/test_vad.py -v
```

### Run with Coverage

```powershell
pip install pytest-cov
pytest --cov=modules test_modules/
```

### Test Files

| File          | Tests                                                     |
| ------------- | --------------------------------------------------------- |
| `test_vad.py` | VAD model loading, threshold validation, speech detection |
| `test_asr.py` | Audio recording, transcription, language modes            |
| `test_llm.py` | Session creation, chat, message storage, summarization    |
| `test_tts.py` | Voice synthesis, async playback, interruption             |

---

## Troubleshooting

### "Ollama server not available"

**Problem**: LLM module can't connect to Ollama.

**Solutions**:

1. Verify Ollama is running: `ollama serve` in a separate terminal
2. Check the host: `ROBOT_OLLAMA_HOST=http://localhost:11434`
3. Firewall may be blocking port 11434; check Windows Defender Firewall

```powershell
# Debug
python -c "import requests; print(requests.get('http://localhost:11434/api/tags'))"
```

### "Failed to import sounddevice"

**Problem**: Audio input/output unavailable.

**Windows Solution**:

```powershell
pip install --upgrade sounddevice

# Ensure PortAudio is installed (sounddevice will use Windows audio drivers)
```

### "No speech detected (VAD stuck in pre-buffer)"

**Problem**: VAD threshold too high or microphone too quiet.

**Solutions**:

1. Lower the VAD threshold: `ROBOT_VAD_THRESHOLD=0.50` (default: 0.60)
2. Check microphone volume in system settings
3. Increase `ROBOT_VAD_PRE_ROLL_SEC` (pre-speech buffer)
4. Test VAD directly:

```python
from modules import vad_module
import numpy as np

# Play audio and check is_speech()
test_audio = np.random.randn(16000).astype(np.float32)
print(vad_module.is_speech(test_audio))
```

### "ASR transcription is empty or wrong language"

**Problem**: ASR can't find speech or detects wrong language.

**Solutions**:

1. Ensure audio is clear and loud
2. Check language mode: `ROBOT_ASR_LANGUAGE_MODE=ar` (force Arabic)
3. Verify sample rate matches: `ROBOT_VAD_SAMPLE_RATE=ROBOT_ASR_SAMPLE_RATE=16000`
4. Google Speech API may be rate-limited; add delays between requests

### "TTS playback is stuttering or delayed"

**Problem**: Audio playback is choppy or lags.

**Solutions**:

1. Increase pygame polling interval: `ROBOT_TTS_POLL_SEC=0.1` (default: 0.05)
2. Check disk space in temp directory: `ROBOT_TTS_TEMP_DIR`
3. Reduce concurrent operations (don't queue multiple speak() calls)

### "Memory usage grows over time"

**Problem**: SQLite database or in-memory cache growing unbounded.

**Solutions**:

1. Clear old sessions: Remove `robot_sessions.db` or run cleanup script
2. Reduce `ROBOT_LLM_WINDOW_SIZE` (default: 10 messages)
3. Check Ollama model size: `ollama list`

### Logging for Debugging

Enable debug-level logging:

```powershell
$env:ROBOT_LOG_LEVEL = "DEBUG"
python main.py
```

This will show detailed info for VAD, ASR, LLM, and TTS operations.

---

## System Prompts

The LLM uses language-specific system prompts to guide behavior. These are defined in `config/settings.py`:

### English Prompt

Guides the robot to explain clearly, use examples, ask clarifying questions, and encourage student thinking.

### Arabic Prompt

Tailored to Modern Standard Arabic with Egyptian colloquial touches, emphasizing simplicity and engagement.

Both prompts enforce:

- ✅ Concise responses (2–3 sentences max)
- ✅ Encouraging student discovery (scaffolding, not direct answers)
- ✅ Relatable real-world examples
- ✅ Positive, enthusiastic tone
- ❌ No complex jargon without explanation
- ❌ No very long responses

---

## Performance Tuning

### For Embedded Systems (Raspberry Pi)

```powershell
# Reduce computational load
$env:ROBOT_VAD_TORCH_THREADS = "1"              # Single-threaded
$env:ROBOT_VAD_CHUNK_MS = "40"                   # Larger chunks = less overhead
$env:ROBOT_LLM_WINDOW_SIZE = "5"                 # Smaller memory window
$env:ROBOT_TTS_POLL_SEC = "0.1"                  # Longer polling interval
```

### For Fast Responses (Desktop)

```powershell
# Optimize for responsiveness
$env:ROBOT_VAD_SILENCE_TIMEOUT_SEC = "0.5"      # Faster end-of-speech detection
$env:ROBOT_VAD_MIN_SPEECH_SEC = "0.3"            # Lower minimum speech threshold
$env:ROBOT_TTS_POLL_SEC = "0.01"                 # Faster polling
```

---

## License & Attribution

This project uses:

- **Silero VAD**: Open-source from [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- **Google Speech Recognition**: Via the `SpeechRecognition` library
- **Ollama**: From [ollama.ai](https://ollama.ai)
- **Edge TTS**: Microsoft Azure Text-to-Speech

---

## Contributing & Customization

### Swapping Components

The modular design allows easy provider swaps:

1. **Change LLM provider**: Edit `llm_module.py` and update settings
2. **Change TTS engine**: Implement new engine in `tts_module.py` (Coqui, Piper, etc.)
3. **Change ASR provider**: Swap `asr_module.py` (Whisper, Azure Speech, etc.)
4. **Add VAD alternatives**: Implement WebRTC VAD or other models

All configuration goes in `config/settings.py` — no hardcoding!

### Adding Custom Features

Examples:

- Gender detection for voice selection
- Emotion analysis for TTS prosody
- Custom logging to external services
- Integration with student LMS platforms

---

## Support & Debugging

For issues, check:

1. Logs: Set `ROBOT_LOG_LEVEL=DEBUG`
2. Configuration: Verify all environment variables are set correctly
3. External services: Ensure Ollama is running and reachable
4. System state: Check microphone, speakers, disk space
5. Tests: Run `pytest` to validate each module independently

---

_Last Updated: March 2026_  
_For questions or contributions, reach out to the development team._
