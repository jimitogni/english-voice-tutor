# english-voice-tutor

An offline/local English conversation tutor built with open-source technologies.

The long-term goal is a local voice loop:

```text
Microphone
  -> audio_recorder.py
  -> stt.py
  -> tutor_agent.py
  -> llm_client.py / Ollama
  -> tts.py / Piper
  -> Speaker
```

The current version records from your microphone, transcribes the WAV file with
`faster-whisper`, sends the text to a local Ollama model, prints the tutor's
response, and can speak the answer with Piper TTS.

## Current Phase

Implemented through Phase 5:

- Project structure
- Environment-based configuration
- Ollama REST client
- English tutor agent
- Prompt definitions for free conversation, interview practice, and vocabulary practice
- Typed terminal conversation loop
- Microphone recording with `sounddevice`
- Timestamped WAV files in `data/audio_inputs/`
- English speech-to-text with `faster-whisper`
- Voice terminal loop with typed fallback
- Local text-to-speech with Piper
- Timestamped WAV outputs in `data/audio_outputs/`
- Audio playback through `aplay`, `paplay`, `pw-play`, `ffplay`, or `sounddevice`
- Tutor modes for free conversation, interview practice, and vocabulary practice
- Optional JSON conversation saving
- Ollama check script
- Typed full-loop test script

## Requirements

- Linux, preferably Debian/Ubuntu or a similar distro
- Python 3.10+
- Ollama installed and running locally
- One local model pulled with Ollama, such as `llama3.2:3b` or `qwen2.5:3b`
- A working microphone
- PortAudio runtime libraries for `sounddevice`
- Piper TTS and a downloaded Piper voice model
- An audio playback tool such as `aplay`

## Setup

From this directory:

```bash
sudo apt install python3-full python3-venv libportaudio2 portaudio19-dev alsa-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

On some Debian/Ubuntu Python 3.13 installs, the virtual environment may have
`python -m pip` but no `pip` executable in `.venv/bin/`. Prefer
`python -m pip ...` for all install commands. If venv creation was interrupted,
remove the incomplete environment and create it again:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If venv creation itself fails, install the distro venv support package:

```bash
sudo apt install python3-full python3-venv
```

Edit `.env` if you want a different local model:

```env
OLLAMA_MODEL=qwen2.5:3b
```

## Install Ollama

Official Linux documentation: https://docs.ollama.com/linux

Quick install:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```bash
ollama serve
```

Or use the system service if you installed it that way:

```bash
sudo systemctl start ollama
sudo systemctl status ollama
```

Pull a local model:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

Other possible models:

```bash
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull gemma3:4b
```

Use smaller models first if your machine has limited RAM or no GPU.

## Check Ollama

```bash
python scripts/check_ollama.py
```

This verifies that Ollama is reachable at `OLLAMA_BASE_URL`, lists available local models, and tells you if the configured model is missing.

## Run Phase 3 Voice Mode

```bash
python -m app.main
```

The default input mode is now `voice`. Press Enter to start recording. The app
records for `RECORD_SECONDS`, saves a WAV file in `data/audio_inputs/`,
transcribes it, sends the transcription to Ollama, prints the tutor response,
generates speech with Piper, and plays the WAV file.

By default, the app asks you to choose a tutor mode at startup.

You can override the recording duration:

```bash
python -m app.main --record-seconds 5
```

Keep the typed fallback for debugging:

```bash
python -m app.main --input typed
```

Disable TTS for one run:

```bash
python -m app.main --no-tts
python -m app.main --input typed --no-tts
```

Select a mode directly:

```bash
python -m app.main --input voice --mode free
python -m app.main --input voice --mode interview
python -m app.main --input voice --mode vocabulary
```

Or open the mode picker explicitly:

```bash
python -m app.main --mode choose
```

Useful commands inside the app:

```text
/help
/quit
```

Run one typed interaction for debugging:

```bash
python scripts/test_full_loop.py "I am thinking in create a local project."
```

Transcribe one existing WAV file:

```bash
python scripts/test_stt.py data/audio_inputs/example.wav
```

The first `faster-whisper` transcription may download the selected open model
into your local cache. After that, it can run locally from the cached files.

## Configuration

The app reads these values from `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
STT_MODEL_SIZE=base
STT_LANGUAGE=en
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
RECORD_SECONDS=8
SAMPLE_RATE=16000
TTS_ENGINE=piper
PIPER_EXECUTABLE=piper
PIPER_MODEL_PATH=./models/piper/en_US-lessac-medium.onnx
PIPER_CONFIG_PATH=./models/piper/en_US-lessac-medium.onnx.json
SAVE_CONVERSATIONS=true
```

## Conversation Memory

The app keeps recent turns in memory and can save sessions to:

```text
data/conversations/
```

Each saved turn includes:

- Timestamp
- User transcription or typed text
- Tutor response
- Ollama model name
- STT model name, for example `typed-input` or `faster-whisper:base`

Disable saving for one run:

```bash
python -m app.main --no-save
```

Or set:

```env
SAVE_CONVERSATIONS=false
```

## Tutor Modes

### Free Conversation

Use this for natural conversation practice. The tutor answers naturally,
corrects grammar gently, suggests useful expressions when relevant, and asks one
follow-up question to keep you speaking.

```bash
python -m app.main --mode free
```

### Interview Practice

Use this to practice professional answers for AI Engineering, Machine Learning,
Data Science, MLOps, and Data Engineering interviews. The tutor asks one
interview question at a time, then gives concise feedback on grammar,
professional phrasing, and content.

```bash
python -m app.main --mode interview
```

### Vocabulary

Use this to learn one useful word or expression at a time. The tutor explains
the expression, gives a short example, asks you to create your own sentence, and
then corrects it gently.

```bash
python -m app.main --mode vocabulary
```

All modes answer in English unless you explicitly ask for Portuguese.

## Piper Text-To-Speech

Piper is the preferred local TTS engine for this project. The maintained Piper
project is now `OHF-Voice/piper1-gpl`, which installs with `pip install
piper-tts`. The older `rhasspy/piper` repository is archived and points to the
newer fork.

Project repository: https://github.com/OHF-Voice/piper1-gpl

Install Piper inside the virtual environment:

```bash
source .venv/bin/activate
python -m pip install piper-tts
```

The project `requirements.txt` already includes `piper-tts`, so this is also
installed by:

```bash
python -m pip install -r requirements.txt
```

Download an English voice model and its matching config file:

```bash
mkdir -p models/piper
wget -O models/piper/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O models/piper/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Test Piper directly:

```bash
echo "Hello, this is a test." | piper \
  --model models/piper/en_US-lessac-medium.onnx \
  --output_file data/audio_outputs/test.wav
```

Play the test file:

```bash
aplay data/audio_outputs/test.wav
```

Test through the project:

```bash
python scripts/test_tts.py "Hello, this is my local English tutor."
```

## Speech-To-Text

Speech-to-text uses `faster-whisper` as the first backend. The `app/stt.py`
module exposes this interface:

```python
transcribe(audio_path: str) -> str
```

Useful model sizes:

```env
STT_MODEL_SIZE=tiny
STT_MODEL_SIZE=base
STT_MODEL_SIZE=small
STT_MODEL_SIZE=medium
```

Start with `base` for a balance of speed and accuracy. Use `tiny` if your CPU is
slow. The app is designed so a later `whisper.cpp` backend can be added without
changing the tutor agent.

For CPU-only Linux machines, the default STT settings are:

```env
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
```

If you later configure CUDA support for `faster-whisper`, you can experiment
with:

```env
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
```

## Troubleshooting

### Ollama not running

Run:

```bash
ollama serve
```

Then check:

```bash
python scripts/check_ollama.py
```

If `ollama serve` says `bind: address already in use`, Ollama is probably
already running as a systemd service. Check it with:

```bash
systemctl status ollama
python scripts/check_ollama.py
```

### `ollama: command not found`

Ollama is not a Python package and is not installed by the virtual environment.
Install it at the system/user level first:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then open a new terminal, or reload your shell, and check:

```bash
command -v ollama
ollama --version
```

If the command is still missing, check whether `/usr/local/bin` is on your
`PATH`:

```bash
echo $PATH
```

### Model not found

Pull the configured model:

```bash
ollama pull llama3.2:3b
```

Or edit `.env`:

```env
OLLAMA_MODEL=qwen2.5:3b
```

### Microphone not working

Check Linux audio devices with:

```bash
arecord -l
python -m sounddevice
```

Install PortAudio packages:

```bash
sudo apt install libportaudio2 portaudio19-dev
```

If your desktop environment uses PipeWire or PulseAudio, make sure the default
input device is selected in your sound settings.

### `sounddevice` or PortAudio import error

Reinstall the Python dependencies inside the virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then install the system audio libraries:

```bash
sudo apt install libportaudio2 portaudio19-dev
```

### faster-whisper model load failure

Use a smaller STT model in `.env`:

```env
STT_MODEL_SIZE=tiny
```

The first run may need internet access to download the open Whisper model. Also
check that you have enough disk space for the model cache.

### Piper not installed

Install Piper in the virtual environment:

```bash
source .venv/bin/activate
python -m pip install piper-tts
```

Then check:

```bash
command -v piper
piper --help
```

If you installed Piper somewhere else, set:

```env
PIPER_EXECUTABLE=/absolute/path/to/piper
```

### Piper voice model missing

Check that these files exist:

```text
models/piper/en_US-lessac-medium.onnx
models/piper/en_US-lessac-medium.onnx.json
```

### Audio playback not working

Install a command-line WAV player:

```bash
sudo apt install alsa-utils
```

The app tries `aplay`, `paplay`, `pw-play`, `ffplay`, and finally Python
`sounddevice` playback.

## Quality And Privacy Notes

- The app uses local Ollama, local `faster-whisper`, and local Piper TTS.
- No paid APIs are used.
- No API keys or secrets are required.
- `.env` is ignored by git.
- Audio inputs, audio outputs, and saved conversations are ignored by git except
  for directory `.gitkeep` files.
- If STT fails for one recording, the app keeps the session alive and lets you
  try again.
- If TTS fails, the app disables speech output for that run and continues in
  text-only mode.

## Roadmap

1. Phase 1: typed terminal tutor loop with Ollama
2. Phase 2: microphone recording and faster-whisper STT
3. Phase 3: Piper TTS and audio playback
4. Phase 4: richer JSON session saving and progress metadata
5. Phase 5: tutor modes for free conversation, interview practice, and vocabulary
6. Phase 6: latency improvements, streaming responses, voice activity detection, and optional Streamlit/FastAPI UI

## Future Version Ideas

- Voice activity detection: replace fixed-duration recording with silence-based
  stop detection using `webrtcvad`, `silero-vad`, or an energy threshold.
- Streaming LLM responses: use Ollama streaming responses so text starts
  appearing before the full answer is complete.
- Streaming TTS: split assistant responses into sentences and synthesize/play
  chunks as they arrive.
- Pronunciation feedback: compare STT output with expected phrases, use Whisper
  timestamps/confidence signals, or add a phoneme-level scoring backend later.
- Docker Compose: package Ollama, app dependencies, mounted model folders, and
  persistent data volumes. Audio device access will need explicit Linux device
  mapping.
- Web interface: add Streamlit for a quick local practice UI, or FastAPI plus a
  small frontend for cleaner audio upload, mode selection, and session history.

## Development Notes

Run Python from the project root:

```bash
python -m app.main
```

The project avoids paid APIs and closed-source API dependencies. Ollama talks to
a local server at `http://localhost:11434`, STT runs through local
`faster-whisper`, and TTS runs through local Piper voice models and local
executables.
