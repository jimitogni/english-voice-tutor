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

The current version records from your microphone, transcribes the audio with
`faster-whisper`, sends the text to a local Ollama model, prints the tutor's
response, streams text/audio when enabled, and can speak the answer with Piper TTS.
It also includes a FastAPI backend and a React frontend for a more structured
local web interface.

## Current Phase

Implemented through Phase 6:

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
- Energy-based voice activity detection
- Browser recording with automatic stop after silence in Streamlit
- Ollama streaming responses
- Sentence-by-sentence Piper TTS
- Lightweight pronunciation feedback
- Streamlit web UI
- FastAPI backend for browser and future mobile clients
- React frontend with fixed sidebar, scrollable chat, bottom voice dock, and auto-scroll
- Docker Compose web stack
- Optional JSON conversation saving
- Optional local RAG with Ollama embeddings and Qdrant
- RAG retrieval metrics in Prometheus and Evidently-compatible reports
- Ollama check script
- Typed full-loop test script

## Requirements

- Linux, preferably Debian/Ubuntu or a similar distro
- Python 3.10+
- Ollama installed and running locally
- One local model pulled with Ollama, such as `llama3.2:3b` or `qwen2.5:3b`
- Optional: one local embedding model pulled with Ollama, such as `embeddinggemma`
- A working microphone
- PortAudio runtime libraries for `sounddevice`
- Piper TTS and a downloaded Piper voice model
- An audio playback tool such as `aplay`
- Node.js 18+ if you want to run the React frontend in development mode

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

For the React frontend, install Node.js 18 or newer. On Debian/Ubuntu you can
use your preferred Node installer, for example `nvm`, NodeSource packages, or
your distro package if it is recent enough:

```bash
node --version
npm --version
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

Pull the default local embedding model if you enable RAG:

```bash
ollama pull embeddinggemma
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

## Run The Terminal App

```bash
python -m app.main
```

The default input mode is now `voice`. Press Enter to start recording. The app
records for `RECORD_SECONDS`, saves a WAV file in `data/audio_inputs/`,
transcribes it, sends the transcription to Ollama, prints the tutor response,
generates speech with Piper, and plays the WAV file.

By default, the app asks you to choose a tutor mode at startup.
Ollama streaming and sentence-by-sentence Piper playback are enabled by default.

You can override the recording duration:

```bash
python -m app.main --record-seconds 5
```

Use voice activity detection so recording stops after silence:

```bash
python -m app.main --record-mode vad --record-seconds 12
```

Disable streaming for one run:

```bash
python -m app.main --no-stream
python -m app.main --no-stream-tts
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
ASSISTANT_NAME=Jarvis
USER_DISPLAY_NAME=Jimi Jeday Marster
CONVERSATION_HISTORY_TURNS=40
FOCUS_WORDS_LIMIT=10
STT_MODEL_SIZE=base
STT_LANGUAGE=en
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
PRONUNCIATION_FEEDBACK=true
RECORD_SECONDS=8
RECORD_MODE=fixed
SAMPLE_RATE=16000
VAD_ENERGY_THRESHOLD=0.015
VAD_SILENCE_SECONDS=2.2
VAD_MIN_SPEECH_SECONDS=0.4
VAD_MAX_SECONDS=300
VAD_CHUNK_MS=30
LLM_STREAM=true
TTS_ENGINE=piper
STREAM_TTS=true
PIPER_CUDA=false
PIPER_EXECUTABLE=piper
PIPER_MODEL_PATH=./models/piper/en_US-lessac-medium.onnx
PIPER_CONFIG_PATH=./models/piper/en_US-lessac-medium.onnx.json
SAVE_CONVERSATIONS=true
RAG_ENABLED=false
RAG_VECTOR_DB=qdrant
RAG_EMBEDDING_MODEL=embeddinggemma
RAG_TOP_K=4
RAG_SCORE_THRESHOLD=0.25
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=english_voice_tutor_knowledge
KNOWLEDGE_DIR=./data/knowledge
```

`ASSISTANT_NAME` tells the tutor what name to respond to. With the default
configuration, when you say "Jarvis", the tutor understands you are talking to
it. `USER_DISPLAY_NAME` is used so new sessions greet you by name.

`CONVERSATION_HISTORY_TURNS` controls how many recent user/assistant turns are
sent back to the LLM as short-term context. The default is now 40, increased
from the earlier MVP value of 8. Higher values give Jarvis more session memory,
but very high values can make local models slower or exceed their context
window.

`FOCUS_WORDS_LIMIT` controls how many fixed practice words or expressions you
can keep in the Streamlit sidebar. The default is 10.

## Conversation Memory

The app keeps recent turns in memory and can save sessions to:

```text
data/conversations/
```

By default, the LLM receives the latest 40 turns from the current session:

```env
CONVERSATION_HISTORY_TURNS=40
```

Saved JSON files still store the whole session, while the active LLM context uses
the latest configured number of turns.

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

## Local RAG Knowledge

RAG is optional and uses local Ollama embeddings plus Qdrant. Add private notes,
interview examples, grammar explanations, or vocabulary material under:

```text
data/knowledge/
```

Supported file types are `.md`, `.txt`, `.json`, `.jsonl`, and `.csv`.
The folder contents are ignored by git by default.

Start Qdrant locally with Docker Compose, pull the embedding model, then index
the knowledge folder:

```bash
docker compose up -d qdrant
ollama pull embeddinggemma
python scripts/index_rag_knowledge.py --reset
```

Enable retrieval in `.env`:

```env
RAG_ENABLED=true
QDRANT_URL=http://localhost:6333
```

For the homelab container stack, run the indexer inside the API container so it
uses the internal `http://qdrant:6333` service URL:

```bash
docker compose --env-file .env.homelab exec -T api python scripts/index_rag_knowledge.py --reset
```

If the Qdrant collection has not been created yet, the tutor skips retrieval and
continues normal chat. API chat responses include `sources`,
`retrieval_count`, and `retrieval_error` fields for evaluation and debugging.

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

## Voice Activity Detection

The app supports two recording modes:

```env
RECORD_MODE=fixed
RECORD_MODE=vad
```

`fixed` records for `RECORD_SECONDS`. `vad` uses a simple local energy
threshold and stops after sustained silence. The Streamlit browser recorder also
uses these silence settings, so one set of values controls both terminal and web
voice capture. Tune these values if recording cuts off too early or waits too
long:

```env
VAD_ENERGY_THRESHOLD=0.015
VAD_SILENCE_SECONDS=2.2
VAD_MIN_SPEECH_SECONDS=0.4
VAD_MAX_SECONDS=300
VAD_CHUNK_MS=30
```

In React, click the sidebar's Voice Detection row to show or hide these values.
In Streamlit, open the sidebar's Voice detection panel to tune them.

## Streaming

Ollama streaming is enabled by default:

```env
LLM_STREAM=true
```

In terminal mode, streamed chunks are printed as they arrive. Piper can also
synthesize sentence chunks as they appear:

```env
STREAM_TTS=true
```

Use these flags to compare behavior:

```bash
python -m app.main --no-stream
python -m app.main --no-stream-tts
```

## Pronunciation Feedback

After STT, the app shows a lightweight pronunciation/audio note based on
Whisper segment confidence and silence probability. This is not phoneme-level
scoring yet, but it gives quick feedback when the transcription looked uncertain
or noisy.

Disable it with:

```env
PRONUNCIATION_FEEDBACK=false
```

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

This project can also map one Piper voice to each local LLM model. The current
default mapping is:

```text
llama3.2:3b -> en_US-lessac-medium
qwen3:4b    -> en_US-amy-medium
gemma3:4b   -> en_US-ryan-medium
```

Download the extra mapped voices with:

```bash
wget -O models/piper/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget -O models/piper/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
wget -O models/piper/en_US-ryan-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
wget -O models/piper/en_US-ryan-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
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

## Streamlit Web UI

The project includes a small Streamlit interface for local browser-based
practice:

```bash
streamlit run ui/streamlit_app.py
```

Open the URL Streamlit prints, usually:

```text
http://localhost:8501
```

The UI supports:

- tutor mode selection
- Ollama model selection for installed local models
- automatic Piper voice mapping per selected LLM model
- typed chat
- a focus-words panel for up to 10 words or expressions you want to practice
- browser microphone recording with automatic stop after silence
- streaming Ollama responses
- optional audio-file upload for STT, including WAV, MP3, M4A, WEBM, and OGG
- automatic browser playback of generated Piper WAV files

Use the "Voice" recorder. Click "Start listening", speak naturally, then pause.
The browser detects silence, stops recording, transcribes your speech, sends it
to the local LLM, generates a Piper WAV answer, and plays it in the browser when
browser audio is enabled. The "Stop now" button is still there as a manual
fallback.

Use the sidebar's "Ollama model" selector to switch between installed local
models during a Streamlit session. The default still comes from `OLLAMA_MODEL`
in `.env`. The current model is also shown near the top of the page and inside
each assistant message, so you can compare responses later. The mapped Piper
voice is shown alongside it.

Leave "Autoplay answer audio" enabled to hear the response without pressing play.
If your browser blocks autoplay with sound, the audio player remains visible so
you can press play manually. If the browser blocks microphone access, allow
microphone permission for `localhost` or use the audio upload/typed chat
fallback.

Use the sidebar's "Focus Words" panel to add words or expressions you want to
practice more. Jarvis sees this list on every response and will naturally reuse
or quiz you on those words when useful. The list is stored locally at:

```text
data/vocabulary/focus_words.json
```

## FastAPI + React Web UI

The newer web interface separates the local backend from the browser frontend:

```text
Browser / React
  -> FastAPI
  -> faster-whisper / Ollama / Piper
  -> generated WAV response
  -> Browser audio player
```

Start the FastAPI backend from the project root:

```bash
source .venv/bin/activate
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

Or use the helper script:

```bash
python scripts/run_api.py
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

In another terminal, start the React frontend:

```bash
cd web
cp .env.example .env
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The React UI includes:

- a fixed left sidebar for tutor mode, Ollama model, mapped Piper voice, TTS toggle, memory, and focus words
- a scrollable chat area with automatic scroll to the latest message
- a fixed bottom voice dock with start/stop recording and typed input
- browser silence detection that stops recording and sends the audio automatically
- automatic playback of the generated Piper response when browser autoplay allows it
- visible model and voice labels inside the assistant messages

Build the React app for production:

```bash
cd web
npm run build
```

After a build, FastAPI serves the compiled frontend at:

```text
http://127.0.0.1:8000/app
```

## Homelab Docker Compose

The homelab deployment runs the newer FastAPI + React architecture:

```text
React / Nginx container
  -> FastAPI API container
  -> Ollama container
  -> mounted ./models and ./data folders
```

Services:

- `web`: Nginx serves the built React app and proxies `/api` to FastAPI.
- `api`: FastAPI runs STT, tutor orchestration, Piper TTS, focus words, and conversation saving.
- `ollama`: Ollama stores local LLM models in a named Docker volume.
- `qdrant`: Qdrant stores local RAG vectors in a named Docker volume.

Persistent paths:

- Ollama models: Docker volume `ollama`
- Qdrant vectors: Docker volume `qdrant`
- Piper voice models: `./models`
- conversations, vocabulary, RAG knowledge, generated audio, uploaded audio: `./data`

On a fresh Debian/Ubuntu homelab server, install Docker first:

```bash
scripts/homelab_install_docker.sh
```

Deploy the app:

```bash
scripts/homelab_deploy.sh
```

The deploy script:

- creates `.env.homelab` from `.env.homelab.example` if needed
- creates `data/` and `models/` folders
- downloads the Lessac, Amy, and Ryan Piper voices
- builds the API and React containers
- starts Ollama, API, and web services
- pulls `OLLAMA_MODEL` plus `EXTRA_OLLAMA_MODELS`
- pulls `RAG_EMBEDDING_MODEL` when `RAG_ENABLED=true`
- runs a health check

Default local homelab URLs:

```text
http://localhost:8080        React UI
http://localhost:8000/docs   FastAPI docs
http://localhost:11434       Ollama API
http://localhost:6333        Qdrant API/dashboard
```

From another computer on your LAN, use your server IP:

```text
http://YOUR_SERVER_IP:8080
```

Important: browser microphone access usually requires `localhost` or HTTPS.
If you open the app from another machine through plain `http://SERVER_IP:8080`,
the browser may block microphone access. Use one of these:

- HTTPS through Traefik, Caddy, Nginx Proxy Manager, or another reverse proxy
- an SSH tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 user@YOUR_SERVER_IP
```

Then open this on your laptop:

```text
http://127.0.0.1:8080
```

Useful homelab commands:

```bash
scripts/homelab_health.sh
scripts/homelab_logs.sh
docker compose --env-file .env.homelab ps
docker compose --env-file .env.homelab down
```

Optional GPU acceleration for Ollama, faster-whisper, and Piper:

```bash
ENABLE_GPU=1 scripts/homelab_deploy.sh
```

This uses `docker-compose.gpu.yml` to start the Ollama and API containers with
`gpus: all`. The host still needs a working NVIDIA driver and NVIDIA Container
Toolkit for Docker. For GPU STT/TTS, set these in `.env.homelab`:

```env
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
PIPER_CUDA=true
```

Verify Ollama GPU use with:

```bash
docker compose --env-file .env.homelab -f docker-compose.yml -f docker-compose.gpu.yml exec -T ollama ollama ps
```

The `PROCESSOR` column should show GPU use instead of `100% CPU`.

## LLMOps Observability

English Voice Tutor includes optional observability for the homelab deployment:

- Langfuse tracing for chat requests, prompt construction, Ollama generations, TTS spans, model names, session IDs, request IDs, latency, and token counts when Ollama returns them.
- Prometheus metrics at `/english/api/metrics`.
- Grafana dashboard JSON at `docs/observability/grafana_llm_dashboard.json`.
- Evidently-compatible quality, retrieval, and latency reports under `data/reports/evidently`.
- Per-interaction evaluation records under `data/evaluation/interactions/`.
- Dataset-based evaluation runs and run summaries under `data/evaluation/results/`.
- Observability summary API at `/api/observability/summary`.
- Structured JSON logs when `LOG_JSON=true`.

The React sidebar now includes a lightweight observability dashboard card that shows recent interaction counts, errors, latency, feedback, tool-call stats, and the latest offline eval run summary.

The evaluation layer now records or computes:

- BLEU, ROUGE-1, and ROUGE-L when a reference answer exists
- semantic similarity through a local embedding backend
- response length, latency, token usage, and estimated cost
- error rate, factuality or hallucination score when reference context exists, and grammar correction quality when a reference correction exists
- task success, tool call success or error rates, and tool calls per interaction
- optional user feedback scores through `POST /api/feedback`

The default semantic backend uses the local Ollama embedding model already configured for RAG. You can switch to `sentence-transformers` later by installing it and changing `.env`:

```env
EVALUATION_ENABLED=true
EVALUATION_DATA_DIR=./data/evaluation
EVALUATION_EMBEDDING_BACKEND=ollama
EVALUATION_EMBEDDING_MODEL=embeddinggemma
EVALUATION_COST_INPUT_PER_MILLION=0
EVALUATION_COST_OUTPUT_PER_MILLION=0
```

Example evaluation dataset:

```text
evals/english_practice_eval.jsonl
```

Langfuse is optional, but this repo now includes a bundled self-hosted overlay in
`docker-compose.langfuse.yml`. To enable it in the homelab stack, set these in
`.env.homelab`:

```env
ENABLE_LANGFUSE_STACK=1
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=<project-public-key>
LANGFUSE_SECRET_KEY=<project-secret-key>
```

The overlay supports headless bootstrap via `LANGFUSE_INIT_*`, so the first org,
project, user, and API keys can be provisioned during startup. If you already run a
private Langfuse elsewhere, keep `ENABLE_LANGFUSE_STACK=0` and point
`LANGFUSE_HOST` / `LANGFUSE_PUBLIC_URL` at that deployment instead.

For internet access behind this repo's Traefik setup, prefer a dedicated hostname
such as `http://langfuse-jimitogni.duckdns.org:8888`. A `/langfuse` path-prefix
deployment is possible, but Langfuse's current self-hosted docs require rebuilding
the web image with a custom base path for that mode.

The existing homelab Prometheus can scrape the app through the web container:

```yaml
- job_name: english-voice-tutor
  metrics_path: /english/api/metrics
  static_configs:
    - targets:
        - english-voice-tutor-web:80
```

Run the local checks:

```bash
scripts/observability/check_observability_stack.sh
scripts/observability/send_test_chat_request.sh
scripts/observability/test_metrics_endpoint.sh
python scripts/observability/run_evidently_eval.py
```

More details are in `docs/observability/`.

Edit `.env.homelab` to change ports, models, VAD settings, identity, or STT/TTS
settings:

```env
WEB_PORT=8080
API_PORT=8000
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2:3b
EXTRA_OLLAMA_MODELS=qwen3:4b gemma3:4b
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

### React frontend command not found

If `npm` or `node` is missing, install Node.js 18 or newer, then run:

```bash
cd web
npm install
npm run dev
```

The Python backend still works without Node:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

### React frontend cannot reach the API

Make sure FastAPI is running on port 8000:

```bash
curl http://127.0.0.1:8000/api/status
```

If you use another backend URL, set it in `web/.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

For the default Vite development setup, `VITE_API_BASE_URL` can stay empty
because Vite proxies `/api` to `http://127.0.0.1:8000`.

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
- Streaming, VAD, pronunciation notes, Docker Compose, Streamlit, and React are local
  features; they do not introduce paid API usage.
- FastAPI and React are local development components; the LLM, STT, and TTS
  still run through Ollama, faster-whisper, and Piper.

## Roadmap

1. Phase 1: typed terminal tutor loop with Ollama
2. Phase 2: microphone recording and faster-whisper STT
3. Phase 3: Piper TTS and audio playback
4. Phase 4: richer JSON session saving and progress metadata
5. Phase 5: tutor modes for free conversation, interview practice, and vocabulary
6. Phase 6: latency improvements, streaming responses, voice activity detection, pronunciation notes, Docker Compose, Streamlit UI, and FastAPI + React UI

## Implemented Advanced Features

- Voice activity detection: energy-based silence detection in the terminal,
  Streamlit browser recorder, and React voice dock.
- Streaming LLM responses: Ollama streaming chunks in terminal and Streamlit.
- Streaming TTS: sentence-by-sentence Piper synthesis in the terminal.
- Pronunciation feedback: lightweight notes from Whisper segment confidence and
  silence probability.
- Local RAG: Ollama embeddings plus Qdrant retrieval over files in
  `data/knowledge/`.
- Docker Compose: homelab stack with Ollama, FastAPI, React/Nginx, and mounted model/data folders.
- Web interfaces: Streamlit UI and React UI for typed chat, silence-based
  browser microphone recording, mode/model selection, focus words, and generated
  audio playback.

## Future Version Ideas

- Replace energy VAD with `webrtcvad` or `silero-vad` for stronger speech
  detection in noisy rooms.
- Make TTS playback concurrent so sentence synthesis does not pause LLM stream
  consumption.
- Add phoneme-level pronunciation scoring and target phrase comparison.
- Add Server-Sent Events or WebSockets for token-by-token streaming in the React UI.
- Add HTTPS reverse-proxy examples for Caddy, Traefik, and Nginx Proxy Manager.
- Add persistent learner profiles, vocabulary review, and progress charts.

## Development Notes

Run Python from the project root:

```bash
python -m app.main
```

The project avoids paid APIs and closed-source API dependencies. Ollama talks to
a local server at `http://localhost:11434`, STT runs through local
`faster-whisper`, and TTS runs through local Piper voice models and local
executables.
