# Interview Explanation

I added an LLMOps observability layer around a local voice-based LLM application running in a Docker homelab.

The app uses FastAPI for the API, Ollama for local LLM inference, Faster Whisper for speech-to-text, and Piper for text-to-speech. Observability is split into three layers:

1. Langfuse captures request-level traces. Each chat request becomes a trace with request ID, session ID, hashed user identity, model name, tutor mode, latency, prompt construction, Ollama generation metadata, and TTS spans.
2. Prometheus exposes operational metrics from the app. Metrics include FastAPI request rate and latency, LLM request counts, LLM latency, errors, estimated or Ollama-provided token counts, STT latency, and TTS latency.
3. Evidently runs offline evaluations against a JSONL dataset. It captures answers, latency, model name, error status, response length, empty answer rate, and expected keyword coverage, then writes HTML reports.

Grafana reuses the existing homelab Grafana and Prometheus stack. The app exposes `/api/metrics`, and Prometheus can scrape the app privately through the Docker network using the web container route `/english/api/metrics`.

For privacy, raw prompts and responses are not written into normal app logs. Langfuse prompt/response capture is controlled by environment variables, and user identity is hashed before being sent to traces.

To troubleshoot poor answers, I correlate the request ID across logs, Prometheus metrics, Langfuse traces, and Evidently evaluation results. That shows whether the issue is latency, model behavior, STT transcription, prompt construction, or TTS.
