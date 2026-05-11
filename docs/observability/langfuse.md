# Langfuse

Langfuse is used for LLM tracing when enabled.

The app records:

- request ID
- session ID
- hashed user ID when enabled
- tutor mode
- model name
- prompt construction span
- Ollama generation event
- token counts from Ollama when available
- latency
- TTS span
- success/error status

## Configuration

Set these in `.env.homelab` after a private Langfuse instance exists:

```env
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://<private-langfuse-host>:3000
LANGFUSE_PUBLIC_KEY=<project-public-key>
LANGFUSE_SECRET_KEY=<project-secret-key>
```

Do not commit real keys. The current `.env.example` uses placeholders.

## Test

```bash
python scripts/observability/test_langfuse_trace.py
scripts/observability/send_test_chat_request.sh
```

If Langfuse is disabled or unavailable, the app continues normally and logs a warning.
