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

This repo includes an optional Langfuse overlay at `docker-compose.langfuse.yml`.
Enable it in `.env.homelab`:

```env
ENABLE_LANGFUSE_STACK=1
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_HOST=langfuse-jimitogni.duckdns.org
LANGFUSE_PUBLIC_URL=http://langfuse-jimitogni.duckdns.org:8888
```

The bundled stack can bootstrap the first Langfuse org, project, user, and API
keys through `LANGFUSE_INIT_*` variables. Keep `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` aligned with `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` and
`LANGFUSE_INIT_PROJECT_SECRET_KEY`.

The overlay publishes Traefik routers for the dedicated Langfuse hostname. That
subdomain approach is the simplest supported internet-facing setup with the stock
Langfuse image. A `/langfuse` path-prefix route would require rebuilding the web
image with a custom base path.

If you are pointing at an existing private Langfuse instance instead, set:

```env
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://<private-langfuse-host>:3000
LANGFUSE_PUBLIC_URL=http://<private-langfuse-host>:3000
LANGFUSE_PUBLIC_KEY=<project-public-key>
LANGFUSE_SECRET_KEY=<project-secret-key>
```

Do not commit real keys. The current `.env.example` uses placeholders.

## Test

```bash
docker compose --env-file .env.homelab -f docker-compose.yml -f docker-compose.langfuse.yml up -d
python scripts/observability/test_langfuse_trace.py
scripts/observability/send_test_chat_request.sh
```

If Langfuse is disabled or unavailable, the app continues normally and logs a warning.
