# Privacy

The app avoids writing raw prompts and responses to normal JSON logs.

Langfuse prompt/response capture is controlled by:

```env
PRIVACY_REDACT_PROMPTS=false
PRIVACY_REDACT_RESPONSES=false
PRIVACY_HASH_USER_ID=true
```

Set the redaction flags to `true` if you want traces to keep metadata but hide prompt and response text.

The app masks common email addresses, phone-like values, and secret-like key/value strings before sending unredacted text to observability helpers.

When RAG is enabled, retrieved snippets from `data/knowledge/` are part of the
constructed prompt. Keep `PRIVACY_REDACT_PROMPTS=true` if those local knowledge
files contain private notes that should not appear in Langfuse traces.

Do not expose Langfuse publicly unless it has authentication and TLS.
