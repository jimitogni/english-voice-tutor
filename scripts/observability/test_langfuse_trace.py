from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.observability.langfuse_client import get_langfuse_tracer


def main() -> None:
    config = load_config()
    tracer = get_langfuse_tracer()
    request_id = f"manual-langfuse-test-{uuid4().hex[:8]}"
    trace = tracer.create_trace(
        name="manual_langfuse_test",
        request_id=request_id,
        session_id="manual-test",
        user_id_hash=None,
        input_text="Manual Langfuse connectivity test.",
        metadata={"service": config.service_name, "environment": config.observability_env},
    )
    tracer.update_trace(trace, output_text="Langfuse test completed.", metadata={"status": "success"})
    tracer.flush()
    if trace is None:
        print("Langfuse is disabled, missing credentials, or unavailable. No trace was sent.")
    else:
        print(f"Sent Langfuse test trace with request_id={request_id}")


if __name__ == "__main__":
    main()
