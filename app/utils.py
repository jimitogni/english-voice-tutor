from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp() -> str:
    """Return an ISO timestamp suitable for saved conversation metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_timestamp() -> str:
    """Return a compact timestamp safe for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path
