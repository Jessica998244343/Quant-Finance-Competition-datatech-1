"""Run-directory policy: new runs use unique IDs and refuse to overwrite by default."""

from __future__ import annotations

from pathlib import Path


class RunExistsError(FileExistsError):
    """Raised when a completed run directory would be overwritten."""


def new_run_dir(path, allow_existing: bool = False) -> Path:
    """Create a run directory; refuse to reuse an existing one unless allowed."""
    path = Path(path)
    if path.exists() and not allow_existing:
        raise RunExistsError(
            f"{path} already exists; pick a new run ID or pass --allow-existing")
    path.mkdir(parents=True, exist_ok=True)
    return path
