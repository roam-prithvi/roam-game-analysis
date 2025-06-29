# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from typing import List


def list_sessions(game_dir: Path) -> List[Path]:
    """Return a list of all session directories for *game_dir*, newest first."""
    if not game_dir.is_dir():
        return []

    # Find all sub-directories (excluding hidden ones that start with ".")
    sessions = [
        d for d in game_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    sessions.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return sessions


def get_latest_session(game_dir: Path) -> Path | None:
    sessions = list_sessions(game_dir)
    return sessions[0] if sessions else None
