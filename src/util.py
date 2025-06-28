# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
import re
from datetime import datetime
from pathlib import Path
from typing import List


def list_sessions(game_dir: Path) -> List[Path]:
    """Return a list of all session directories for *game_dir*, newest first."""
    if not game_dir.is_dir():
        return []
    
    # Session names are timestamps, e.g., "08-06-25_at_22.06.31"
    # This regex ensures we only pick up directories that match this format.
    session_pattern = re.compile(r"^\d{2}-\d{2}-\d{2}_at_\d{2}\.\d{2}\.\d{2}$")

    # Find all sub-directories that match the session name pattern
    sessions = [
        d
        for d in game_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and session_pattern.match(d.name)
    ]
    sessions.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return sessions

def get_latest_session(game_dir: Path) -> Path | None:
    sessions = list_sessions(game_dir)
    return sessions[0] if sessions else None
