# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
import re
from pathlib import Path
from typing import List


def list_sessions(game_dir: Path) -> List[Path]:
    """Return all immediate sub-directories of *game_dir*, newest first."""
    subdirs: List[Path] = [p for p in game_dir.iterdir() if p.is_dir()]
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs


# ---------------------------------------------------------------------------
# 🏗️  Session-directory preparation helpers
# ---------------------------------------------------------------------------


def sanitize_path_component(component: str) -> str:
    """Return a filesystem-safe version of *component*."""
    # Keep letters, numbers, underscore, hyphen and space; drop the rest.
    safe = re.sub(r"[^A-Za-z0-9 _\-]", "", component).strip()
    # Compress whitespace to single spaces.
    safe = re.sub(r"\s+", " ", safe)
    return safe or "unnamed"
