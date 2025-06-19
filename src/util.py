# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
from pathlib import Path
from typing import List


def list_sessions(game_dir: Path) -> List[Path]:
    """Return all immediate sub-directories of *game_dir*, newest first."""
    subdirs: List[Path] = [p for p in game_dir.iterdir() if p.is_dir()]
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs
