from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

# Re-use the robust log parser already defined in the visualisation helper
from src.processing.visualize import parse_touch_log
from src.streaming.android_streamer import sanitize_path_component
from src.util import list_sessions
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
#  CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------------
# Interval for generic timeline thumbnails (seconds)
FRAME_INTERVAL: float = 1.0
# Minimum pause after a BTN_TOUCH/"up" event to conclude that one interaction
# has finished (seconds)
TOUCH_COOLDOWN: float = 0.05
# How many frames before / after the interaction to export
N_BEFORE: int = 3
N_AFTER: int = 3
# Temporal spacing for pre/post frames (seconds)
PRE_INTERVAL: float = 0.3
POST_INTERVAL: float = 0.3

# ---------------------------------------------------------------------------
#  INTERNAL HELPERS
# ---------------------------------------------------------------------------


def _collect_interaction_times(touches: Sequence[Dict[str, float]]) -> List[float]:
    """Return a list with *interaction* timestamps derived from *touches*.

    We treat every sequence that ends with an "up" event and is followed by at
    least ``TOUCH_COOLDOWN`` seconds of inactivity as *one* user interaction.
    The timestamp returned is that of the corresponding "up" event.
    """
    if not touches:
        return []

    interactions: List[float] = []
    n = len(touches)
    for i, ev in enumerate(touches):
        if ev.get("state") != "up":
            continue
        t_up: float = float(ev["time"])
        next_t = float(touches[i + 1]["time"]) if i + 1 < n else None
        if next_t is None or (next_t - t_up) > TOUCH_COOLDOWN:
            interactions.append(t_up)
    return interactions


def _schedule_frames(
    fps: float,
    n_frames: int,
    duration: float,
    interactions: Sequence[float],
) -> Dict[int, str]:
    """Compose a mapping *frame_index ➜ filename* for all frames to export."""
    scheduled: Dict[int, str] = {}

    # 1) Generic timeline thumbnails ----------------------------------------
    t: float = 0.0
    while t < duration:
        idx = int(round(t * fps))
        if idx < n_frames:
            scheduled.setdefault(idx, f"{t:.3f}_time.png")
        t += FRAME_INTERVAL

    # 2) Interaction-centric thumbnails -------------------------------------
    for t_int in interactions:
        # Central frame (touch): choose the *last* frame *before* the touch
        # event occurred.  Using ``floor`` (implicit in ``int``) instead of
        # ``round`` guarantees that the selected frame timestamp is *≤* the
        # interaction timestamp.  We additionally clamp the index to the
        # valid range so that we *always* schedule a central touch frame,
        # even if the interaction log slightly out-runs the actual video
        # (e.g. when recording stopped earlier than the touch logger).

        idx_touch: int = int(t_int * fps)  # floor – latest frame *before* touch
        if idx_touch >= n_frames:
            idx_touch = n_frames - 1  # fall back to very last frame of video
        if idx_touch < 0:
            idx_touch = 0

        scheduled[idx_touch] = f"{t_int:.3f}_touch.png"

        # Preceding frames
        for k in range(1, N_BEFORE + 1):
            t_pre = max(0.0, t_int - k * PRE_INTERVAL)
            idx_pre = int(round(t_pre * fps))
            if 0 <= idx_pre < n_frames:
                scheduled.setdefault(idx_pre, f"{t_pre:.3f}_pre_touch.png")

        # Following frames
        for k in range(1, N_AFTER + 1):
            t_post = t_int + k * POST_INTERVAL
            if t_post > duration:
                break
            idx_post = int(round(t_post * fps))
            if idx_post < n_frames:
                scheduled.setdefault(idx_post, f"{t_post:.3f}_post_touch.png")

    return scheduled


# ---------------------------------------------------------------------------
#  CORE WORKFLOW
# ---------------------------------------------------------------------------


def process_directory(directory: str | Path) -> None:
    """Extract still frames for *directory* according to the strategy above.

    The folder must contain:
        • ``screen_recording.mp4``      – the raw recording
        • ``touch_events.log``          – processed touch-event log

    It will use ``screen_recording_overlay.mp4`` if present.

    A new sub-folder named ``frames`` will be (re)created inside *directory*
    containing all exported ``.png`` files.
    """

    dir_path: Path = Path(directory)

    video_raw: Path = dir_path / "screen_recording.mp4"
    video_overlay: Path = dir_path / "screen_recording_overlay.mp4"
    log_file: Path = dir_path / "touch_events.log"
    out_dir: Path = dir_path / "frames"

    if not video_raw.exists():
        raise FileNotFoundError(f"⚠️  {video_raw} not found.")
    if not log_file.exists():
        raise FileNotFoundError(f"⚠️  {log_file} not found.")

    has_overlay = video_overlay.exists()
    if not has_overlay:
        print("⚠️  Overlay video not found. Using raw video for all frames.")

    # Fresh output folder ----------------------------------------------------
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Analyse the video --------------------------------------------------
    cap_raw = cv2.VideoCapture(str(video_raw))
    if not cap_raw.isOpened():
        raise RuntimeError(f"❌ Could not open raw video: {video_raw}")
    
    video_for_overlay_frames = video_overlay if has_overlay else video_raw
    cap_overlay = cv2.VideoCapture(str(video_for_overlay_frames))
    if not cap_overlay.isOpened():
         # This case should ideally not happen if video_raw exists
        raise RuntimeError(f"❌ Could not open video for overlay frames: {video_for_overlay_frames}")

    fps: float = cap_raw.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:  # Fallback
        fps = 30.0
    n_frames: int = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    duration: float = n_frames / fps if fps > 0 else 0.0

    print(f"🎬  Video: {n_frames} frames @ {fps:.3f} fps – {duration:.2f} s")

    # --- Parse the touch log ------------------------------------------------
    print("🔍  Parsing touch-event log …")
    touches = parse_touch_log(log_file)
    interactions = _collect_interaction_times(touches)
    print(f"   ↳ {len(interactions):,} distinct user interactions found.")

    # --- Build export schedule ---------------------------------------------
    schedule = _schedule_frames(fps, n_frames, duration, interactions)
    total_to_save = len(schedule)
    print(f"🖼️  Scheduled {total_to_save} frames to export …")

    # --- Iterate & save -----------------------------------------------------
    bar = tqdm(total=n_frames, unit="frame", desc="Scanning video")
    frame_idx = 0
    saved = 0
    while True:
        ok_r, frame_r = cap_raw.read()
        if not ok_r:
            break

        frame_o = frame_r  # Default to raw
        if has_overlay:
            ok_o, frame_o_maybe = cap_overlay.read()
            if ok_o:
                frame_o = frame_o_maybe

        if frame_idx in schedule:
            filename = schedule[frame_idx]
            # Generic timeline frames (suffix '_time') come from the *raw* recording.
            use_raw: bool = filename.endswith("_time.png")
            frame_to_save = frame_r if use_raw else frame_o
            cv2.imwrite(str(out_dir / filename), frame_to_save)
            saved += 1
            if saved == total_to_save:
                # Collected everything — skip remaining decode work
                bar.update(n_frames - frame_idx)
                break

        frame_idx += 1
        bar.update()
    bar.close()
    cap_overlay.release()
    cap_raw.release()

    print(f"✅  Finished – {saved}/{total_to_save} frames written to: {out_dir}")


# ---------------------------------------------------------------------------
#  CLI ENTRY POINT (mirrors visualise.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import datetime

    base_data_dir: Path = Path("data")

    # 1) Ask for the game name ------------------------------------------------
    game_name: str = input("🎮  Enter the game name: ").strip()
    while not game_name:
        game_name = input("Please enter a non-empty game name: ").strip()
    game_dir: Path = base_data_dir / sanitize_path_component(game_name)
    if not game_dir.exists():
        print(f"❌  Game directory not found: {game_dir}", file=sys.stderr)
        sys.exit(1)

    # 2) Enumerate sessions ---------------------------------------------------
    sessions: List[Path] = list_sessions(game_dir)
    if not sessions:
        print(f"⚠️  No recording sessions found in {game_dir}", file=sys.stderr)
        sys.exit(1)

    print("\n📂  Available recording sessions:")
    for idx, sess in enumerate(sessions, start=1):
        ts: str = datetime.fromtimestamp(sess.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"  {idx:>2}. {sess.name} ({ts})")

    # 3) Let the user choose --------------------------------------------------
    choice_str: str = input(
        f"\nSelect a session [1-{len(sessions)}] (default 1): "
    ).strip()
    if not choice_str:
        choice_idx = 1
    else:
        try:
            choice_idx = int(choice_str)
            if not (1 <= choice_idx <= len(sessions)):
                raise ValueError
        except ValueError:
            print("❌  Invalid selection – exiting.", file=sys.stderr)
            sys.exit(1)

    chosen_dir: Path = sessions[choice_idx - 1]
    print(f"\n▶️  Processing: {chosen_dir}\n")
    process_directory(chosen_dir)
