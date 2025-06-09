"""
Quick-start helper for overlaying Android *touch_events.log* data
onto an existing screen-recording (``screen_recording.mp4``).

The original recording produced by *android_streamer.py* uses real
wall-clock timestamps, therefore the **container's FPS metadata may be
wrong** (e.g. the file might say 30 fps even though it actually varies).
To keep touch overlays in perfect sync we first determine the *true*
average FPS from the video's timestamps and **always use that value**.
"""

import bisect  # NEW
from pathlib import Path

import cv2, numpy as np, os, re, subprocess
from tqdm.auto import tqdm  # nicer in notebooks

# ------------- USER-CONFIGURABLE ------------- #
VIDEO_FILE = Path("screen_recording.mp4")
LOG_FILE = Path("touch_events.log")  # your "@touch_events.log"
OUTPUT_FILE = Path("screen_recording_overlay.mp4")
TRAIL_SECS = 2  # how long the fading trail lasts
# --------------------------------------------- #

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()  # overwrite old result


# ---------------------------------------------------------------
#  UTILITY ── Try several codec / container combos until one works
# ---------------------------------------------------------------
def _open_video_writer(out_path: Path, w: int, h: int, fps: float):
    """
    Returns (VideoWriter, real_output_path).

    We try common codec / container pairs because OpenCV-FFmpeg builds
    differ wildly across platforms.  If MP4 fails we fall back to AVI,
    guaranteeing that the resulting file can be opened by standard
    players.
    """
    candidates = (
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    )

    for fourcc_name, suffix in candidates:
        target = out_path.with_suffix(suffix)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        vw = cv2.VideoWriter(str(target), fourcc, fps, (w, h))
        if vw.isOpened():
            return vw, target
    raise RuntimeError(
        "❌  Could not open a VideoWriter for any known codec.\n"
        "    Please ensure your Python/OpenCV installation has FFmpeg "
        "support (or install a pre-built wheel that includes it)."
    )


# ---------------------------------------------------------------
#  UTILITY ── Robust FPS detection
# ---------------------------------------------------------------
def _probe_video_fps(cap: cv2.VideoCapture, path: Path) -> float:
    """
    Determine the most trustworthy FPS for *path*.

    Order of preference:
    1. ffprobe's calculated average_frame_rate (if FFmpeg present)
    2. Derive from (nb_frames ÷ duration) via ffprobe
    3. Compute from timestamps of first/last frame via OpenCV
    4. Fall back to CAP_PROP_FPS or finally 30.0
    """
    # --- 1) Try ffprobe avg_frame_rate --------------------------
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,duration,nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        out = subprocess.check_output(cmd, text=True)
        vals = [l.strip() for l in out.splitlines()]
        if vals:
            afr_txt = vals[0]  # avg_frame_rate like "24000/1001"
            if "/" in afr_txt:
                num, den = map(float, afr_txt.split("/"))
                if den:
                    fps = num / den
                    if fps > 1:  # sanity check
                        return fps
            else:
                fps = float(afr_txt)
                if fps > 1:
                    return fps
            # optional: nb_frames/duration fallback
            if len(vals) >= 3 and vals[1] and vals[2]:
                duration = float(vals[1])
                nb_frames = float(vals[2])
                if duration > 0:
                    fps = nb_frames / duration
                    if fps > 1:
                        return fps
    except Exception:
        pass  # ffprobe unavailable or failed

    # --- 2) OpenCV timestamps (first vs. last frame) ------------
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames > 1:
            pos_old = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames - 1)
            t_end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0 if pos_old < 0 else pos_old)
            if t_end > 0:
                return (n_frames - 1) / t_end
    except Exception:
        pass

    # --- 3) Metadata reported by OpenCV -------------------------
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    if meta_fps and meta_fps > 1:
        return meta_fps

    # --- 4) Final fallback --------------------------------------
    return 30.0  # assume something sensible


# -----------------------------------------------------------------
#  STEP 1 ── Parse *touch_events.log* into chronological events
# -----------------------------------------------------------------
_TS_RE = re.compile(r"\[\s*(\d+\.\d+)\]")  # extracts the timestamp


def parse_touch_log(path: Path, *, max_time: float = 5_000.0):
    """
    Robustly parse an Android *touch_events.log* file.

    • Lines that cannot be parsed are skipped.
    • Events beyond `max_time` seconds are ignored (outliers).
    """
    touches, active = [], {}  # active[id] = (x, y)

    # Helper -------------------------------------------------------
    def _safe_int(txt: str, base: int = 10):
        try:
            return int(txt, base)
        except ValueError:
            return None

    # -------------------------------------------------------------

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            m_time = _TS_RE.match(ln)
            if not m_time:
                continue
            try:
                t = float(m_time.group(1))
            except ValueError:
                continue
            if t > max_time:
                continue

            ln2 = ln[m_time.end() :]  # remainder of the line

            # New / lifted finger?
            if "ABS_MT_TRACKING_ID" in ln2:
                id_txt = ln2.split()[-1]
                if id_txt.lower() == "ffffffff":  # lift
                    for ev_id, (x, y) in list(active.items()):
                        touches.append(
                            {"time": t, "id": ev_id, "x": x, "y": y, "state": "up"}
                        )
                    active.clear()
                else:
                    ev_id = _safe_int(id_txt, 16)
                    if ev_id is not None:
                        active[ev_id] = (None, None)  # will update soon

            # Position updates
            elif "ABS_MT_POSITION_X" in ln2:
                val = _safe_int(ln2.split()[-1])
                if val is None:
                    continue
                for ev_id in active:
                    active[ev_id] = (val, active[ev_id][1])

            elif "ABS_MT_POSITION_Y" in ln2:
                val = _safe_int(ln2.split()[-1])
                if val is None:
                    continue
                for ev_id in active:
                    active[ev_id] = (active[ev_id][0], val)

            # Sync report → commit all currently-known positions
            elif "SYN_REPORT" in ln2:
                for ev_id, (x, y) in active.items():
                    if x is not None and y is not None:
                        touches.append(
                            {"time": t, "id": ev_id, "x": x, "y": y, "state": "move"}
                        )
    return touches


# ---------------------------------------------------------------
#  UTILITY ── Collect per-frame timestamps (handles VFR sources)
# ---------------------------------------------------------------
def _collect_frame_timestamps(path: Path):
    """
    Return a list with the presentation-timestamp (in seconds) of every
    decoded frame in *path*.
    We perform a quick pass over the file using OpenCV.  Even though this
    decodes the video once, it is still faster and simpler than trying to
    reverse-engineer the container's time-base.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"❌  Could not open video: {path}")

    pts = []
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        # CAP_PROP_POS_MSEC returns the timestamp *after* grabbing/decoding
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    cap.release()
    return pts


# -----------------------------------------------------------------
#  VFR-aware helper ── Convert raw "touches" into per-frame overlays
# -----------------------------------------------------------------
def precompute_touch_data_vfr(touches, frame_times):
    n_frames = len(frame_times)
    per_frame = [[] for _ in range(n_frames)]
    trail_secs = TRAIL_SECS

    # ---- Bucket events by target frame using bisect ----------------
    buckets = {}
    for ev in touches:
        fi = bisect.bisect_left(frame_times, ev["time"])
        if 0 <= fi < n_frames:
            buckets.setdefault(fi, []).append(ev)

    # ---- Build trails frame-by-frame --------------------------------
    trails = {}  # id → list[(t,(x,y))]
    for fi in range(n_frames):
        current_t = frame_times[fi]

        if fi in buckets:  # new events this frame?
            for ev in buckets[fi]:
                if ev["state"] == "up":
                    trails.pop(ev["id"], None)
                else:
                    trails.setdefault(ev["id"], []).append(
                        (ev["time"], (ev["x"], ev["y"]))
                    )

        # Trim old trail points
        for ev_id in list(trails):
            trails[ev_id] = [p for p in trails[ev_id] if current_t - p[0] <= trail_secs]
            if not trails[ev_id]:
                trails.pop(ev_id)

        # Prepare drawing instructions for this frame
        for idx, pts in trails.items():
            ordered_pts = [p[1] for p in sorted(pts, key=lambda x: x[0], reverse=True)]
            per_frame[fi].append(
                {"touch_idx": idx, "pos": ordered_pts[0], "trail": ordered_pts}
            )

    return per_frame


# -----------------------------------------------------------------
#  STEP 3 ── Render / overlay everything onto the video
# -----------------------------------------------------------------
def overlay_touches_on_video(video_path: Path, output_path: Path, touches):
    print(f"🔍  Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"❌  Could not open video: {video_path}")

    # We will still compute a *nominal* FPS for writing/output purposes but
    # the actual event-to-frame mapping uses real PTS (frame_times).
    fps_meta = _probe_video_fps(cap, video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_times = _collect_frame_timestamps(video_path)

    # Derive an output FPS as the *median* interval – this is robust even for VFR
    if len(frame_times) >= 2:
        median_dt = float(np.median(np.diff(frame_times)))
        fps_out = 1.0 / median_dt if median_dt > 0 else fps_meta
    else:
        fps_out = fps_meta

    print(
        f"🎬  Video: {W}×{H} | meta {fps_meta:.3f} fps, median {fps_out:.3f} fps | {N} frames"
    )

    frame_touches = precompute_touch_data_vfr(touches, frame_times)
    out_writer, real_out = _open_video_writer(output_path, W, H, fps_out)

    colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 165, 255),
        (128, 0, 128),
    ]  # BGR

    bar = tqdm(total=N, unit="frame", desc="Encoding")
    for fi in range(N):
        ok, frame = cap.read()
        if not ok:
            break
        for item in frame_touches[fi]:
            idx, pos, trail = item["touch_idx"], item["pos"], item["trail"]
            col = colors[idx % len(colors)]

            # Fading trail
            for i in range(1, len(trail)):
                a = i / len(trail)
                col2 = tuple(int(c * a) for c in col)
                cv2.line(
                    frame, trail[i - 1], trail[i], col2, 9
                )  # Changed thickness to 9

            # Current point
            cv2.circle(frame, pos, 40, col, 3)
            cv2.circle(frame, pos, 30, (255, 255, 255), -1)
            cv2.circle(frame, pos, 30, col, 2)
            # cv2.putText(
            #     frame,
            #     f"T{idx}",
            #     (pos[0] + 25, pos[1] - 25),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     col,
            #     2,
            # )

        out_writer.write(frame)
        bar.update()
    bar.close()

    cap.release()
    out_writer.release()
    print(f"✅  Finished – output saved to: {real_out}")


if __name__ == "__main__":
    if OUTPUT_FILE.exists():
        os.remove(OUTPUT_FILE)  # Remove the previous overlay file
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(f"⚠️  {VIDEO_FILE} not found.")
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"⚠️  {LOG_FILE} not found.")

    print("🔍  Parsing touch-event log …")
    touch_data = parse_touch_log(LOG_FILE)
    print(f"   ↳ {len(touch_data):,} events parsed.")

    overlay_touches_on_video(VIDEO_FILE, OUTPUT_FILE, touch_data)
