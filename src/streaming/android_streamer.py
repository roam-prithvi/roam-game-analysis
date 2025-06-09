import difflib  # For fuzzy name suggestions
import fcntl
import os
import re
import shlex
import signal
import subprocess
import sys
import threading  # New: for monitoring FFmpeg stderr asynchronously
import time
from typing import Any, Callable, IO, Optional, Tuple  # ➕ typing imports

# --- Configuration ---
# TODO: if adb not in PATH, try to find it, then download it, then add it to PATH and set ADB_PATH to the full path
# if adb in path, set ADB_PATH to "adb"
ADB_PATH = "adb"  # Or provide a full path if 'adb' is not in your system's PATH
EVENT_LOG_FILE = "touch_events.log"  # The file where raw touch events will be saved
VIDEO_OUTPUT_FILE = "screen_recording.mp4"  # The file where video will be saved
VIDEO_ERROR_LOG_FILE = (
    "video_error.log"  # File to capture FFmpeg/ADB errors and diagnostics
)
EVENT_FUDGE = 0.20  # seconds, TODO increase -> touch events appear later

# --- Global variable to hold our running processes ---
processes = []
event_log_file = None  # Global to be accessible by signal_handler
coordinate_translator = None  # Global translator function
video_log_file = None  # Log file handle for FFmpeg/ADB diagnostics

# --- Timing synchronization variables ---
stream_start_time = None  # When video recording subprocess is launched
first_tap_time = None  # When first touch event is processed
first_video_frame_time = None  # When video file is first modified (first frame written)
video_frame_detected = False  # Flag to track if first video frame has been detected

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


def prepare_output_paths() -> str:
    """Prompt for the current game's name and prepare the output folder tree.

    The layout produced is:
        data/<game_name>/<DD-MM-YY_at_HH.MM.SS>/
            ├── touch_events.log
            ├── screen_recording.mp4
            └── video_error.log

    Returns the absolute path to the freshly-created session directory.
    """
    game_name = input(
        "🎮 Enter the name of the game you're currently playing: "
    ).strip()
    while not game_name:
        game_name = input("Please enter a non-empty game name: ").strip()

    safe_game = sanitize_path_component(game_name)

    # --- "Maybe you meant" logic -----------------------------------------
    # Look for a close existing folder name using difflib's SequenceMatcher.
    existing_games = []
    if os.path.isdir("data"):
        existing_games = [
            d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))
        ]

    if existing_games:
        close_matches = difflib.get_close_matches(
            safe_game, existing_games, n=1, cutoff=0.6
        )
        if close_matches and close_matches[0] != safe_game:
            suggested = close_matches[0]
            resp = input(
                f"🤔 Did you mean '{suggested}'? Press Enter to accept, or type 'n' then press Enter to reject: "
            )
            # Accept suggestion if the user just hits Enter (empty string)
            if resp == "":
                safe_game = suggested

    base_dir = os.path.join("data", safe_game)
    os.makedirs(base_dir, exist_ok=True)

    timestamp = time.strftime("%d-%m-%y_at_%H.%M.%S")
    session_dir = os.path.join(base_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)

    # Update global output paths so the rest of the script can remain unchanged.
    global EVENT_LOG_FILE, VIDEO_OUTPUT_FILE, VIDEO_ERROR_LOG_FILE
    EVENT_LOG_FILE = os.path.join(session_dir, "touch_events.log")
    VIDEO_OUTPUT_FILE = os.path.join(session_dir, "screen_recording.mp4")
    VIDEO_ERROR_LOG_FILE = os.path.join(session_dir, "video_error.log")

    print(f"📁 Saving all outputs to: {session_dir}")
    return session_dir


def find_touchscreen_device() -> str:
    """
    Finds the event device path for the main touchscreen.

    Connects to the device via ADB and parses the output of 'getevent -lp'
    to find a device that has 'touchscreen' in its name and supports
    multi-touch X and Y absolute events.
    """
    print("🔎 Searching for touchscreen device...")
    try:
        # Run the adb command to list input devices
        command = f"{ADB_PATH} shell getevent -lp"
        result = subprocess.run(
            shlex.split(command), capture_output=True, text=True, check=True
        )

        # Split the output into blocks for each device
        # A new device block starts with "add device"
        device_blocks = result.stdout.strip().split("add device")

        for block in device_blocks:
            if not block.strip():
                continue

            # Check for the required characteristics
            is_touchscreen = "touchscreen" in block.lower()
            has_abs_mt_x = "ABS_MT_POSITION_X" in block
            has_abs_mt_y = "ABS_MT_POSITION_Y" in block

            if is_touchscreen and has_abs_mt_x and has_abs_mt_y:
                # Extract the device path from the first line of the block
                # Format is typically: " 7: /dev/input/event11"
                lines = block.strip().split("\n")
                if lines:
                    first_line = lines[0].strip()
                    # Look for the pattern "number: /dev/input/eventX"
                    match = re.search(r"\d+:\s*(/dev/input/event\d+)", first_line)
                    if match:
                        device_path = match.group(1)
                        print(f"✅ Found touchscreen: {device_path}")
                        return device_path

    except FileNotFoundError:
        print(
            f"❌ Error: '{ADB_PATH}' command not found. Is ADB installed and in your PATH?"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"❌ Error executing ADB command: {e}\nIs your device connected and authorized?"
        )
        sys.exit(1)

    print("❌ Error: Could not find a suitable touchscreen device.")
    sys.exit(1)


def get_screen_size() -> Tuple[Optional[int], Optional[int]]:
    """Gets the physical screen size (resolution) of the device."""
    print("📏 Getting screen size...")
    try:
        command = f"{ADB_PATH} shell wm size"
        result = subprocess.run(
            shlex.split(command), capture_output=True, text=True, check=True
        )

        # Find the line with "Physical size:"
        match = re.search(r"Physical size: (\d+)x(\d+)", result.stdout)
        if match:
            width, height = match.group(1), match.group(2)
            print(f"✅ Screen size: {width}x{height}")
            return int(width), int(height)

    except Exception as e:
        print(f"❌ Error getting screen size: {e}")

    print("⚠️ Could not determine screen size, continuing without it.")
    return None, None


def get_input_device_ranges(
    device_path: str,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Gets the coordinate ranges for ABS_MT_POSITION_X and ABS_MT_POSITION_Y
    from the input device to enable coordinate translation.

    Returns:
        tuple: (x_min, x_max, y_min, y_max) or (None, None, None, None) if not found
    """
    print(f"📐 Getting coordinate ranges for {device_path}...")
    try:
        command = f"{ADB_PATH} shell getevent -lp {device_path}"
        result = subprocess.run(
            shlex.split(command), capture_output=True, text=True, check=True
        )

        x_min = x_max = y_min = y_max = None

        for line in result.stdout.split("\n"):
            # Look for ABS_MT_POSITION_X line with min/max values
            # Format: "    ABS_MT_POSITION_X    : value 0, min 0, max 3119, fuzz 0, flat 0, resolution 0"
            if "ABS_MT_POSITION_X" in line:
                match = re.search(r"min\s+(\d+),\s*max\s+(\d+)", line)
                if match:
                    x_min, x_max = int(match.group(1)), int(match.group(2))

            elif "ABS_MT_POSITION_Y" in line:
                match = re.search(r"min\s+(\d+),\s*max\s+(\d+)", line)
                if match:
                    y_min, y_max = int(match.group(1)), int(match.group(2))

        if all(v is not None for v in [x_min, x_max, y_min, y_max]):
            print(f"✅ Coordinate ranges - X: {x_min}-{x_max}, Y: {y_min}-{y_max}")
            return x_min, x_max, y_min, y_max
        else:
            print("⚠️ Could not find coordinate ranges in device info")
            return None, None, None, None

    except Exception as e:
        print(f"❌ Error getting coordinate ranges: {e}")
        return None, None, None, None


def create_coordinate_translator(
    x_min: Optional[int],
    x_max: Optional[int],
    y_min: Optional[int],
    y_max: Optional[int],
    screen_width: Optional[int],
    screen_height: Optional[int],
) -> Optional[Callable[[int, int], Tuple[int, int]]]:
    """
    Creates a function that translates raw touch coordinates to pixel coordinates.

    Args:
        x_min, x_max: Raw coordinate range for X axis
        y_min, y_max: Raw coordinate range for Y axis
        screen_width, screen_height: Screen resolution in pixels

    Returns:
        function: A translator function that takes (raw_x, raw_y) and returns (pixel_x, pixel_y)
    """
    if any(
        v is None for v in [x_min, x_max, y_min, y_max, screen_width, screen_height]
    ):
        print("⚠️ Missing coordinate info, translation will not be available")
        return None

    def translate(raw_x: int, raw_y: int) -> Tuple[int, int]:
        # Linear interpolation from raw range to pixel range
        pixel_x = int((raw_x - x_min) * screen_width / (x_max - x_min))
        pixel_y = int((raw_y - y_min) * screen_height / (y_max - y_min))
        return pixel_x, pixel_y

    print(
        f"✅ Coordinate translator created: {x_max-x_min+1}x{y_max-y_min+1} -> {screen_width}x{screen_height}"
    )
    return translate


def signal_handler(sig: int, frame: Any) -> None:
    """Handles Ctrl+C to gracefully shut down all subprocesses."""
    print("\n🛑 Ctrl+C detected! Shutting down streams...")
    # Wait for a moment for processes to terminate
    time.sleep(1)
    for p in processes:
        try:
            p.terminate()  # Send termination signal
        except ProcessLookupError:
            pass  # Process already finished

    time.sleep(2)

    for p in processes:
        if p.poll() is None:  # If process is still running
            p.kill()  # Force kill it

    global event_log_file, video_log_file
    if event_log_file and not event_log_file.closed:
        event_log_file.close()
        print("✅ Event log file closed.")
        # 🔧 Post-process the log to fix split lines & convert leftovers
        cleanup_log()

    # Close video error log if open
    if video_log_file and not video_log_file.closed:
        video_log_file.close()
        print("✅ Video error log file closed.")

    print("✅ All streams stopped.")
    sys.exit(0)


def check_first_video_frame() -> bool:
    """
    Check if the first video frame has been written by monitoring file size.
    Returns True if first frame detected, False otherwise.
    """
    global first_video_frame_time, video_frame_detected

    if video_frame_detected:
        return True

    try:
        current_size = os.path.getsize(VIDEO_OUTPUT_FILE)
        if current_size > 0:
            first_video_frame_time = time.time()
            video_frame_detected = True
            print(f"🎬 First video frame detected at {first_video_frame_time:.6f}")
            return True
    except (FileNotFoundError, OSError):
        pass  # File doesn't exist yet or can't be accessed

    return False


def monitor_ffmpeg_stderr(stderr_pipe: IO[bytes], log_handle: IO[str]) -> None:
    """Monitors FFmpeg stderr, writes it to *log_handle* and grabs the first-frame epoch.

    The function looks for a line that contains "start: <epoch>" which is printed by
    FFmpeg while probing the H.264 input.  That number is the host wall-clock time
    (seconds since epoch) at which FFmpeg stamped the *very first* video frame.
    As soon as we find it we use it as a high-precision substitute for
    ``first_video_frame_time``.
    """
    global first_video_frame_time, video_frame_detected

    start_re = re.compile(r"start:\s*([0-9]+\.[0-9]+)")

    for raw in iter(stderr_pipe.readline, b""):
        line = raw.decode(errors="replace")
        # Always replicate the line to the on-disk log
        log_handle.write(line)

        if not video_frame_detected:
            m = start_re.search(line)
            if m:
                ts = float(m.group(1))
                # Ignore the value if it is obviously "0" or not an epoch
                if ts > 2_000_000_000:  # rough cut-off: year ≈ 2033
                    first_video_frame_time = ts
                    video_frame_detected = True

        log_handle.flush()

    stderr_pipe.close()


def process_event_line(line: str) -> Optional[str]:
    """
    Process a single event line, converting coordinates to pixels and adjusting timestamp.

    Args:
        line: Raw event line from getevent

    Returns:
        Processed line string or None if line should be skipped
    """
    global coordinate_translator, first_tap_time, first_video_frame_time, video_frame_detected

    if not line.strip():
        return None

    # Parse the line to extract timestamp and event info
    match = re.match(r"\[\s*(\d+\.\d+)\s*\]\s+(\S+)\s+(\S+)\s+(.+)", line)
    if not match:
        return line  # Return original line if parsing fails

    android_timestamp_str, ev_type, event_code, value_str = match.groups()

    # Calculate video-relative timestamp
    current_time = time.time()
    if video_frame_detected and first_video_frame_time is not None:
        # Use video frame time as reference (time 0)
        video_timestamp = current_time - first_video_frame_time + EVENT_FUDGE
    else:
        # Fallback: use first tap time as temporary reference
        video_timestamp = current_time - first_tap_time if first_tap_time else 0.0

    new_ts_part = f"[{video_timestamp:13.6f}]"

    # Handle coordinate translation for touch position and size events
    if (
        event_code
        in [
            "ABS_MT_POSITION_X",
            "ABS_MT_POSITION_Y",
            "ABS_MT_TOUCH_MAJOR",
            "ABS_MT_TOUCH_MINOR",
        ]
        and coordinate_translator
    ):
        # Convert hex value to decimal
        value_str = value_str.strip()
        try:
            if value_str.startswith("0x"):
                raw_value = int(value_str, 16)
            elif value_str.startswith("0") and len(value_str) > 1:
                raw_value = int(value_str, 16)
            else:
                raw_value = int(value_str)
        except ValueError:
            # If conversion fails, keep original
            return f"{new_ts_part} {ev_type:<12} {event_code:<20} {value_str}\n"

        # For coordinate and touch size events, convert to pixels
        if event_code == "ABS_MT_POSITION_X":
            # Convert X coordinate (assuming Y=0 for single coordinate conversion)
            if coordinate_translator:
                pixel_x, _ = coordinate_translator(raw_value, 0)
                return f"{new_ts_part} {ev_type:<12} {event_code:<20} {pixel_x:>12}\n"
        elif event_code == "ABS_MT_POSITION_Y":
            # Convert Y coordinate (assuming X=0 for single coordinate conversion)
            if coordinate_translator:
                _, pixel_y = coordinate_translator(0, raw_value)
                return f"{new_ts_part} {ev_type:<12} {event_code:<20} {pixel_y:>12}\n"
        elif event_code == "ABS_MT_TOUCH_MAJOR":
            # Convert major axis size - use X axis scaling for consistency
            if coordinate_translator:
                pixel_major, _ = coordinate_translator(raw_value, 0)
                return (
                    f"{new_ts_part} {ev_type:<12} {event_code:<20} {pixel_major:>12}\n"
                )
        elif event_code == "ABS_MT_TOUCH_MINOR":
            # Convert minor axis size - use Y axis scaling for consistency
            if coordinate_translator:
                _, pixel_minor = coordinate_translator(0, raw_value)
                return (
                    f"{new_ts_part} {ev_type:<12} {event_code:<20} {pixel_minor:>12}\n"
                )

    # For non-coordinate events, just adjust timestamp and format
    return f"{new_ts_part} {ev_type:<12} {event_code:<20} {value_str}\n"


def cleanup_log() -> None:
    """
    Post-processes the touch-event log so that:
    1. Every record begins with a "[" character (after optional whitespace).
       If a line does not begin with "[" it is assumed to be a continuation of
       the *previous* line and is therefore concatenated to that line.
    2. Any still-raw hexadecimal/decimal touch values are converted to pixel
       coordinates or sizes using the already-prepared *coordinate_translator*.

       This routine must not rely on the high-resolution wall-clock logic used
       elsewhere, therefore it performs its own lightweight parsing and value
       conversion.
    """

    # We cannot do anything if the file does not exist (e.g. recording aborted)
    if not os.path.exists(EVENT_LOG_FILE):
        print(f"⚠️  Log file '{EVENT_LOG_FILE}' not found – skipping cleanup.")
        return

    try:
        # One-pass clean-up: glue continuation lines *and* convert their values
        with open(EVENT_LOG_FILE, "r") as fh:
            raw_lines = fh.readlines()

        # Mapping to know which event codes are coordinates/sizes
        coord_codes = {
            "ABS_MT_POSITION_X": True,  # True  => X-axis
            "ABS_MT_POSITION_Y": False,  # False => Y-axis
            "ABS_MT_TOUCH_MAJOR": True,
            "ABS_MT_TOUCH_MINOR": False,
        }

        cleaned_lines: list[str] = []

        for line in raw_lines:
            if re.match(r"\s*\[", line):
                # Proper line → just stash it (untouched)
                cleaned_lines.append(line)
                continue

            # Otherwise this is a continuation containing the value portion.
            if not cleaned_lines:
                # No previous line to attach to; skip
                continue

            prev = cleaned_lines.pop().rstrip("\n")
            glued = prev + " " + line.strip()  # combine & strip extra NLs/spaces

            # Try to convert the newly-added value *only for this glued line*.
            m = re.match(r"(\[\s*[^]]+\]\s+)(\S+)\s+(\S+)\s+(\S+)", glued)
            if m and coordinate_translator is not None and m.group(3) in coord_codes:
                ts_part, ev_type, event_code, value_str = m.groups()

                # Convert numeric string (hex or decimal) → int
                try:
                    raw_val = (
                        int(value_str, 16)
                        if value_str.startswith("0")
                        else int(value_str)
                    )
                except ValueError:
                    cleaned_lines.append(glued + "\n")
                    continue

                if coord_codes[event_code]:
                    px, _ = coordinate_translator(raw_val, 0)
                    new_val = f"{px:>12}"
                else:
                    _, py = coordinate_translator(0, raw_val)
                    new_val = f"{py:>12}"

                glued = f"{ts_part}{ev_type:<12} {event_code:<20} {new_val}"

            cleaned_lines.append(glued + "\n")

        # --- Second pass: smooth outlier timestamps ---------------------------------

        def extract_ts(line: str) -> Optional[float]:
            """Return float timestamp inside leading brackets or None."""
            m_ts = re.match(r"\s*\[\s*([0-9]+\.[0-9]+)\s*\]", line)
            return float(m_ts.group(1)) if m_ts else None

        fixed_lines = cleaned_lines[:]
        for i in range(1, len(cleaned_lines) - 1):
            prev_ts = extract_ts(cleaned_lines[i - 1])
            cur_ts = extract_ts(cleaned_lines[i])
            next_ts = extract_ts(cleaned_lines[i + 1])
            if None in (prev_ts, cur_ts, next_ts):
                print("⚠️ Warning: Log line without timestamp found")
                continue

            if cur_ts > 100 * prev_ts:
                new_ts_val = (prev_ts + next_ts) / 2.0
                # Re-format with same 13.6f width used elsewhere
                new_ts_str = f"[{new_ts_val:13.6f}]"
                # Substitute only the bracketed timestamp portion
                fixed_lines[i] = re.sub(
                    r"^\s*\[[^]]+\]", new_ts_str, cleaned_lines[i], count=1
                )

        # Overwrite the original log with the cleaned & smoothed content
        with open(EVENT_LOG_FILE, "w") as fh:
            fh.writelines(fixed_lines)

        print("🧹 Log cleanup completed – saved to", EVENT_LOG_FILE)
    except Exception as e:
        print(f"❌ Error during log cleanup: {e}")


def main() -> None:
    """Main function to set up and run the concurrent streams."""
    global event_log_file, coordinate_translator, stream_start_time, first_tap_time, first_video_frame_time, video_frame_detected, video_log_file
    # Set up the Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)

    # 🆕 Prepare game-specific output paths ------------------------------------
    prepare_output_paths()

    # 1. Find the device and get device info
    touch_device = find_touchscreen_device()
    screen_width, screen_height = get_screen_size()

    # 2. Get coordinate ranges for translation
    x_min, x_max, y_min, y_max = get_input_device_ranges(touch_device)
    coordinate_translator = create_coordinate_translator(
        x_min, x_max, y_min, y_max, screen_width, screen_height
    )

    # 3. Start event stream but wait for first touch to start video
    event_command = shlex.split(f"{ADB_PATH} shell getevent -lt {touch_device}")
    event_process = subprocess.Popen(
        event_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    processes.append(event_process)
    print(f"🕒 Waiting for the first touch event to start recording...")

    video_process = None

    # Block and wait for the first event to arrive
    for line in iter(event_process.stdout.readline, ""):
        if not line.strip():
            continue

        # Capture first tap time
        first_tap_time = time.time()
        print(
            f"👆 First touch detected at {first_tap_time:.6f}! Starting video recording..."
        )

        # --- Video Stream ---
        # Build video command with proper resolution if available
        ffmpeg_base = (
            "ffmpeg -y -use_wallclock_as_timestamps 1 -fflags +genpts "
            "-f h264 -i - -c copy"
        )

        # Quote the output path in case it contains spaces or special chars.
        quoted_output_file = shlex.quote(VIDEO_OUTPUT_FILE)

        if screen_width and screen_height:
            video_command = (
                f"{ADB_PATH} shell screenrecord --size {screen_width}x{screen_height} "
                f"--output-format=h264 - | {ffmpeg_base} {quoted_output_file}"
            )
            print(f"🎥 Recording at {screen_width}x{screen_height} resolution")
        else:
            video_command = (
                f"{ADB_PATH} shell screenrecord --output-format=h264 - "
                f"| {ffmpeg_base} {quoted_output_file}"
            )
            print("🎥 Recording at default resolution")

        # Prepare video error log
        video_log_file = open(VIDEO_ERROR_LOG_FILE, "a")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        video_log_file.write(
            f"\n[{timestamp}] Launching video command: {video_command}\n"
        )
        video_log_file.flush()

        # Capture stream start time when video recording begins
        stream_start_time = time.time()
        video_process = subprocess.Popen(
            video_command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        processes.append(video_process)
        print(f"🔴 REC: Video recording started at {stream_start_time:.6f}")
        print(f"🔴 REC: Video is being recorded to '{VIDEO_OUTPUT_FILE}'")
        print(
            f"🔴 REC: Touch events are being recorded to '{EVENT_LOG_FILE}' (with pixel coordinates and video-relative timestamps)"
        )

        event_log_file = open(EVENT_LOG_FILE, "w")

        # Process the first line
        processed_line = process_event_line(line)
        if processed_line:
            event_log_file.write(processed_line)
        event_log_file.flush()

        # Start a background thread to tee FFmpeg stderr to the log file and
        # extract high-precision first-frame timing information.
        threading.Thread(
            target=monitor_ffmpeg_stderr,
            args=(video_process.stderr, video_log_file),
            daemon=True,
        ).start()

        break  # Exit loop after handling the first event

    if first_tap_time is None:
        print("ℹ️ No touch events received. The 'getevent' process may have exited.")
        stderr_output = event_process.stderr.read()
        if stderr_output:
            print("--- getevent stderr ---")
            print(stderr_output)
            print("-----------------------")
        signal_handler(None, None)
        return

    # Set event stream to non-blocking for the main loop
    fd = event_process.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    print("\n=========================================================")
    print(">>> Analysis session is running. Press Ctrl+C to stop. <<<")
    print("=========================================================")

    # 4. Main event processing loop with video frame monitoring
    event_counter = 0
    timeout_start = time.time()

    while True:
        if video_process.poll() is not None:
            print("ℹ️ Video recording process ended.")
            signal_handler(None, None)

        # Check for first video frame every 10 events or if not detected yet
        if event_counter % 10 == 0 or not video_frame_detected:
            check_first_video_frame()

            # Handle timeout for video frame detection
            if not video_frame_detected and (time.time() - timeout_start) > 20:
                print("⚠️ Timeout: First video frame not detected within 20 seconds")
                print("⚠️ Continuing with timestamps relative to first tap...")
                # Set video frame time to first tap time as fallback
                first_video_frame_time = first_tap_time
                video_frame_detected = True

        # Process any available touch events non-blockingly
        while True:
            try:
                line = event_process.stdout.readline()
                if not line:  # Empty string means EOF
                    # If process ended, shutdown everything
                    if event_process.poll() is not None:
                        print("ℹ️ Touch event stream ended.")
                        signal_handler(None, None)
                    break  # No more data for now

                if not line.strip():
                    continue

                # Process the line with video-relative timing
                processed_line = process_event_line(line)
                if processed_line:
                    event_log_file.write(processed_line)
                    event_counter += 1

            except BlockingIOError:
                break  # No more data available at the moment
            except ValueError:
                # Partial line read can cause float conversion errors, ignore for now
                pass

        event_log_file.flush()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
