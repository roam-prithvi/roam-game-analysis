# THIS FILE IS THE ENTIRETY OF THE DATA COLLECTOR APP USED BY CROWDWORKERS.
# (build.sh to build the data collector app; it will appear in dist/ folder)

import difflib  # For fuzzy name suggestions
try:
    import fcntl  # POSIX-only; used for non-blocking I/O on Unix-like systems
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading  # New: for monitoring FFmpeg stderr asynchronously
import time
import urllib.request
import zipfile
from typing import Any, Callable, IO, Optional, Tuple, List

PROD: bool = True  # Set to True for production (PyInstaller app)

# --- Configuration ---
ADB_DIR: str = os.path.expanduser(os.path.join("~", "Downloads", "adb"))
ADB_BIN: str = "adb"  # Will be set to full path after setup
FFMPEG_DIR: str = os.path.expanduser(os.path.join("~", "Downloads", "ffmpeg"))
FFMPEG_BIN: str = "ffmpeg"  # Will be set to full path after setup

# Recording duration limit in seconds.
# Set to None to record indefinitely (until manually stopped).
RECORD_TIME_LIMIT_SECONDS: Optional[int] = None

# scrcpy quality settings (lower = less CPU/network)
SCRCPY_MAX_SIZE: int = 1080  # 0 = original; default 1080p
SCRCPY_MAX_FPS: int = 30   # 0 = device default; 15–30 reduces load


def get_adb_path() -> str:
    """Ensure ADB is installed in ~/Downloads/adb and return its path."""
    sys_os: str = platform.system().lower()
    if sys_os == "darwin":
        url = (
            "https://dl.google.com/android/repository/platform-tools-latest-darwin.zip"
        )
        adb_exe = "adb"
    elif sys_os == "windows":
        url = (
            "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
        )
        adb_exe = "adb.exe"
    elif sys_os == "linux":
        url = "https://dl.google.com/android/repository/platform-tools-latest-linux.zip"
        adb_exe = "adb"
    else:
        raise RuntimeError(f"Unsupported OS for adb auto-install: {sys_os}")

    adb_path: str = os.path.join(ADB_DIR, adb_exe)
    if os.path.exists(adb_path) and os.access(adb_path, os.X_OK):
        return adb_path

    # Download and extract if not present
    os.makedirs(ADB_DIR, exist_ok=True)
    zip_path: str = os.path.join(ADB_DIR, "platform-tools.zip")
    print(
        f"⬇️  Downloading adb platform-tools for {sys_os} to {zip_path} (This is necessary for your computer to communicate with the connected Android device)..."
    )
    urllib.request.urlretrieve(url, zip_path)
    print(f"📦 Extracting platform-tools...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(ADB_DIR)
    # Move adb binary to ADB_DIR
    extracted_dir = os.path.join(ADB_DIR, "platform-tools")
    extracted_adb = os.path.join(extracted_dir, adb_exe)
    if not os.path.exists(extracted_adb):
        raise RuntimeError(f"adb binary not found after extraction: {extracted_adb}")
    shutil.move(extracted_adb, adb_path)
    # Clean up extracted folder and zip
    shutil.rmtree(extracted_dir)
    os.remove(zip_path)
    if sys_os != "windows":
        os.chmod(adb_path, 0o755)
    print(f"✅ adb installed at {adb_path}")
    return adb_path


def get_ffmpeg_path() -> str:
    """Ensure FFmpeg is installed in ~/Downloads/ffmpeg and return its path."""
    sys_os: str = platform.system().lower()
    arch: str = platform.machine().lower()
    if sys_os == "darwin":
        # macOS universal binary (Intel/Apple Silicon)
        url = "https://evermeet.cx/ffmpeg/getrelease/zip"
        ffmpeg_exe = "ffmpeg"
    elif sys_os == "windows":
        # Windows static build (64-bit)
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        ffmpeg_exe = "ffmpeg.exe"
    elif sys_os == "linux":
        # Linux static build (x86_64)
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        ffmpeg_exe = "ffmpeg"
    else:
        raise RuntimeError(f"Unsupported OS for ffmpeg auto-install: {sys_os}")

    ffmpeg_path: str = os.path.join(FFMPEG_DIR, ffmpeg_exe)
    if os.path.exists(ffmpeg_path) and os.access(ffmpeg_path, os.X_OK):
        return ffmpeg_path

    # Download and extract if not present
    os.makedirs(FFMPEG_DIR, exist_ok=True)
    print(f"⬇️  Downloading ffmpeg static binary for {sys_os} to {FFMPEG_DIR}...")
    if sys_os == "windows":
        zip_path = os.path.join(FFMPEG_DIR, "ffmpeg.zip")
        urllib.request.urlretrieve(url, zip_path)
        print(f"📦 Extracting ffmpeg...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Find ffmpeg.exe in the archive
            for member in zip_ref.namelist():
                if member.endswith(ffmpeg_exe):
                    zip_ref.extract(member, FFMPEG_DIR)
                    src = os.path.join(FFMPEG_DIR, member)
                    shutil.move(src, ffmpeg_path)
                    break
        os.remove(zip_path)
    elif sys_os == "darwin":
        zip_path = os.path.join(FFMPEG_DIR, "ffmpeg.zip")
        urllib.request.urlretrieve(url, zip_path)
        print(f"📦 Extracting ffmpeg...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if member == ffmpeg_exe:
                    zip_ref.extract(member, FFMPEG_DIR)
                    src = os.path.join(FFMPEG_DIR, member)
                    shutil.move(src, ffmpeg_path)
                    break
        os.remove(zip_path)
        os.chmod(ffmpeg_path, 0o755)
    elif sys_os == "linux":
        tar_path = os.path.join(FFMPEG_DIR, "ffmpeg.tar.xz")
        urllib.request.urlretrieve(url, tar_path)
        print(f"📦 Extracting ffmpeg...")
        import tarfile

        with tarfile.open(tar_path, "r:xz") as tar_ref:
            for member in tar_ref.getmembers():
                if member.isfile() and member.name.endswith(f"/ffmpeg"):
                    tar_ref.extract(member, FFMPEG_DIR)
                    src = os.path.join(FFMPEG_DIR, member.name)
                    shutil.move(src, ffmpeg_path)
                    break
        os.remove(tar_path)
        os.chmod(ffmpeg_path, 0o755)
    print(f"✅ ffmpeg installed at {ffmpeg_path}")
    return ffmpeg_path


def _find_bundled_tool(exe_win: str, exe_posix: str) -> Optional[str]:
    """Find a tool next to the frozen executable or on PATH.

    Returns absolute path if found, else None.
    """
    sys_os = platform.system().lower()
    exe_name = exe_win if sys_os == "windows" else exe_posix

    candidates: List[str] = []

    # If running as a PyInstaller onefile/onedir app, first check the extraction dir
    # In onefile mode, PyInstaller extracts added binaries to sys._MEIPASS
    meipass_dir = getattr(sys, "_MEIPASS", None)
    if meipass_dir:
        candidates.append(os.path.join(meipass_dir, exe_name))

    # Also check alongside the frozen executable (helpful for onedir or manual placement)
    try:
        base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else None
    except Exception:
        base_dir = None
    if base_dir:
        candidates.append(os.path.join(base_dir, exe_name))
    # PATH fallback
    which = shutil.which(exe_name)
    if which:
        candidates.append(which)
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

# Resolve tool paths per-OS: prefer bundled when available
sys_os_lower = platform.system().lower()
if sys_os_lower == "windows":
    # Windows: require bundled or PATH (no runtime downloads to avoid AV flags)
    ADB_BIN = _find_bundled_tool("adb.exe", "adb")
    FFMPEG_BIN = _find_bundled_tool("ffmpeg.exe", "ffmpeg")
    if not ADB_BIN:
        print("❌ adb.exe not found. Please ensure it is bundled next to the EXE or installed on PATH.")
        sys.exit(1)
    if not FFMPEG_BIN:
        print("❌ ffmpeg.exe not found. Please ensure it is bundled next to the EXE or installed on PATH.")
        sys.exit(1)
else:
    # macOS/Linux: prefer bundled (sys._MEIPASS or app dir); fallback to convenience auto-install
    ADB_BIN = _find_bundled_tool("adb.exe", "adb")
    FFMPEG_BIN = _find_bundled_tool("ffmpeg.exe", "ffmpeg")
    if not ADB_BIN:
        ADB_BIN = get_adb_path()
    if not FFMPEG_BIN:
        FFMPEG_BIN = get_ffmpeg_path()

# --- Ensure adb directory is on PATH for downstream tools like scrcpy ---------
adb_dir_in_path = os.path.dirname(ADB_BIN)
current_path = os.environ.get("PATH", "")
if adb_dir_in_path and adb_dir_in_path not in current_path.split(os.pathsep):
    os.environ["PATH"] = f"{adb_dir_in_path}{os.pathsep}{current_path}"

# Explicit variable so scrcpy can find it even if PATH tricks fail
os.environ["ADB"] = ADB_BIN

# --- Audio capture helper --------------------------------------------

# Cross-platform quoting for shell commands
# On Windows, prefer double quotes; on POSIX, defer to shlex.quote
def quote_arg(arg: str) -> str:
    sys_os = platform.system().lower()
    if sys_os == "windows":
        # Quote only if spaces or special CMD characters present
        if any(c in arg for c in ' <>|&()^') or " " in arg:
            return f'"{arg}"'
        return arg
    else:
        return shlex.quote(arg)


def get_audio_input_ffmpeg_args() -> List[str]:
    """Return FFmpeg input args (tokenized) to capture host microphone.

    Empty list if unsupported.
    - macOS:  avfoundation default audio (index 0)
    - Windows: dshow default capture device via virtual-audio-capturer
    - Linux:  PulseAudio default source
    """
    sys_os: str = platform.system().lower()

    if sys_os == "darwin":
        # ffmpeg -f avfoundation -i ":0"
        return ["-f", "avfoundation", "-i", ":0"]
    elif sys_os == "windows":
        # ffmpeg -f dshow -i audio=virtual-audio-capturer
        return ["-f", "dshow", "-i", "audio=virtual-audio-capturer"]
    elif sys_os == "linux":
        # ffmpeg -f pulse -i default
        return ["-f", "pulse", "-i", "default"]
    else:
        return []


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


# ATTENTION: NOT in util.py because pyinstaller doesn't like it when we import from another file
def sanitize_path_component(component: str) -> str:
    """Return a filesystem-safe version of *component*."""
    # Keep letters, numbers, underscore, hyphen and space; drop the rest.
    safe = re.sub(r"[^A-Za-z0-9 _\-]", "", component).strip()
    # Compress whitespace to single spaces.
    safe = re.sub(r"\s+", " ", safe)
    return safe or "unnamed"


def get_data_base_dir() -> str:
    """Return the base directory for data storage depending on PROD flag."""
    if PROD:
        return os.path.expanduser(os.path.join("~", "Downloads", "data"))
    else:
        return "data"


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
    data_base_dir: str = get_data_base_dir()
    if os.path.isdir(data_base_dir):
        existing_games = [
            d
            for d in os.listdir(data_base_dir)
            if os.path.isdir(os.path.join(data_base_dir, d))
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

    base_dir = os.path.join(data_base_dir, safe_game)
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
    print("🔎 Searching for touchscreen Android device...")
    try:
        # Run the adb command to list input devices
        if platform.system().lower() == "windows":
            cmd = [ADB_BIN, "shell", "getevent", "-lp"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        else:
            command = f"{ADB_BIN} shell getevent -lp"
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
            # Some OEMs label the touch device without the literal word "touchscreen" (e.g. "goodix_ts").
            name_match = re.search(r'name:\s+"([^"]+)"', block, re.IGNORECASE)
            dev_name = name_match.group(1).lower() if name_match else ""
            keywords = ("touchscreen", "touch", "ts", "goodix", "synaptics", "fts")
            has_abs_mt_x = "ABS_MT_POSITION_X" in block
            has_abs_mt_y = "ABS_MT_POSITION_Y" in block
            is_touchscreen = any(k in dev_name for k in keywords)

            # Accept if either heuristic matches and the ABS coordinates are present
            if (is_touchscreen or (has_abs_mt_x and has_abs_mt_y)) and has_abs_mt_x and has_abs_mt_y:
                # Extract the device path from the first line of the block
                # Format is typically: " 7: /dev/input/event11"
                lines = block.strip().split("\n")
                if lines:
                    first_line = lines[0].strip()
                    # Look for the pattern "number: /dev/input/eventX"
                    match = re.search(r"\d+:\s*(/dev/input/event\d+)", first_line)
                    if match:
                        device_path = match.group(1)
                        print(f"✅ Found touchscreen Android device: {device_path}")
                        return device_path

    except FileNotFoundError:
        print(
            f"❌ Error: '{ADB_BIN}' command not found. Ensure there is the 'adb' folder in your Downloads folder. If not, rerun this app."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"❌ Error executing ADB command: {e}"
            "❌Is your Android phone connected and authorized? See guide at: https://www.notion.so/Instructions-for-Data-Collectors-217eefc8733380268679c3597593c3c9"
            "❌Connect your Android phone to your computer via USB first, then launch this app on your computer!"
        )
        sys.exit(1)

    print("❌ Error: Could not find a suitable touchscreen Android device.")
    sys.exit(1)


def get_screen_size() -> Tuple[Optional[int], Optional[int]]:
    """Gets the physical screen size (resolution) of the device."""
    print("📏 Getting screen size...")
    try:
        # Run the adb command to get screen size
        if platform.system().lower() == "windows":
            command_list = [ADB_BIN, "shell", "wm", "size"]
            result = subprocess.run(
                command_list, capture_output=True, text=True, check=True
            )
        else:
            command = f"{ADB_BIN} shell wm size"
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


def _round_to_multiple(value: int, base: int = 16) -> int:
    """Round *value* to the nearest lower multiple of *base* (>= base)."""
    if value < base:
        return base
    return value - (value % base)


def pick_recording_max_size(default_max: int = 1080) -> int:
    """Prompt the collector for desired recording size (longest side).

    Returns an integer like 720, 1080, 1440, or 0 for original device size.
    """
    print("")
    print("📺 Default recording resolution is 1080p (longest side).")
    print("   You can change it now if needed.")
    print("   Enter one of: 0 (original), 720, 1080, 1440, or a custom integer.")
    resp = input(f"➡️  Enter desired max size [press Enter for {default_max}]: ").strip()
    if not resp:
        return default_max
    try:
        val = int(resp)
        if val < 0:
            raise ValueError
        return val
    except ValueError:
        print("⚠️ Invalid input; using default 1080p.")
        return default_max


def compute_scaled_size(
    orig_w: Optional[int], orig_h: Optional[int], max_side: int
) -> Tuple[Optional[int], Optional[int]]:
    """Compute WxH scaled so that the longest side equals *max_side*.

    Returns multiples of 16 for codec compatibility. If inputs are missing,
    returns (None, None) to use device defaults.
    """
    if not orig_w or not orig_h or max_side <= 0:
        return None, None
    longest = max(orig_w, orig_h)
    if longest == 0:
        return None, None
    if longest == max_side:
        w, h = orig_w, orig_h
    else:
        scale = max_side / float(longest)
        w = int(round(orig_w * scale))
        h = int(round(orig_h * scale))
    # Round down to multiples of 16
    w = _round_to_multiple(w, 16)
    h = _round_to_multiple(h, 16)
    return max(w, 16), max(h, 16)


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
        # Run the adb command to get device info
        if platform.system().lower() == "windows":
            command_list = [ADB_BIN, "shell", "getevent", "-lp", device_path]
            result = subprocess.run(
                command_list, capture_output=True, text=True, check=True
            )
        else:
            command = f"{ADB_BIN} shell getevent -lp {device_path}"
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
    print(
        "\n🛑 Ctrl+C detected! Starting shutdown. PLEASE DO NOT CLOSE THIS WINDOW UNTIL TOLD TO DO SO."
    )
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

    print("✅ All streams stopped. You can now close this window.")
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

    print("\n=========================================================")
    print("🤗 Welcome to the Game Data Collector! 🤗")
    print("=========================================================")
    print("")

    # 🆕 Prepare game-specific output paths ------------------------------------
    prepare_output_paths()

    # 1. Find the device and get device info
    touch_device = find_touchscreen_device()
    screen_width, screen_height = get_screen_size()

    # 🆕 Ask collector for preferred recording max size (longest side)
    user_max_side = pick_recording_max_size(default_max=SCRCPY_MAX_SIZE)
    print(f"✅ Using max size: {user_max_side if user_max_side else 'original (no downscale)'}")

    # 2. Get coordinate ranges for translation
    x_min, x_max, y_min, y_max = get_input_device_ranges(touch_device)
    coordinate_translator = create_coordinate_translator(
        x_min, x_max, y_min, y_max, screen_width, screen_height
    )

    # 3. Start event stream but wait for first touch to start video
    if platform.system().lower() == "windows":
        event_command = [ADB_BIN, "shell", "getevent", "-lt", touch_device]
    else:
        event_command = shlex.split(f"{ADB_BIN} shell getevent -lt {touch_device}")
    event_process = subprocess.Popen(
        event_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    processes.append(event_process)
    print(
        f"🕒 Touch your Android phone screen to start recording. Waiting for the touch..."
    )

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
        # Prefer scrcpy on any OS if available: it can record internal device audio and run indefinitely.
        scrcpy_bin = shutil.which("scrcpy")
        use_scrcpy = scrcpy_bin is not None

        # Prepare quoted output path (may contain spaces)
        quoted_output_file = quote_arg(VIDEO_OUTPUT_FILE)

        if use_scrcpy:
            # scrcpy records on-device encoded video *and* audio, muxed on host.
            # --no-playback disables live window; --no-control avoids input capture.
            # We keep --no-window to avoid spawning GUI windows in the packaged app.
            video_args = [
                scrcpy_bin,
                "--no-playback", "--no-control", "--no-window",
                f"--max-size={user_max_side}", f"--max-fps={SCRCPY_MAX_FPS}",
                "--audio-codec=aac", f"--record={VIDEO_OUTPUT_FILE}",
            ]
            if RECORD_TIME_LIMIT_SECONDS:
                video_args.append(f"--time-limit={RECORD_TIME_LIMIT_SECONDS}")
            print("🎥 Recording via scrcpy with internal device audio")
        else:
            # Fallback to adb screenrecord piped to FFmpeg plus optional host-audio capture.
            audio_args = get_audio_input_ffmpeg_args()  # List[str]

            # Build ADB screenrecord command (writes raw H.264 to stdout)
            adb_cmd: List[str] = [ADB_BIN, "shell", "screenrecord"]
            # Apply scaling if user requested a cap (>0); else use device default/original
            if user_max_side and user_max_side > 0 and screen_width and screen_height:
                scaled_w, scaled_h = compute_scaled_size(screen_width, screen_height, user_max_side)
                if scaled_w and scaled_h:
                    adb_cmd += ["--size", f"{scaled_w}x{scaled_h}"]
                    print(f"🎥 Recording (scaled) at {scaled_w}x{scaled_h} (max-side {user_max_side})")
                else:
                    print("🎥 Recording at device default resolution (could not compute scale)")
            elif screen_width and screen_height:
                # Use the device physical resolution when available (original)
                adb_cmd += ["--size", f"{screen_width}x{screen_height}"]
                print(f"🎥 Recording at {screen_width}x{screen_height} resolution (original)")
            else:
                print("🎥 Recording at device default resolution (fallback)")
            if RECORD_TIME_LIMIT_SECONDS:
                adb_cmd += ["--time-limit", str(RECORD_TIME_LIMIT_SECONDS)]
            else:
                print("⚠️ Android 'screenrecord' may stop after a few minutes without a time limit. Install scrcpy for unlimited recording.")
            adb_cmd += ["--output-format=h264", "-"]

            # Build FFmpeg command reading from stdin (pipe:0)
            ffmpeg_args: List[str] = [
                FFMPEG_BIN,
                "-y",
                "-use_wallclock_as_timestamps", "1",
                "-fflags", "+genpts",
                "-f", "h264", "-i", "pipe:0",
            ]
            if audio_args:
                ffmpeg_args += audio_args + [
                    "-shortest",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                ]
            else:
                ffmpeg_args += ["-c", "copy"]
            ffmpeg_args += [VIDEO_OUTPUT_FILE]

        # Prepare video error log
        video_log_file = open(VIDEO_ERROR_LOG_FILE, "a")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if use_scrcpy:
            video_log_file.write(
                f"\n[{timestamp}] Launching scrcpy: {' '.join(map(quote_arg, video_args))}\n"
            )
        else:
            video_log_file.write(
                f"\n[{timestamp}] Launching pipeline:\n  ADB: {' '.join(map(quote_arg, adb_cmd))}\n  FFmpeg: {' '.join(map(quote_arg, ffmpeg_args))}\n"
            )
        video_log_file.flush()

        # Capture stream start time when video recording begins
        stream_start_time = time.time()
        if use_scrcpy:
            video_process = subprocess.Popen(
                video_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            processes.append(video_process)
        else:
            # Start ADB → FFmpeg pipeline programmatically (no shell/pipes)
            adb_process = subprocess.Popen(
                adb_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            ffmpeg_process = subprocess.Popen(
                ffmpeg_args,
                stdin=adb_process.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Ensure FFmpeg is the tracked video_process for stderr monitoring
            video_process = ffmpeg_process
            processes.extend([adb_process, ffmpeg_process])
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

    # Set event stream to non-blocking for the main loop (Unix-like systems only)
    if FCNTL_AVAILABLE:
        fd = event_process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    else:
        print("ℹ️ Non-blocking stream not available on this OS; continuing in blocking mode")

    print("\n=========================================================")
    print(
        ">>> Recording session successfully started! Press Ctrl+C in this window to stop. <<<"
    )
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
