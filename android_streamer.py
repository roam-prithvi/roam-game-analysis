import fcntl
import os
import re
import shlex
import signal
import subprocess
import sys
import time

# --- Configuration ---
# TODO: if adb not in PATH, try to find it, then download it, then add it to PATH and set ADB_PATH to the full path
# if adb in path, set ADB_PATH to "adb"
ADB_PATH = "adb"  # Or provide a full path if 'adb' is not in your system's PATH
EVENT_LOG_FILE = "touch_events.log"  # The file where raw touch events will be saved
VIDEO_OUTPUT_FILE = "screen_recording.mp4"  # The file where video will be saved

# --- Global variable to hold our running processes ---
processes = []
event_log_file = None  # Global to be accessible by signal_handler


def find_touchscreen_device():
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


def get_screen_size():
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


def signal_handler(sig, frame):
    """Handles Ctrl+C to gracefully shut down all subprocesses."""
    print("\n🛑 Ctrl+C detected! Shutting down streams...")
    for p in processes:
        try:
            p.terminate()  # Send termination signal
        except ProcessLookupError:
            pass  # Process already finished

    # Wait for a moment for processes to terminate
    time.sleep(0.5)
    for p in processes:
        if p.poll() is None:  # If process is still running
            p.kill()  # Force kill it

    global event_log_file
    if event_log_file and not event_log_file.closed:
        event_log_file.close()
        print("✅ Event log file closed.")

    print("✅ All streams stopped.")
    sys.exit(0)


def main():
    """Main function to set up and run the concurrent streams."""
    global event_log_file
    # Set up the Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Find the device and get device info
    touch_device = find_touchscreen_device()
    screen_width, screen_height = get_screen_size()

    # 2. Start event stream but wait for first touch to start video
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

    first_timestamp = None
    video_process = None

    # Block and wait for the first event to arrive
    for line in iter(event_process.stdout.readline, ""):
        if not line.strip():
            continue

        print("👆 First touch detected! Starting video recording...")
        # --- Video Stream ---
        # Build video command with proper resolution if available
        if screen_width and screen_height:
            video_command = (
                f"{ADB_PATH} shell screenrecord --size {screen_width}x{screen_height} "
                f"--output-format=h264 - | ffmpeg -f h264 -i - -c copy {VIDEO_OUTPUT_FILE}"
            )
            print(f"🎥 Recording at {screen_width}x{screen_height} resolution")
        else:
            video_command = (
                f"{ADB_PATH} shell screenrecord --output-format=h264 - "
                f"| ffmpeg -f h264 -i - -c copy {VIDEO_OUTPUT_FILE}"
            )
            print("🎥 Recording at default resolution")

        video_process = subprocess.Popen(
            video_command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(video_process)
        print(f"🔴 REC: Video is being recorded to '{VIDEO_OUTPUT_FILE}'")
        print(f"🔴 REC: Touch events are being streamed to '{EVENT_LOG_FILE}'")

        event_log_file = open(EVENT_LOG_FILE, "w")

        # Process the first line
        match = re.search(r"\[\s*(\d+\.\d+)\s*\]", line)
        if match:
            first_timestamp = float(match.group(1))
            new_ts_part = f"[{0.0:13.6f}]"
            adjusted_line = line.replace(match.group(0), new_ts_part)
            event_log_file.write(adjusted_line)
        else:
            event_log_file.write(line)
        event_log_file.flush()
        break  # Exit loop after handling the first event

    if first_timestamp is None:
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

    # 3. Wait until the user interrupts
    while True:
        if video_process.poll() is not None:
            print("ℹ️ Video recording process ended.")
            signal_handler(None, None)

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

                match = re.search(r"\[\s*(\d+\.\d+)\s*\]", line)
                if match:
                    current_timestamp = float(match.group(1))
                    adjusted_timestamp = current_timestamp - first_timestamp
                    new_ts_part = f"[{adjusted_timestamp:13.6f}]"
                    adjusted_line = line.replace(match.group(0), new_ts_part)
                    event_log_file.write(adjusted_line)
                else:
                    event_log_file.write(line)

            except BlockingIOError:
                break  # No more data available at the moment
            except ValueError:
                # Partial line read can cause float conversion errors, ignore for now
                pass

        event_log_file.flush()
        time.sleep(0.01)  # Prevent busy-waiting


if __name__ == "__main__":
    main()
