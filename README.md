# roam-game-db
tools for creating a database of user interactions in mobile games to facilitate AI-driven game creation

see companion notion page at https://www.notion.so/Game-Scraping-217eefc87333809d9d03d93af9afaf64

## Installation

- install adb (one of android platform tools) - [for mac](https://dl.google.com/android/repository/platform-tools-latest-darwin.zip), [for windows](https://dl.google.com/android/repository/platform-tools-latest-windows.zip), [for linux](https://dl.google.com/android/repository/platform-tools-latest-linux.zip)
- install ffmpeg:
    - for mac: `brew install ffmpeg`
    - for windows: download the latest static build from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) and add the `bin` folder to your PATH
    - for linux: use your package manager, e.g. `sudo apt install ffmpeg` for Ubuntu/Debian
- install Python requirements using uv:
    uv sync
- (Optional) For advanced video analysis with Grounded-SAM-2, run the setup script:
    ./scripts/setup_grounded_sam_mac.sh  # For macOS
    # This downloads ~2GB of AI models for object detection


## Data folder layout

```
data/
├── <game_name>/                        # e.g. "subway surfers"
│   └── <TIMESTAMP>/                    # e.g. 08-06-25_at_22.06.31 - each recording session
│       ├── screen_recording.mp4        # Raw screen recording (from android_streamer)
│       ├── touch_events.log            # Raw touch event log (from android_streamer)
│       ├── video_error.log             # Raw FFmpeg/ADB error log (from android_streamer). Not really used
│       ├── screen_recording_overlay.mp4# Video with touch visualization overlay (created by visualize.py)
│       ├── frames/                     # Frames captured from the video at intervals (created by frame_cutter.py; all CONSTANTS in this section are inside frame_cutter.py). They will be fed into the LLM and segmentation model by analyze.py
│       │   ├── <timestamp>_time.png        # Timeline snapshot (every FRAME_INTERVAL)
│       │   ├── <timestamp>_touch.png       # Frame at the moment of user touch
│       │   ├── <timestamp>_pre_touch.png   # k×PRE_INTERVAL before user touch
│       │   └── <timestamp>_post_touch.png  # k×POST_INTERVAL after user touch
│       │                                     where k is in [N_BEFORE, N_AFTER]
│       ├── analysis/                  # LLM analysis results (created by analyze.py and sam2.py)
│       │   ├── frame_analysis.json        # Per-frame analysis output. See SceneAnalysis in base_models.py for JSON schema
│       │   ├── action_analysis.json       # Per-action analysis output. See ActionAnalysis in base_models.py for JSON schema
│       │   └── segmented_assets/          # Segmented (cutout) assets by type
│       │       ├── Player Character/      # Example asset type
│       │       │   └── <timestamp>_time.png  # Cutout asset as a transparent PNG file. Ready to be used for 2D->3D AI model e.g. Meshy
```

Each *game_name* directory holds multiple recording *sessions*, one per *TIMESTAMP* folder. The helper `src.util.list_sessions()` enumerates them newest → oldest.

- The data collectors (crowdworkers) should be sending you only the raw files - `screen_recording.mp4`, `touch_events.log`, `video_error.log`. You then run the postprocessing on your machine
