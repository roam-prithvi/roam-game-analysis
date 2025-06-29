# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mobile game analysis toolkit that records gameplay sessions, extracts frames, analyzes player behavior using AI models (Gemini, YOLO, SAM2), and generates behavior trees. The project is designed to facilitate AI-driven game creation by understanding gameplay mechanics and extracting game assets.

## Common Development Commands

### Data Collection & Processing
```bash
# Build the crowdworker data collection app
./build.sh  # Creates dist/Game_Data_Collector_V1

# Process a recording session (creates overlay video and extracts frames)
python -m src.processing.visualize <session_path>
python -m src.processing.frame_cutter <session_path>

# Create video chunks for spatial understanding pipeline
python -m src.processing.video_chunker --game "subway surfers"  # Latest session
python -m src.processing.video_chunker --game "subway surfers" --all  # All sessions
```

### Analysis Commands
```bash
# Run complete analysis on a game session
python -m src.analysis.analyze <session_path>

# Generate behavior trees
python -m src.analysis.bt_generator --game <game_name> --session <number>

# Run video-based behavior analysis
python -m src.analysis.video_behavior_analysis.run_video_analysis

# Run YOLO object detection analysis
python -m src.analysis.video_behavior_analysis.run_yolo_analysis
```

### Testing
```bash
# Test single session analysis
python test_single_session.py

# Test universal pipeline
python test_universal_pipeline.py

# Test grounded detector
python test_grounded_detector.py
```

### Re-analysis Scripts
```bash
# Re-analyze Plants vs Zombies 2 sessions
./reanalyze_pvz2.sh
```

## Architecture & Key Components

### Data Flow
1. **Recording**: Android streamer app (`src/streaming/android_streamer.py`) captures screen and touch events
2. **Processing**: Frame extraction (`src/processing/frame_cutter.py`) and touch visualization (`src/processing/visualize.py`)
3. **Analysis**: Multi-model analysis pipeline in `src/analysis/`
   - `analyze.py`: Main orchestrator using Gemini for scene/action understanding
   - `video_behavior_analysis/`: Extended video analysis with YOLO
   - `sam2.py`: Asset segmentation using SAM2
4. **Output**: Behavior trees (`bt_generator.py`) and segmented assets

### Key Models & Classes
- `src/analysis/base_models.py`: Core data models (SceneAnalysis, ActionAnalysis)
- `src/analysis/video_behavior_analysis/models.py`: Video analysis data structures
- `src/streaming/models.py`: Touch event models

### Environment Setup
The project uses environment variables from `.env` file for API keys:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

### Data Organization
```
data/<game_name>/<timestamp>/
├── screen_recording.mp4        # Raw recording
├── touch_events.log           # Touch data
├── screen_recording_overlay.mp4 # With touch visualization
├── chunked/                   # Video chunks for spatial understanding
│   ├── 0.mp4                  # 5-second chunks with 2s overlap
│   ├── 1.mp4
│   └── metadata.json          # Chunk timing info
├── frames/                    # Extracted frames
└── analysis/                  # Analysis results
    ├── frame_analysis.json
    ├── action_analysis.json
    └── segmented_assets/
```

## Development Notes

- The project uses `uv` as the package manager (see `uv.lock`)
- No formal linting setup - follow existing code style
- Tests are standalone scripts without a test framework
- PyInstaller is used to build the crowdworker executable
- Frame extraction intervals are configured in `src/processing/frame_cutter.py`

## Documentation & Integration Notes

- Use Context7 MCP Server for Google ADK documentation. `context7CompatibleLibraryID: "/google/adk-docs"`