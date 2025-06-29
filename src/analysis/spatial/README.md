# Spatial Reasoning Pipeline

The spatial reasoning pipeline analyzes gameplay videos to extract 3D spatial information and generate Unity-compatible scene representations. It uses Google's Gemini AI to understand video content and reconstruct game environments as 3D primitives.

## Overview

The pipeline consists of three main components:

1. **Video Chunker** - Splits gameplay recordings into overlapping chunks
2. **Chunk Uploader** - Uploads video chunks to Google GenAI in parallel
3. **Agent Runner** - Processes chunks through the spatial reasoning agent

## Quick Start

### 1. Prepare Video Chunks

First, create video chunks from your gameplay recording:

```bash
# Create chunks for the latest session
python -m src.processing.video_chunker --game "subway surfers"

# Create chunks for all sessions of a game
python -m src.processing.video_chunker --game "subway surfers" --all

# Create chunks with custom parameters
python -m src.processing.video_chunker --game "brawl stars" --chunk-duration 10 --overlap 5
```

Default settings:
- Chunk duration: 10 seconds
- Overlap: 2 seconds

### 2. Upload Chunks to Google GenAI

Upload the chunks for processing:

```bash
# Interactive mode - select which sessions to upload
python -m src.analysis.spatial.chunk_uploader --interactive

# Filter by game
python -m src.analysis.spatial.chunk_uploader --interactive --game "subway surfers"

# Upload specific directory
python -m src.analysis.spatial.chunk_uploader --dir "data/subway surfers/08-06-25_at_19.33.00/chunked"

# List available sessions without uploading
python -m src.analysis.spatial.chunk_uploader --list
```

This creates an `upload_results_*.json` file with the uploaded file references.

### 3. Run Spatial Reasoning Agent

Process the uploaded chunks through the AI agent:

```bash
# Interactive mode - select chunks and upload results
python -m src.analysis.spatial.agent_run --interactive

# Specify explicit paths
python -m src.analysis.spatial.agent_run \
  --chunks-dir "data/subway surfers/08-06-25_at_19.33.00/chunked" \
  --upload-results "upload_results_20250629_002312.json"

# Override game detection (useful for custom directory structures)
python -m src.analysis.spatial.agent_run \
  --chunks-dir "custom/path/to/chunks" \
  --upload-results "upload_results.json" \
  --game "brawl stars"
```

**Game-Specific Prompts:**
The agent automatically detects the game from the directory path and uses appropriate prompts:
- **Subway Surfers**: Focuses on barriers, coins, trains, powerups, tunnels
- **Brawl Stars**: Focuses on walls, bushes, power cubes, spawn points, map boundaries

You can override auto-detection with the `--game` flag.

The agent will:
- Process each chunk sequentially
- Extract 3D spatial information
- Save output to `unity/<game>/<session>/output.json`
- Log all operations to `logs/spatial_reasoning_*.log`

**Output Organization:**
```
unity/
├── subway_surfers/
│   ├── 08-06-25_at_19.33.00/
│   │   └── output.json
│   └── 09-06-25_at_10.15.00/
│       └── output.json
└── brawl_stars/
    ├── 1/
    │   └── output.json
    └── 2/
        └── output.json
```

Each game and session has its own output file, preventing overwrites between runs.

## Output Format

The agent generates a Unity-compatible JSON file for each session:

```json
{
  "scene": {
    "name": "Subway Surfers - Trainyard",
    "description": "3D representation of gameplay environment"
  },
  "objects": [
    {
      "name": "Barrier",
      "model": "Cube",
      "position": { "x": 0, "y": 0.75, "z": 60 },
      "scale": { "x": 1.8, "y": 1.5, "z": 0.5 },
      "color": "#FF4500",
      "start_timestamp": 8.0,
      "end_timestamp": 18.0
    }
  ]
}
```

## Environment Setup

1. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

2. Install dependencies:
   ```bash
   uv pip install google-genai tqdm
   ```

## Common Workflows

### Full Pipeline for New Recording

```bash
# 1. Create chunks
python -m src.processing.video_chunker --game "subway surfers"

# 2. Upload chunks
python -m src.analysis.spatial.chunk_uploader --interactive

# 3. Run spatial reasoning
python -m src.analysis.spatial.agent_run --interactive
```

### Process Multiple Sessions

```bash
# Chunk all sessions
python -m src.processing.video_chunker --game "brawl stars" --all

# Upload with game filter
python -m src.analysis.spatial.chunk_uploader --interactive --game "brawl stars"

# Process each session
python -m src.analysis.spatial.agent_run --interactive
```

## Module Documentation

### chunk_uploader.py

Handles parallel uploads of video chunks to Google GenAI.

**Key Features:**
- Parallel uploads with progress tracking
- Interactive session selection
- Game filtering
- Automatic retry on failures
- Saves upload results for later use

**Classes:**
- `ChunkUploader`: Main upload manager with configurable workers

### agent_run.py

Runs the spatial reasoning agent on uploaded video chunks.

**Key Features:**
- Sequential chunk processing
- Automatic function calling for file operations
- Comprehensive logging
- Retry logic for API failures
- Function call statistics

**Key Functions:**
- `main()`: CLI entry point
- `validate_chunks_and_results()`: Ensures consistency
- `select_chunks_interactive()`: Interactive selection UI

### file_tools.py

File operation tools compatible with Google GenAI's function calling.

**Available Tools:**
- `read_file()`: Read JSON files with line numbers
- `write_file()`: Create new files
- `edit_file()`: Modify existing files

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not set"**
   - Set the environment variable: `export GEMINI_API_KEY="your-key"`

2. **"No chunks found"**
   - Ensure you've run the video chunker first
   - Check the chunk directory exists: `data/<game>/<session>/chunked/`

3. **"Upload results do not match chunks"**
   - The upload results file doesn't contain the selected chunks
   - Re-upload the chunks or select matching files

4. **503 Service Unavailable**
   - The model is overloaded
   - The agent will automatically retry with exponential backoff

### Logs

Check the detailed logs for debugging:
- Upload logs: Console output and `upload_results_*.json`
- Agent logs: `logs/spatial_reasoning_*.log`

## Advanced Usage

### Custom Chunk Parameters

```bash
# Longer chunks with more overlap
python -m src.processing.video_chunker \
  --game "subway surfers" \
  --chunk-duration 15 \
  --overlap 5
```

### Parallel Upload Configuration

```bash
# Use more workers for faster uploads
python -m src.analysis.spatial.chunk_uploader \
  --interactive \
  --workers 10
```

### Processing Specific Chunks

The agent processes chunks in order by their numeric names (0.mp4, 1.mp4, etc.). Each chunk includes timestamp information from the original video.

## See Also

- [SPATIAL_REASONING.md](SPATIAL_REASONING.md) - Technical architecture details
- [prompts.py](prompts.py) - AI prompt templates
- [file_tools.py](file_tools.py) - File operation implementations