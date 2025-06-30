# Spatial Reasoning Pipeline Documentation

## The Problem

Mobile games present rich 3D environments that are difficult to analyze programmatically. Traditional computer vision approaches struggle with:

1. **Dynamic Viewpoints**: Mobile games have constantly moving cameras following player actions
2. **2D to 3D Mapping**: Converting 2D screen coordinates to 3D world positions requires spatial understanding
3. **Temporal Continuity**: Objects appear, move, and disappear across time
4. **Game-Specific Logic**: Each game has unique visual styles, objects, and spatial layouts

### Why This Matters

- **Game Development**: Understanding spatial layouts helps in level design and balancing
- **AI Training**: 3D scene understanding is crucial for game-playing agents
- **Asset Extraction**: Identifying and positioning game objects for reuse
- **Gameplay Analysis**: Understanding player navigation patterns in 3D space

## Our Solution: AI-Powered Spatial Reasoning

We use Google's Gemini 2.0 model with custom prompts to analyze gameplay videos and reconstruct 3D scenes as Unity-compatible JSON. The pipeline converts temporal 2D footage into static 3D representations.

### Key Innovation: Compositional Function Calling

Instead of asking the AI to output JSON directly, we provide it with file manipulation tools:
- `read_file`: Check existing scene data
- `write_file`: Create initial scene
- `edit_file`: Update scene incrementally

This approach allows the AI to build scenes iteratively, maintaining consistency across video chunks.

## Complete Pipeline Walkthrough

### Prerequisites

```bash
# Install dependencies
pip install google-generativeai tqdm

# Set environment variable
export GEMINI_API_KEY="your-api-key-here"
```

### Step 1: Record Gameplay

Use the Android data collection app:

```bash
# Build the app
./build.sh

# Install on device and record gameplay
# Output: data/<game>/<timestamp>/screen_recording.mp4
```

### Step 2: Create Video Chunks

Split recordings into manageable segments:

```bash
# Process latest recording
python -m src.processing.video_chunker --game "subway surfers"

# Process specific session
python -m src.processing.video_chunker \
  --game "subway surfers" \
  --session "08-06-25_at_19.33.00"

# Custom chunk parameters
python -m src.processing.video_chunker \
  --game "brawl stars" \
  --chunk-duration 10 \
  --overlap 2
```

**Output Structure:**
```
data/subway surfers/08-06-25_at_19.33.00/
├── screen_recording.mp4      # Original
├── chunked/
│   ├── 0.mp4                # 0-10s
│   ├── 1.mp4                # 8-18s (2s overlap)
│   ├── 2.mp4                # 16-26s
│   └── metadata.json        # Timing information
```

### Step 3: Upload Chunks to Gemini

Upload chunks for AI processing:

```bash
# Interactive selection
python -m src.analysis.spatial.chunk_uploader --interactive

# Filter by game
python -m src.analysis.spatial.chunk_uploader \
  --interactive \
  --game "subway surfers"

# Upload specific directory
python -m src.analysis.spatial.chunk_uploader \
  --dir "data/subway surfers/08-06-25_at_19.33.00/chunked"
```

**Features:**
- Parallel uploads (5 workers default)
- Progress tracking with tqdm
- Automatic retry on failures
- Saves results to `upload_results_*.json`

### Step 4: Run Spatial Reasoning

Process uploaded chunks through the AI:

```bash
# Interactive mode (recommended)
python -m src.analysis.spatial.agent_run --interactive

# Direct execution
python -m src.analysis.spatial.agent_run \
  --chunks-dir "data/subway surfers/08-06-25_at_19.33.00/chunked" \
  --upload-results "upload_results_20250629_142726.json"

# Override game detection
python -m src.analysis.spatial.agent_run \
  --chunks-dir "custom/path/chunks" \
  --upload-results "results.json" \
  --game "brawl stars"
```

**Output:** `unity/subway_surfers/08-06-25_at_19.33.00/output.json`

## The AI Process

### 1. Video Analysis
The AI watches each 10-second chunk and identifies:
- Static geometry (walls, floors, obstacles)
- Dynamic objects (coins, power-ups, enemies)
- Spatial relationships and scale

### 2. 3D Reconstruction
Using game-specific prompts, the AI:
- Places objects in 3D coordinates
- Selects appropriate primitive shapes
- Assigns colors and properties
- Tracks object lifecycles with timestamps

### 3. Iterative Building
Through function calling, the AI:
1. Reads existing scene data
2. Adds new objects from current chunk
3. Updates positions based on camera movement
4. Maintains consistency across chunks

## Example Output

```json
{
  "scene": {
    "name": "Subway Surfers - Train Yard",
    "description": "Reconstructed 3D environment from gameplay footage"
  },
  "objects": [
    {
      "name": "Ground_Track",
      "model": "Plane",
      "position": { "x": 0, "y": 0, "z": 50 },
      "scale": { "x": 10, "y": 1, "z": 100 },
      "color": "#8B4513",
      "start_timestamp": 0.0,
      "end_timestamp": 10.0
    },
    {
      "name": "Train_Left",
      "model": "Cube",
      "position": { "x": -3, "y": 1.5, "z": 30 },
      "scale": { "x": 2, "y": 3, "z": 20 },
      "color": "#FF0000",
      "start_timestamp": 2.5,
      "end_timestamp": 10.0
    }
  ]
}
```

## Game-Specific Configurations

### Subway Surfers
- **Focus**: Rails, trains, barriers, coins, power-ups
- **Camera**: Behind-player view, forward movement
- **Challenges**: Fast motion, object spawning

### Brawl Stars
- **Focus**: Arena walls, bushes, power cubes, spawn points
- **Camera**: Top-down 45-degree angle
- **Challenges**: Symmetrical maps, destructible objects

## Troubleshooting

### Common Issues

1. **"Model is overloaded" (503 error)**
   - The agent automatically retries with exponential backoff
   - Maximum 5 attempts with increasing delays

2. **"No chunks found"**
   - Ensure video_chunker was run first
   - Check chunk directory exists: `ls data/game/session/chunked/`

3. **"Upload results don't match chunks"**
   - Chunks directory and upload results must correspond
   - Re-run upload if chunks were recreated

### Debugging

Check logs for detailed information:
```bash
# View latest agent log
tail -f logs/spatial_reasoning_*.log

# Check function calls
grep "Function calls detected" logs/spatial_reasoning_*.log

# View retry attempts
grep "retry attempt" logs/spatial_reasoning_*.log
```

## Advanced Usage

### Batch Processing

Process multiple sessions:
```bash
#!/bin/bash
for session in data/subway_surfers/*/chunked; do
  echo "Processing $session"
  python -m src.analysis.spatial.agent_run \
    --chunks-dir "$session" \
    --upload-results "latest_upload.json"
done
```

### Custom Prompts

Add new games by creating prompts in `prompts.py`:
```python
TEMPLE_RUN = """
The game is Temple Run. It is a 3D endless runner.
Objects to track:
- Path segments and turns
- Obstacles (trees, flames, gaps)
- Coins and power-ups
- Environmental decorations
...
"""
```

### Merging Results

Combine multiple chunks into a single scene:
```python
import json
from pathlib import Path

# Load all chunk outputs
scenes = []
for output in Path("unity/game/session").glob("chunk_*.json"):
    with open(output) as f:
        scenes.append(json.load(f))

# Merge objects by position and timestamp
merged = merge_scenes(scenes)
```

## Performance Metrics

Typical processing times:
- **Chunking**: ~5 seconds per minute of video
- **Upload**: ~4 seconds per chunk (parallel)
- **AI Processing**: ~15 seconds per chunk
- **Total**: ~5 minutes for 2-minute gameplay

## Future Enhancements

1. **Multi-chunk correlation**: Merge overlapping chunks intelligently
2. **Confidence scores**: AI uncertainty quantification
3. **Visual validation**: Generate Unity scenes for verification
4. **Real-time processing**: Stream processing during gameplay

## Architecture Benefits

1. **Scalability**: Process videos of any length by chunking
2. **Reliability**: Retry logic handles API failures
3. **Flexibility**: Easy to add new games with custom prompts
4. **Modularity**: Each component can be improved independently

This pipeline demonstrates how modern AI can solve complex spatial reasoning tasks that would be extremely difficult with traditional computer vision approaches.