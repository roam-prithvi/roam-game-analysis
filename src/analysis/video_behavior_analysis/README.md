# Video Behavior Analysis

This module uses **Gemini 2.5 Flash's native video understanding** to analyze gameplay recordings and extract detailed behavioral patterns, game mechanics, and strategic insights for AI behavior tree generation.

## Overview

Unlike the frame-by-frame analysis in the main analysis module, this system:
- **Analyzes entire videos** using Gemini 2.5 Flash's multimodal capabilities
- **Incorporates touch events** to understand player input patterns
- **Focuses on gameplay dynamics** and object relationships
- **Extracts strategic insights** for creating compelling AI behaviors
- **Is game-agnostic** - works with any type of mobile game

## New: YOLO-Based Object Detection and Segmentation

We now provide two powerful YOLO-based analyzers that use YOLOv11 for object detection to guide SAM 2 segmentation:

### 1. YOLOVideoAnalyzer
A general-purpose analyzer that:
- Detects all objects in game frames using YOLOv11
- Segments detected objects with high precision using SAM 2
- Extracts game assets as transparent PNGs
- Works with any game without prior configuration

### 2. GameAwareAnalyzer
A smart analyzer that:
- Uses game-specific object mappings for better categorization
- Filters detections to only relevant game objects
- Organizes assets by game-meaningful categories (player, obstacles, collectibles, etc.)
- Merges overlapping detections for cleaner results
- Supports popular games like Subway Surfers, Temple Run, Crossy Road, PvZ, and more

## Quick Start: YOLO Analysis

### Installation
```bash
# Make sure you have the required dependencies
pip install -r requirements.txt

# Set your Sieve API key (required for SAM 2)
export SIEVE_API_KEY='your-sieve-api-key'
```

### Running YOLO Analysis

#### Interactive Mode
```bash
python -m src.analysis.video_behavior_analysis.run_yolo_analysis
```

#### Command Line Options
```bash
# Analyze a specific session
python -m src.analysis.video_behavior_analysis.run_yolo_analysis data/subway_surfers/08-06-25_at_19.33.00

# Use game-aware analyzer for Subway Surfers
python -m src.analysis.video_behavior_analysis.run_yolo_analysis --game "subway surfers" --analyzer game_aware

# Batch process multiple sessions
python -m src.analysis.video_behavior_analysis.run_yolo_analysis --game "subway surfers" --batch --max-sessions 10

# Use basic YOLO analyzer
python -m src.analysis.video_behavior_analysis.run_yolo_analysis --analyzer basic
```

### Output Structure

YOLO analysis creates the following in each session's analysis folder:

```
analysis/
├── yolo_segmentation/           # Basic YOLO analyzer output
│   ├── yolo_analysis_results.json
│   └── segmented_assets/
│       ├── person/
│       ├── car/
│       └── ...
└── game_aware_segmentation/     # Game-aware analyzer output
    ├── game_aware_analysis.json
    └── segmented_assets/
        ├── player/              # Game-specific categories
        ├── obstacles/
        ├── collectibles/
        └── environment/
```

### Example Results

The analyzers produce:
- **JSON analysis files** with detection statistics and metadata
- **Transparent PNG cutouts** of detected objects, ready for 2D→3D conversion
- **Organized asset folders** categorized by object type

## Features

### Comprehensive Analysis
- **Player Behavior Patterns**: Decision triggers, reaction times, skill progression
- **Object/Enemy Behaviors**: Movement patterns, behavioral rules, timing characteristics  
- **Game Mechanics**: Core systems, resource management, win/lose conditions
- **Strategic Relationships**: How different elements interact to create depth
- **Challenge Design**: What makes the game engaging and difficult

### Touch Event Integration
- Correlates player inputs with visual gameplay
- Analyzes input timing and frequency patterns
- Understands player reaction speed to game events

### AI Behavior Insights
- Identifies effective enemy/object behaviors
- Analyzes what makes good "traps" or challenging scenarios
- Extracts timing and positioning strategies
- Provides insights for behavior tree generation

### Multi-Session Analysis

To analyze multiple sessions together for more comprehensive pattern recognition:

```bash
# Select all sessions from a game directory
python -m src.analysis.video_behavior_analysis.video_analyzer data/subway_surfers --all-sessions

# Or interactively select which sessions to analyze
python -m src.analysis.video_behavior_analysis.video_analyzer data/subway_surfers
```

Multi-session analysis provides:
- Cross-session behavioral patterns
- Consistency analysis across different gameplay sessions
- Aggregated insights for more robust behavior trees

## Objective Generation

After running video analysis, you can generate natural language objective descriptions suitable for the [objective-builder-v2-roam](https://objective-builder-v2-roam.streamlit.app/) tool:

### Running Objective Generation

```bash
# Generate objectives from a specific analysis file
python -m src.analysis.video_behavior_analysis.objective_generator data/subway_surfers/24-06-25_at_23.51.52/analysis/video_behavior_analysis/detailed_analysis_single_session_20250625_213251.txt

# List all available analyses and select one
python -m src.analysis.video_behavior_analysis.objective_generator --list-analyses
```

### Output

The objective generator creates:

1. **Structured Objectives Document** (`objectives_[game]_[timestamp].md`):
   - Core gameplay loop description
   - Primary objectives (main goals)
   - Secondary objectives (bonus goals)
   - Resources to track (coins, enemies, power-ups)
   - Natural language prompt for objective builder

2. **Natural Language Prompt** (`objective_prompt_[game]_[timestamp].txt`):
   - Ready-to-use prompt for the objective builder tool
   - Describes all objectives in clear, sequential language
   - Can be directly pasted into the objective builder

### Example Output

For a game like Subway Surfers, the generator might produce:

```
The player must survive as long as possible while collecting coins. 
Every 30 seconds, the speed increases by 10%. The player fails if 
they collide with any obstacle. Additionally, the player should 
collect power-ups to gain temporary abilities: collect the magnet 
to attract coins for 10 seconds, or collect the rocket to fly 
above obstacles for 15 seconds.
```

This prompt can then be used in the objective builder to generate the appropriate JSON configuration.

## Usage

### Interactive Mode
```bash
python -m src.analysis.video_behavior_analysis.run_video_analysis
```

The script will prompt you to:
1. Enter the game name
2. Select from available recording sessions

### Direct Path Mode
```bash
python -m src.analysis.video_behavior_analysis.run_video_analysis path/to/session
```

## Required Files

For each session, you need:
- `screen_recording.mp4` - The gameplay video
- `touch_events.log` - Touch interaction logs from android_streamer

## Output

The analysis creates a `video_behavior_analysis/` directory in the session's analysis folder with:

### `detailed_analysis_YYYYMMDD_HHMMSS.txt`
Comprehensive text analysis including:
- Touch events summary and patterns
- Detailed gameplay analysis from Gemini 2.5 Flash
- Strategic insights and behavioral patterns
- AI behavior tree recommendations

## Example Output Sections

The analysis includes detailed sections on:

### Player Behavior Analysis
- Decision triggers and reaction patterns
- Skill manifestation and error patterns
- Adaptation strategies

### Object/Entity Behavior Analysis  
- Movement patterns and behavioral rules
- Interaction dynamics with player
- Strategic importance and timing

### Game Mechanics Deep Dive
- Core rules and system interactions
- Resource management patterns
- Win/lose condition analysis

### Strategic Depth Analysis
- Risk/reward scenarios
- Timing and positioning effects
- Meaningful decision points

### AI Behavior Tree Insights
- Effective opponent behavioral patterns
- Challenge creation strategies
- Timing and positioning tactics

## Technical Details

### YOLOv11 Integration
- Uses YOLOv11 medium model for optimal speed/accuracy balance
- Configurable confidence and IoU thresholds
- Supports batch processing of frames

### SAM 2 Integration via Sieve
- Uses large SAM model for best segmentation quality
- Guided by YOLO bounding boxes for precise results
- Produces high-quality transparent PNG cutouts

### Gemini 2.5 Flash Integration
- Uses native video understanding (no frame extraction needed)
- Processes videos up to 1 hour at default resolution
- Correlates visual analysis with touch event timing

### Touch Event Processing
- Parses android_streamer log format
- Samples events for high-volume sessions
- Provides statistical analysis of input patterns

### Game-Agnostic Design
- No hardcoded game-specific templates
- Adaptive analysis based on observed patterns
- Works across different game genres

## Performance Notes

- **YOLO inference**: ~50-100ms per frame on GPU, 200-500ms on CPU
- **SAM segmentation**: 1-3 seconds per frame (via Sieve API)
- **Video upload**: Large videos may take time to upload to Gemini
- **Analysis time**: 30-60 seconds for typical gameplay sessions  
- **API usage**: Uses Gemini 2.5 Flash which has optimized performance

## Choosing the Right Analyzer

- **Use YOLOVideoAnalyzer when**:
  - You want to detect all possible objects
  - Working with a new/unknown game
  - Need comprehensive object detection
  
- **Use GameAwareAnalyzer when**:
  - Working with supported games (Subway Surfers, Temple Run, etc.)
  - Want game-specific categorization
  - Need cleaner, filtered results

- **Use Video Behavior Analysis (Gemini) when**:
  - Need comprehensive gameplay understanding
  - Want strategic insights and patterns
  - Analyzing player behavior and game mechanics

## Future Enhancements

Potential improvements:
- Custom YOLO models trained on game-specific data
- Real-time video analysis during gameplay
- Integration with behavior tree generator
- Support for more game titles
- Multi-object tracking across frames

## Troubleshooting

### Common Issues

**"Video file not found"**
- Ensure `screen_recording.mp4` exists in the session directory

**"Touch events log not found"** 
- Ensure `touch_events.log` exists in the session directory

**"SIEVE_API_KEY not set"**
- Sign up at https://www.sievedata.com/ and set your API key
- Export it: `export SIEVE_API_KEY='your-key'`

**"YOLOv11 model download failed"**
- Check internet connection
- Model will auto-download on first use (~50MB)

**"Upload failed"**
- Check internet connection and Gemini API key
- Large videos (>100MB) may have upload issues

**"Analysis incomplete"**
- Check Gemini API quota and rate limits
- Retry the analysis after a few minutes 