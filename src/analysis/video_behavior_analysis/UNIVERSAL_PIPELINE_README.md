# Universal Object Detection Pipeline

## Overview

The Universal Object Detection Pipeline is a game-agnostic solution that automatically detects and segments objects in any mobile game without requiring pre-configured game-specific labels. This solves the scalability problem of the original PvZ2-only implementation.

## How It Works

### Two-Pronged Approach

1. **Gemini-Driven Object Discovery**
   - Uses Gemini 2.5 Pro to analyze game videos
   - Extracts object names and behavioral descriptions
   - Automatically generates detection prompts from analysis

2. **Universal Object Detection**
   - Uses Grounding DINO + SAM 2 for open-vocabulary detection
   - Combines universal game object prompts with Gemini discoveries
   - Works with any game without manual configuration

### Pipeline Steps

1. **Video Analysis**: Gemini analyzes the game video to identify objects and behaviors
2. **Object Extraction**: Extract object names from Gemini's analysis using pattern matching
3. **Objective Generation**: Generate game objectives and natural language prompts
4. **Frame Extraction**: Sample frames from the video for processing
5. **Universal Detection**: Run Grounded SAM 2 with combined prompts on each frame
6. **Categorization**: Automatically categorize detected objects into game-relevant types
7. **Segmentation**: Extract precise object cutouts with transparency

## Usage

### Command Line

```bash
# Analyze any game session with universal pipeline
python -m src.analysis.video_behavior_analysis.run_improved_analysis \
    data/GameName/session_folder --detector universal

# Or run the pipeline directly
python -m src.analysis.video_behavior_analysis.universal_grounded_sam2_pipeline \
    data/GameName/session_folder
```

### Interactive Mode

```bash
python -m src.analysis.video_behavior_analysis.run_improved_analysis
# Select option [3] Universal Pipeline
```

## Key Advantages

### ✅ Scalable
- Works with **any mobile game** without pre-configuration
- No need to manually define game-specific object lists
- Automatically adapts to new games

### ✅ Intelligent
- Uses Gemini's video understanding to discover objects
- Combines AI analysis with computer vision detection
- Learns object names from behavioral descriptions

### ✅ Comprehensive
- Detects all types of objects: characters, obstacles, collectibles, UI elements
- Provides precise segmentation masks for each object
- Categorizes objects automatically
- Generates game objectives and natural language prompts

### ✅ Future-Proof
- No hardcoded game knowledge required
- Easily extensible to new game genres
- Maintains compatibility with existing workflows

## Output Structure

```
session_folder/
├── analysis/
│   ├── universal_grounded_sam2/
│   │   ├── frame_detections/           # Per-frame detection results
│   │   │   ├── frame_0000/
│   │   │   │   ├── grounded_sam2_annotated_image_with_mask.jpg
│   │   │   │   └── grounded_sam2_custom_demo_results.json
│   │   │   └── frame_0001/
│   │   │       └── ...
│   │   └── universal_analysis_summary.json  # Comprehensive results
│   └── video_behavior_analysis/
│       ├── detailed_analysis_single_session_20241225_123456.txt
│       ├── extracted_objects_20241225_123456.json
│       └── objectives/                 # Generated objectives
│           ├── objectives_gamename_20241225_123456.md
│           └── objective_prompt_gamename_20241225_123456.txt
└── frames/                             # Extracted frames
```

## Configuration

### Default Parameters
- **Max Frames**: 100 (adjustable with `--max-frames`)
- **Frame Interval**: 30 (every 30th frame, adjustable with `--frame-interval`)
- **Detection Thresholds**: 
  - Box threshold: 0.25
  - Text threshold: 0.20

### Customization

```bash
# Analyze more frames with higher frequency
python universal_grounded_sam2_pipeline.py session_folder \
    --max-frames 200 --frame-interval 15

# Skip objective generation for faster processing
python universal_grounded_sam2_pipeline.py session_folder --no-objectives
```

## Technical Details

### Object Categories
- **Player**: Main character, avatar, hero
- **Enemies**: Opponents, monsters, zombies
- **Obstacles**: Barriers, walls, vehicles, hazards
- **Collectibles**: Coins, gems, powerups, items
- **UI**: Buttons, scores, health bars, menus
- **Environment**: Background elements, platforms, decorations
- **Interactive**: Doors, switches, tools, weapons

### Universal Prompts
The system uses a curated list of universal game object prompts that work across genres:
- Characters and entities
- Interactive objects  
- Environment elements
- UI components

### Gemini Integration
- Analyzes video content with behavioral focus
- Extracts object names using pattern matching
- Creates dynamic detection prompts
- Saves extracted objects for reuse

## Comparison with Game-Specific Approach

| Aspect | Game-Specific | Universal |
|--------|---------------|-----------|
| **Setup** | Manual prompt engineering | Automatic |
| **Scalability** | One game at a time | Any game |
| **Accuracy** | High for known games | Good for all games |
| **Maintenance** | Requires updates per game | Self-adapting |
| **Coverage** | Limited to pre-defined objects | Discovers new objects |

## Requirements

- Gemini API access (for video analysis)
- Grounded SAM 2 models downloaded
- OpenCV for frame extraction
- Standard detection dependencies

## Future Enhancements

1. **Frame-Specific Analysis**: Adapt prompts based on individual frame content
2. **Temporal Tracking**: Track objects across frames for behavior analysis
3. **Multi-Game Learning**: Build knowledge base from multiple game analyses
4. **Real-Time Detection**: Optimize for live game analysis

## Troubleshooting

### Common Issues

1. **No objects detected**: 
   - Check if Grounded SAM 2 models are downloaded
   - Verify video quality and frame extraction
   - Try lowering detection thresholds

2. **Gemini analysis fails**:
   - Verify Gemini API key configuration
   - Check video upload size limits
   - Ensure video format compatibility

3. **Performance issues**:
   - Reduce max frames or increase frame interval
   - Use GPU acceleration if available
   - Monitor memory usage during processing

### Getting Help

For issues or questions:
1. Check the main README for setup instructions
2. Verify all dependencies are installed
3. Test with a known working session first
4. Review error logs for specific issues

---

**🎮 The Universal Pipeline makes game analysis truly scalable - no more manual configuration for each new game!** 