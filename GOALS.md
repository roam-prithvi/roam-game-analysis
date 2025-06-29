# GOALS.md - Spatial Understanding Pipeline

## Vision

Build a spatial understanding pipeline that bridges 2D game analysis with 3D scene reconstruction, enabling AI systems to understand the 3D structure of mobile games from screen recordings.

## Current State

### What We Have

1. **2D Analysis Pipeline** (roam-game-analysis)
   - Extracts objects from game footage with bounding boxes
   - Identifies game elements (characters, coins, obstacles, UI)
   - Tracks player actions and touch events
   - Segments individual assets as transparent PNGs
   - Converts individual assets to 3D models via Meshy AI

2. **3D Visualization System** (Spatial-Reasoning)
   - Unity-compatible web renderer using Three.js
   - Accepts JSON scene descriptions with 3D positions
   - Supports primitive shapes with exact Unity coordinates
   - Provides interactive 3D scene exploration

### The Gap

No connection between 2D analysis and 3D visualization. When we detect a coin at pixel (300, 200), we cannot place it in 3D space.

## Core Challenge: 2D → 3D Reconstruction

### The Fundamental Problem

Recovering 3D information from 2D images is an under-constrained problem. A single 2D point could correspond to infinitely many 3D points along a ray from the camera.

### Why It's Solvable for Games

1. **Structured Environments**: Games have predictable layouts (lanes, grids, fixed camera angles)
2. **Repeated Patterns**: Same objects appear multiple times, enabling learning
3. **Known Constraints**: Game-specific rules limit possible 3D configurations
4. **Motion Cues**: Object movement across frames provides depth information

## Proposed Solution: Vision-Language Model Approach

### Core Insight

Rather than using traditional computer vision techniques, leverage a Vision-Language Model (Gemini 2.5 Pro) to directly understand and reconstruct 3D scenes from gameplay video.

### Key Components

1. **Fidelity Control**
   - Sliding scale for object tracking granularity
   - Low fidelity: Essential gameplay elements only
   - High fidelity: Include decorative and background elements
   - Adapts based on game complexity and requirements

2. **Game Type Awareness**
   - Game genre determines object importance
   - Specialized prompt templates per genre
   - Focus areas:
     - Platformers: Platforms, enemies, collectibles
     - Runners: Lanes, obstacles, power-ups
     - Tower Defense: Grid, towers, paths
     - Clickers: UI elements, interactive zones

3. **Incremental Scene Building**
   - Process 5-second video chunks with overlap
   - VLM incrementally refines Unity 3D JSON
   - Progressive enhancement approach
   - Maintains temporal consistency

### Processing Pipeline

1. **Input Preparation**
   - 5-second video chunks from gameplay
   - Optional: Existing 2D analysis metadata
   - Optional: Touch events and action logs

2. **VLM Processing** (Gemini 2.5 Pro)
   - Analyzes video with game-specific prompts
   - Generates/updates 3D scene representation
   - Places objects in Unity coordinate space
   - Handles occlusions and depth relationships

3. **Scene Export**
   - Outputs Unity-compatible JSON
   - Proper coordinate system (Y-up)
   - Appropriate primitive mappings
   - Consistent object naming

### Expected JSON Output Format

```json
{
  "scene": {
    "name": "Game_Level_1",
    "description": "Extracted 3D scene from gameplay"
  },
  "objects": [
    {
      "name": "Player_Character",
      "model": "Capsule",
      "position": { "x": 0, "y": 1, "z": 0 },
      "rotation": { "x": 0, "y": 0, "z": 0 },
      "scale": { "x": 1, "y": 2, "z": 1 },
      "color": "#FF0000",
      "text": "PLAYER"
    }
  ]
}
```

## Implementation Strategy

### Phase 1: Proof of Concept
- Single game type (Subway Surfers)
- Fixed fidelity level
- Basic prompt template
- Validate VLM can place objects in 3D

### Phase 2: Multi-Game Support
- Add prompt templates for different genres
- Implement fidelity control
- Test on varied game types

### Phase 3: Optimization
- Experiment with input combinations
- Tune sliding window parameters
- Improve temporal consistency

### Phase 4: Production Pipeline
- Integrate with existing 2D analysis
- Add quality validation
- Enable batch processing

## Success Metrics

1. **Spatial Accuracy**: Objects positioned correctly relative to each other
2. **Temporal Stability**: Consistent positions across video chunks
3. **Game Logic Preservation**: 3D scene maintains gameplay viability
4. **Processing Efficiency**: Reasonable processing time per video

## Technical Advantages

1. **No Camera Calibration Needed**: VLM infers spatial relationships directly
2. **Handles Occlusions Naturally**: VLM understands object permanence
3. **Game-Aware**: Leverages semantic understanding of gameplay
4. **Flexible Input**: Can incorporate various metadata sources

## Future Extensions

1. **Real-Time Processing**: Live 3D reconstruction during gameplay
2. **Multi-View Fusion**: Combine multiple gameplay videos
3. **Physics Inference**: Detect collision volumes and physics properties
4. **Complete Unity Export**: Generate full Unity projects with scripts

This VLM-based approach represents a paradigm shift from traditional computer vision, leveraging the semantic understanding capabilities of modern AI to solve the 2D→3D reconstruction challenge for games.