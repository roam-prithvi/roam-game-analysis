# Improved Game Video Analysis System

A comprehensive video analysis system with pluggable object detectors for game footage analysis. This system addresses the limitations of YOLO-only detection by providing open-vocabulary detection capabilities with Grounding DINO + SAM 2.

## 🚀 Key Features

### Detector Abstraction Layer
- **Hot-swappable detectors**: Easily switch between YOLO, Grounding DINO, and future detectors
- **Standardized interface**: Consistent API across all detection methods
- **Performance benchmarking**: Built-in tools to compare detector performance

### Game-Aware Detection
- **Game-specific prompts**: Curated text prompts for popular mobile games
- **Context-aware filtering**: Intelligent object categorization based on game context
- **Asset extraction**: Automated extraction of game assets with high-quality segmentation

### Advanced Analysis Pipeline
- **Multi-session analysis**: Process entire game directories efficiently
- **Comprehensive evaluation**: Detailed performance metrics and visual comparisons
- **Export capabilities**: Save results in multiple formats for further analysis

## 🏗️ Architecture

```
src/analysis/video_behavior_analysis/
├── base_detector.py              # Abstract detector interface
├── grounded_sam_detector.py      # Grounding DINO + SAM 2 implementation  
├── yolo_detector.py              # YOLO + SAM wrapper
├── game_prompts.py               # Game-specific prompt dictionaries
├── improved_analyzer.py          # Main analysis engine
├── detector_evaluation.py        # Evaluation and comparison tools
├── run_improved_analysis.py      # CLI interface
└── SETUP_GROUNDED_SAM.md        # Setup instructions
```

## 🎮 Supported Games

- Subway Surfers
- Temple Run
- Crossy Road
- Plants vs Zombies (1 & 2)
- Candy Crush Saga
- Angry Birds
- Clash Royale
- Pokemon GO
- Mario Kart Tour
- Generic runners, puzzles, and strategy games

## 🔧 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install ultralytics opencv-python pillow numpy matplotlib sieve
```

2. **Set up Grounding DINO + SAM 2** (for best results):
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
pip install -r requirements.txt
bash download_models.sh
```

3. **Configure API keys:**
```bash
export SIEVE_API_KEY="your_sieve_api_key"  # For SAM segmentation
```

### Basic Usage

#### Interactive Mode
```bash
python run_improved_analysis.py
```

#### Analyze Single Session
```bash
python run_improved_analysis.py path/to/session --detector grounded_sam
```

#### Analyze All Sessions in Game Directory
```bash
python run_improved_analysis.py path/to/game_dir --detector grounded_sam --all-sessions
```

#### Compare Detectors
```bash
python run_improved_analysis.py --compare test_images_dir --game subway_surfers
```

## 🎯 Detector Comparison

### YOLO + SAM
**Pros:**
- Fast processing (3-5 FPS)
- No external dependencies
- Good for general objects

**Cons:**
- Limited to 80 COCO classes
- Poor performance on game sprites
- Many false positives in game footage

### Grounding DINO + SAM 2
**Pros:**
- Open-vocabulary detection
- Game-specific text prompts
- Superior accuracy on game objects
- High-quality segmentation

**Cons:**
- Slower processing (0.5-1 FPS)
- Requires additional setup
- Higher memory usage

## 📊 Performance Results

Based on evaluation across multiple games:

| Metric | YOLO + SAM | Grounding DINO + SAM |
|--------|------------|---------------------|
| **Accuracy** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Game Relevance** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Complexity** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 🔍 Game-Specific Prompts

Example prompts for Subway Surfers:
```python
primary_prompts = [
    "character running", "train", "police officer", 
    "coin", "power up", "barrier", "obstacle"
]
```

The system automatically uses appropriate prompts based on detected game type, or you can specify custom prompts.

## 📈 Evaluation Tools

### Comprehensive Metrics
- Processing speed (FPS)
- Detection accuracy
- Confidence distributions
- Object categorization
- Memory usage

### Visual Comparisons
- Side-by-side detection results
- Performance charts
- Confidence histograms
- Processing time analysis

### Example Evaluation
```python
from detector_evaluation import DetectorEvaluator

evaluator = DetectorEvaluator(test_images_dir, "subway_surfers")
results = await evaluator.run_comparison(output_dir)
```

## 🎨 Asset Extraction

The system automatically extracts and categorizes game assets:

```
analysis/
├── grounded_sam_game_aware/
│   └── segmented_assets/
│       ├── player/           # Character sprites
│       ├── enemies/          # Enemy objects  
│       ├── collectibles/     # Coins, power-ups
│       ├── obstacles/        # Barriers, trains
│       └── ui/              # Interface elements
```

Each asset includes:
- High-quality segmentation mask
- Transparent PNG cutout
- Confidence score and metadata
- Frame timestamp and location

## 🔬 Advanced Usage

### Custom Detector Configuration
```python
from base_detector import DetectionConfig

config = DetectionConfig(
    confidence_threshold=0.35,
    max_detections=50,
    min_object_size=10,
    enable_segmentation=True
)

analyzer = ImprovedAnalyzer(session_path, "grounded_sam")
analyzer.detector_config = config
```

### Batch Processing
```python
# Process multiple sessions efficiently
sessions = list_sessions(game_directory)
results = await analyze_multiple_sessions(sessions, "grounded_sam")
```

### Custom Game Prompts
```python
# Add new game to game_prompts.py
"my_custom_game": GamePromptConfig(
    primary_prompts=["hero", "villain", "treasure"],
    secondary_prompts=["platform", "trap"],
    confidence_thresholds={"hero": 0.3},
    size_filters={"treasure": {"min_size": 15}}
)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Grounded-SAM-2 not found**
   - Ensure repository is cloned to expected location
   - Or specify path manually in detector initialization

2. **Poor detection performance**
   - Adjust confidence thresholds
   - Use game-specific prompts
   - Try different detector models

3. **Memory issues**
   - Reduce batch size
   - Use smaller SAM model
   - Process fewer frames

4. **Slow processing**
   - Use YOLO detector for speed
   - Increase frame sampling interval
   - Enable GPU acceleration

### Performance Optimization

```python
# For speed-critical applications
analyzer = ImprovedAnalyzer(session_path, "yolo")
await analyzer.analyze_video(max_frames=50, frame_interval=60)

# For accuracy-critical applications  
analyzer = ImprovedAnalyzer(session_path, "grounded_sam")
await analyzer.analyze_video(max_frames=200, frame_interval=15)
```

## 🔮 Future Enhancements

### Planned Features
- **Fine-tuned game models**: Custom YOLO models trained on game data
- **Temporal consistency**: Object tracking across frames
- **Behavior analysis**: Game mechanic extraction from object interactions
- **Real-time processing**: Live game analysis capabilities
- **Model quantization**: Optimized models for mobile/edge deployment

### Extension Points
- **Custom detectors**: Easy integration of new detection methods
- **Game plugins**: Modular game-specific analysis modules
- **Export formats**: Additional output formats (JSON, XML, CSV)
- **Cloud integration**: Distributed processing capabilities

## 📚 API Reference

### BaseDetector Interface
```python
class BaseDetector(ABC):
    async def initialize() -> None
    async def detect(image, prompts=None) -> List[Detection]
    def get_supported_features() -> Dict[str, bool]
    def filter_detections(detections) -> List[Detection]
```

### Detection Result Format
```python
@dataclass
class Detection:
    bbox: List[float]           # [x1, y1, x2, y2]
    confidence: float           # 0.0 to 1.0
    class_name: str            # Object class name
    class_id: Optional[int]    # Class ID if available
    mask: Optional[np.ndarray] # Segmentation mask
```

## 🤝 Contributing

1. **Add new games**: Extend `game_prompts.py` with game-specific prompts
2. **Improve detectors**: Enhance existing detector implementations
3. **Add evaluations**: Create new evaluation metrics and visualizations
4. **Optimize performance**: Implement caching, batching, and acceleration

## 📝 License

This project extends the existing game analysis system. Please refer to individual component licenses:
- Grounded-SAM-2: Apache 2.0 License
- YOLOv11: AGPL-3.0 License
- SAM 2: Apache 2.0 License

## 🙏 Acknowledgments

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for the excellent open-vocabulary detection framework
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11 implementation
- [Sieve](https://www.sievedata.com/) for SAM 2 API access 