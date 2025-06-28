# Grounded-SAM-2 Setup Guide

This guide explains how to set up the official [Grounded-SAM-2 repository](https://github.com/IDEA-Research/Grounded-SAM-2) for use with our improved game video analysis system.

## Quick Setup

### 1. Clone the Repository

```bash
# Clone the official repository
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```

### 2. Install Dependencies

```bash
# Install the required packages
pip install -r requirements.txt

# Fix missing dependencies that may not be in requirements.txt
pip install hydra-core omegaconf supervision

# Additional packages that might be needed
pip install matplotlib seaborn
```

#### Mac-Specific Setup (Apple Silicon)

For Apple Silicon Macs (M1/M2/M3), ensure you have the MPS-enabled PyTorch:

```bash
# Verify you have MPS-enabled PyTorch (should show True if available)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# If MPS is not available, install/upgrade PyTorch
pip install torch torchvision torchaudio

# Additional Mac-specific dependencies
pip install opencv-python-headless  # Better compatibility on macOS
```

### 3. Download Models

**Important:** The repository doesn't include an automatic download script, so you need to manually download the required models:

#### Download Required Models

**For Mac (using curl instead of wget):**
```zsh
# Create directories
mkdir -p checkpoints gdino_checkpoints

# Download SAM 2 models
cd checkpoints
curl -L -o sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
curl -L -o sam2_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
cd ..

# Download Grounding DINO models  
cd gdino_checkpoints
curl -L -o groundingdino_swinb_cogcoor.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
```

**For Linux/WSL (using wget):**
```bash
# Create directories
mkdir -p checkpoints gdino_checkpoints

# Download SAM 2 models
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
cd ..

# Download Grounding DINO models  
cd gdino_checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
```

### 4. Verify Installation

Test the installation with a simple image:

```bash
python grounded_sam2_florence2_image_demo.py \
    --pipeline open_vocabulary_detection_segmentation \
    --image_path ./demo_images/cars.jpg \
    --text_input "car <and> building"
```

## Integration with Our System

### Path Configuration

Our system will automatically search for the Grounded-SAM-2 repository in these locations:

1. `./Grounded-SAM-2` (current directory)
2. `./grounded-sam-2` (lowercase variant)
3. `~/Grounded-SAM-2` (home directory)
4. `/tmp/Grounded-SAM-2` (temporary directory)

### Manual Path Specification

If you installed the repository in a different location, you can specify it:

```python
from src.analysis.video_behavior_analysis.grounded_sam_detector import GroundedSAMDetector

# Specify custom path
detector = GroundedSAMDetector(grounded_sam_path="/path/to/your/Grounded-SAM-2")
```

## Usage Examples

### Basic Detection

```python
from src.analysis.video_behavior_analysis.grounded_sam_detector import create_grounded_sam_detector

# Initialize detector
detector = await create_grounded_sam_detector()

# Detect objects with text prompts
prompts = ["character", "coin", "obstacle"]
detections = await detector.detect("game_screenshot.png", prompts=prompts)
```

### Game-Specific Analysis

```python
from src.analysis.video_behavior_analysis.improved_analyzer import ImprovedAnalyzer

# Analyze with game-specific prompts
analyzer = ImprovedAnalyzer("path/to/session", detector_type="grounded_sam")
results = await analyzer.analyze()
```

### Detector Comparison

```python
from src.analysis.video_behavior_analysis.detector_evaluation import DetectorEvaluator

# Compare YOLO vs Grounding DINO performance
evaluator = DetectorEvaluator(test_images_dir, "subway_surfers")
results = await evaluator.run_comparison(output_dir)
```

## Troubleshooting

### Common Issues

#### 1. Repository Not Found
```
FileNotFoundError: Grounded-SAM-2 repository not found
```

**Solution:** Clone the repository to one of the expected locations or specify the path manually.

#### 2. Missing Models
```
FileNotFoundError: Missing required items in Grounded-SAM-2 repository: ['checkpoints']
```

**Solution:** Manually download the required models using the commands provided in the setup section.

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'groundingdino'
```

**Solution:** Install missing dependencies:
```bash
pip install -r requirements.txt
pip install groundingdino-py
```

#### 4. GPU Acceleration Issues

**CUDA Issues (Linux/Windows with NVIDIA GPUs):**
```
RuntimeError: No CUDA GPUs are available
```

**Apple Silicon Mac (M1/M2/M3) - MPS Acceleration:**
```
RuntimeError: MPS backend out of memory
```

**Solutions:**
- **Apple Silicon Macs:** The system will automatically use MPS (Metal Performance Shaders) for acceleration. If you get memory issues, reduce batch size or use smaller models.
- **NVIDIA GPUs:** Ensure CUDA is properly installed and compatible with your PyTorch version.
- **CPU Fallback:** The system will fall back to CPU mode if neither MPS nor CUDA is available, but performance will be slower.

### Performance Optimization

#### GPU Memory
If you encounter GPU memory issues, try:

```python
# Use smaller SAM model
detector.model_type = "base"  # instead of "large"

# Lower confidence thresholds to reduce detections
detector.box_threshold = 0.5
detector.text_threshold = 0.3
```

#### Processing Speed
For faster processing:

```python
# Process fewer frames
analyzer.max_frames = 50

# Use longer frame intervals
analyzer.frame_interval = 60  # Every 2 seconds at 30fps
```

## Available Models

### SAM 2 Models
- `sam2_hiera_large.pt` - Best quality, slower
- `sam2_hiera_base_plus.pt` - Good balance of speed/quality
- `sam2_hiera_small.pt` - Fastest, lower quality

### Grounding DINO Models
- `groundingdino_swinb_cogcoor.pth` - Standard model
- `groundingdino_swint_ogc.pth` - Alternative variant

## Advanced Configuration

### Custom Detection Parameters

```python
detector.set_detection_parameters(
    box_threshold=0.35,      # Bounding box confidence threshold
    text_threshold=0.25,     # Text matching threshold  
    model_type="large"       # SAM model size
)
```

### Game-Specific Prompts

Add new games to `game_prompts.py`:

```python
"my_game": GamePromptConfig(
    primary_prompts=["player", "enemy", "collectible"],
    secondary_prompts=["platform", "background"],
    ui_prompts=["button", "score"],
    background_prompts=["scenery"],
    confidence_thresholds={"player": 0.3, "enemy": 0.4},
    size_filters={"collectible": {"min_size": 10, "max_size": 50}}
)
```

## System Requirements

### Minimum Requirements
- **Python 3.11-3.12** (avoid 3.13 due to compatibility issues)
- 8GB RAM
- 4GB free disk space

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- **NVIDIA GPU with 8GB+ VRAM** OR **Apple Silicon Mac (M1 Pro/Max, M2, M3) with 16GB+ unified memory**
- 10GB free disk space

### Performance Expectations by Platform
- **Apple Silicon M3/M2 Pro/Max:** ~1-2 FPS with Grounding DINO + SAM
- **Apple Silicon M1/M2 Base:** ~0.5-1 FPS with Grounding DINO + SAM  
- **NVIDIA RTX 3080/4080:** ~2-3 FPS with Grounding DINO + SAM
- **CPU Only (any platform):** ~0.1-0.3 FPS with Grounding DINO + SAM

## Support

If you encounter issues:

1. Check the [official Grounded-SAM-2 repository](https://github.com/IDEA-Research/Grounded-SAM-2) for updates
2. Verify all dependencies are installed correctly
3. Ensure models are downloaded and in the correct directories
4. Check GPU drivers if using CUDA acceleration

For game analysis specific issues, refer to our detector evaluation tools to compare performance and identify optimal settings for your use case.