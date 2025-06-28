# Grounded DINO Detector Script Invocation Improvements

## Overview
Updated the `GroundedSAMDetector` to use the more reliable `grounded_sam2_hf_model_demo.py` script instead of the Florence-based demo, providing better stability and easier integration.

## Key Changes Made

### 1. Script Selection Improvement
**Before:** Used `grounded_sam2_florence2_image_demo.py`
**After:** Uses `grounded_sam2_hf_model_demo.py`

**Benefits:**
- ✅ More stable HuggingFace model integration
- ✅ Clearer command-line interface
- ✅ Standardized JSON output format
- ✅ Better error handling
- ✅ Automatic CPU/CUDA detection

### 2. Enhanced Command Construction
```python
# New command structure
cmd = [
    sys.executable, str(script_path),
    "--text-prompt", prompt_text,           # Clean prompt format
    "--img-path", image_path,               # Simple image path
    "--output-dir", str(output_dir),        # Organized output
    "--grounding-model", "IDEA-Research/grounding-dino-tiny",  # Faster inference
    "--sam2-checkpoint", str(checkpoint_path),
    "--sam2-model-config", str(config_path)
]

# Auto-detect device capabilities
if not torch.cuda.is_available():
    cmd.append("--force-cpu")

# Suppress JSON for warm-up runs
if warm_up:
    cmd.append("--no-dump-json")
```

### 3. Improved Result Parsing
The detector now properly parses the standardized JSON output:
```json
{
    "image_path": "path/to/image.jpg",
    "annotations": [
        {
            "class_name": "character",
            "bbox": [x1, y1, x2, y2],
            "segmentation": {"counts": "...", "size": [h, w]},
            "score": 0.85
        }
    ],
    "box_format": "xyxy",
    "img_width": 1920,
    "img_height": 1080
}
```

### 4. Robust Error Handling
- **Repository Detection:** Automatically finds Grounded-SAM-2 in common locations
- **Script Validation:** Checks for required scripts during initialization
- **Fallback Parsing:** Multiple JSON format support for different script versions
- **RLE Mask Decoding:** Converts segmentation masks using pycocotools
- **Graceful Degradation:** Works even if some components fail

### 5. Platform Compatibility
- **CPU Support:** Automatically detects and uses CPU when CUDA unavailable
- **Mac Support:** Works with Apple Silicon and Intel Macs
- **CUDA Support:** Optimized for GPU acceleration when available

## Usage Examples

### Basic Detection
```python
import asyncio
from src.analysis.video_behavior_analysis.grounded_sam_detector import create_grounded_sam_detector

async def detect_objects():
    detector = await create_grounded_sam_detector()
    
    detections = await detector.detect(
        "path/to/image.jpg",
        prompts=["character", "obstacle", "coin"]
    )
    
    for detection in detections:
        print(f"Found {detection.class_name} with confidence {detection.confidence:.2f}")
```

### Game-Specific Detection
```python
# Uses predefined game prompts
detections = await detector.detect(
    "game_screenshot.jpg",
    game_name="subway_surfers"
)
```

### Batch Processing
```python
image_paths = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
all_detections = await detector.detect_batch(
    image_paths,
    prompts=["player", "enemy", "collectible"]
)
```

## Testing
Run the test script to verify the improved integration:
```bash
python test_grounded_detector.py
```

## Benefits Over Previous Implementation

| Feature | Before | After |
|---------|--------|--------|
| **Script Reliability** | Florence demo (experimental) | HF model demo (stable) |
| **Command Interface** | Complex pipeline args | Simple, clear arguments |
| **JSON Output** | Non-standard format | Standardized annotations |
| **Error Handling** | Basic subprocess errors | Comprehensive error management |
| **Platform Support** | CUDA-only assumptions | Cross-platform compatibility |
| **Performance** | Uses large models | Optimized with tiny model option |

## Performance Expectations

| Platform | Expected FPS | Memory Usage |
|----------|-------------|--------------|
| NVIDIA RTX 3080/4080 | 2-3 FPS | ~4-6 GB VRAM |
| Apple M3 Pro/Max | 1-2 FPS | ~3-4 GB RAM |
| Apple M1/M2 Base | 0.5-1 FPS | ~2-3 GB RAM |
| CPU Only | 0.1-0.3 FPS | ~2-4 GB RAM |

## Future Improvements
- [ ] Implement true batch processing for multiple images
- [ ] Add support for video tracking with temporal consistency
- [ ] Optimize prompt caching for repeated game sessions
- [ ] Add support for custom model fine-tuning
- [ ] Implement model quantization for faster inference

## Troubleshooting

### Common Issues
1. **"Repository not found"** → Clone Grounded-SAM-2 repository
2. **"Models not downloaded"** → Follow INSTALL.md in the repository
3. **CUDA errors on Mac** → System automatically uses CPU mode
4. **Slow performance** → Consider using smaller SAM model or CPU optimization

### Dependencies
```bash
# Core dependencies
pip install torch torchvision transformers
pip install opencv-python pillow numpy
pip install pycocotools supervision

# For Grounded-SAM-2 repository
cd Grounded-SAM-2
pip install -r requirements.txt
``` 