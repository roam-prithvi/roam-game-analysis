"""
Grounding DINO + SAM 2 detector implementation.

This module integrates the official Grounded-SAM-2 repository to provide
open-vocabulary object detection and segmentation for game analysis.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch

from .base_detector import BaseDetector, Detection, DetectionConfig
from .game_prompts import get_grounding_prompt_for_game


class GroundedSAMDetector(BaseDetector):
    """Detector using Grounding DINO + SAM 2 for open-vocabulary detection."""
    
    def __init__(self, 
                 config: Optional[DetectionConfig] = None,
                 grounded_sam_path: Optional[str] = None,
                 detection_mode: str = "open_vocabulary"):
        """
        Initialize Grounded SAM detector.
        
        Args:
            config: Detection configuration
            grounded_sam_path: Path to Grounded-SAM-2 repository
            detection_mode: Detection mode ('open_vocabulary', 'phrase_grounding', etc.)
        """
        super().__init__(config)
        
        # Set up paths
        self.grounded_sam_path = Path(grounded_sam_path) if grounded_sam_path else self._find_grounded_sam_repo()
        self.detection_mode = detection_mode
        self.temp_dir = Path(tempfile.mkdtemp(prefix="grounded_sam_"))
        
        # Detection parameters
        self.box_threshold = 0.25
        self.text_threshold = 0.15
        self.model_type = "large"  # SAM model type
        
        # Cache for repeated prompts
        self._prompt_cache: Dict[str, List[Detection]] = {}
        
    def _find_grounded_sam_repo(self) -> Path:
        """Try to find Grounded-SAM-2 repository in common locations."""
        search_paths = [
            Path.cwd() / "Grounded-SAM-2",
            Path.cwd() / "grounded-sam-2", 
            Path.home() / "Grounded-SAM-2",
            Path("/tmp/Grounded-SAM-2")
        ]
        
        for path in search_paths:
            if path.exists() and (path / "grounded_sam2_hf_model_demo.py").exists():
                return path
                
        # Repository not found
        raise FileNotFoundError(
            "Grounded-SAM-2 repository not found. Please:\n"
            "1. Clone it: git clone https://github.com/IDEA-Research/Grounded-SAM-2.git\n"
            "2. Install dependencies: cd Grounded-SAM-2 && pip install -r requirements.txt\n"
            "3. Download models: bash download_models.sh\n"
            f"Or specify the path manually with grounded_sam_path parameter."
        )
    
    async def initialize(self) -> None:
        """Initialize the detector by checking dependencies."""
        if self.is_initialized:
            return
            
        print(f"🔧 Initializing Grounded SAM detector...")
        print(f"📁 Repository path: {self.grounded_sam_path}")
        
        # Check if repository exists
        if not self.grounded_sam_path.exists():
            raise FileNotFoundError(f"Grounded-SAM-2 repository not found at: {self.grounded_sam_path}")
            
        # Check if required scripts exist
        required_scripts = [
            "grounded_sam2_hf_model_demo.py",
            "checkpoints",
            "gdino_checkpoints"
        ]
        
        missing_items = []
        for item in required_scripts:
            if not (self.grounded_sam_path / item).exists():
                missing_items.append(item)
                
        if missing_items:
            raise FileNotFoundError(
                f"Missing required items in Grounded-SAM-2 repository: {missing_items}\n"
                "Please ensure the repository is properly set up with models downloaded."
            )
        
        # Test run with a simple image to warm up models
        try:
            test_image = self._create_test_image()
            await self._run_detection(test_image, "test object", warm_up=True)
            print("✅ Grounded SAM detector initialized successfully")
        except Exception as e:
            print(f"⚠️ Detector initialization test failed: {e}")
            # Continue anyway, as it might work with real images
            
        self.is_initialized = True
    
    def _create_test_image(self) -> str:
        """Create a simple test image for initialization."""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [255, 255, 255]  # White square
        
        test_path = str(self.temp_dir / "test_image.jpg")
        cv2.imwrite(test_path, test_img)
        return test_path
    
    async def detect(self, 
                    image: Union[str, Path, Image.Image, np.ndarray],
                    prompts: Optional[List[str]] = None,
                    game_name: Optional[str] = None,
                    **kwargs) -> List[Detection]:
        """
        Detect objects using Grounding DINO + SAM 2.
        
        Args:
            image: Input image
            prompts: Text prompts for detection
            game_name: Game name to use game-specific prompts
            **kwargs: Additional parameters
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Prepare image path
        image_path = self._prepare_image(image)
        
        # Prepare prompts
        if prompts:
            prompt_text = " <and> ".join(prompts[:20])  # Limit prompts
        elif game_name:
            prompt_text = get_grounding_prompt_for_game(game_name, primary_only=True)
        else:
            prompt_text = "character <and> obstacle <and> collectible <and> enemy"
            
        if not prompt_text:
            return []
            
        # Check cache
        cache_key = f"{image_path}:{prompt_text}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
            
        # Run detection
        detections = await self._run_detection(image_path, prompt_text)
        
        # Apply filters
        filtered_detections = self.filter_detections(detections)
        
        # Cache results
        self._prompt_cache[cache_key] = filtered_detections
        
        return filtered_detections
    
    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """Convert image to file path for processing."""
        if isinstance(image, (str, Path)):
            # Convert to absolute path to avoid issues with working directory
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            return str(image_path.resolve())
        
        # Convert other formats to temporary file
        temp_path = str(self.temp_dir / f"temp_image_{id(image)}.jpg")
        
        if isinstance(image, Image.Image):
            image.save(temp_path)
        elif isinstance(image, np.ndarray):
            # Assume BGR format from OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                cv2.imwrite(temp_path, image)
            else:
                # Convert grayscale or other formats
                cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        return temp_path
    
    async def _run_detection(self, 
                           image_path: str, 
                           prompt_text: str, 
                           warm_up: bool = False) -> List[Detection]:
        """Run the actual detection using Grounded-SAM-2 HF model demo script."""
        output_dir = self.temp_dir / f"output_{id(image_path)}_{id(prompt_text)}"
        output_dir.mkdir(exist_ok=True)
        
        # Use our custom demo script that works with available checkpoints
        script_path = self.grounded_sam_path / "grounded_sam2_custom_demo.py"
        
        # Ensure text prompt ends with period (required by Grounding DINO)
        if not prompt_text.strip().endswith('.'):
            prompt_text = prompt_text.strip() + '.'
        
        cmd = [
            sys.executable, str(script_path),
            "--text-prompt", prompt_text,
            "--img-path", image_path,
            "--output-dir", str(output_dir),
            "--sam2-checkpoint", "./checkpoints/sam2_hiera_large.pt",  # Use available checkpoint
            "--sam2-model-config", "sam2_hiera_l.yaml",  # Use just filename for Hydra
            "--box-threshold", str(self.box_threshold),
            "--text-threshold", str(self.text_threshold)
        ]
        
        # Add CPU flag if no CUDA available
        if not torch.cuda.is_available():
            cmd.append("--force-cpu")
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.grounded_sam_path)
        
        try:
            # Run detection
            if warm_up:
                # For warm-up, run with minimal output and suppress JSON dump
                cmd.append("--no-dump-json")
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.grounded_sam_path,
                    env=env,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await result.wait()
                return []
            else:
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.grounded_sam_path,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    print(f"⚠️ Grounded SAM detection failed: {error_msg}")
                    return []
                    
            # Parse results from the JSON output
            return self._parse_detection_results(output_dir, image_path)
            
        except Exception as e:
            print(f"⚠️ Error running Grounded SAM detection: {e}")
            return []
    
    def _parse_detection_results(self, output_dir: Path, image_path: str) -> List[Detection]:
        """Parse detection results from Grounded-SAM-2 HF model demo JSON output."""
        detections = []
        
        try:
            # Look for the specific JSON output file from custom demo script
            json_file = output_dir / "grounded_sam2_custom_demo_results.json"
            
            if json_file.exists():
                # Parse JSON annotations
                import json
                with open(json_file, 'r') as f:
                    results = json.load(f)
                    
                # Extract annotations
                annotations = results.get('annotations', [])
                
                for ann in annotations:
                    # Parse bounding box (should be in xyxy format)
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    
                    # Handle confidence that might be in different formats
                    confidence_raw = ann.get('score', 0.5)
                    if isinstance(confidence_raw, list):
                        confidence = float(confidence_raw[0]) if confidence_raw else 0.5
                    else:
                        confidence = float(confidence_raw)
                        
                    class_name = ann.get('class_name', 'unknown')
                    
                    # Clean up class name - remove angle brackets and strip whitespace
                    class_name = class_name.strip()
                    if class_name.startswith('>') and class_name.endswith('<'):
                        class_name = class_name[1:-1].strip()
                    elif class_name.startswith('>'):
                        class_name = class_name[1:].strip()
                    elif class_name.endswith('<'):
                        class_name = class_name[:-1].strip()
                    
                    # Ensure we have a valid class name
                    if not class_name:
                        class_name = 'unknown'
                    
                    # Handle segmentation mask if available
                    mask = None
                    segmentation = ann.get('segmentation')
                    if segmentation:
                        # Convert RLE to mask if needed
                        mask = self._rle_to_mask(segmentation, results.get('img_height', 480), results.get('img_width', 640))
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=class_name,
                        mask=mask
                    )
                    detections.append(detection)
                    
            else:
                # Fallback: Look for other possible output files
                annotation_files = list(output_dir.glob("*.json"))
                mask_files = list(output_dir.glob("*mask*.png"))
                
                if annotation_files:
                    # Try to parse any JSON file found
                    with open(annotation_files[0], 'r') as f:
                        data = json.load(f)
                        
                    # Handle different JSON formats
                    if isinstance(data, dict) and 'annotations' in data:
                        # Standard format
                        for ann in data['annotations']:
                            # Handle confidence type conversion
                            confidence_raw = ann.get('score', 0.5)
                            if isinstance(confidence_raw, list):
                                confidence = float(confidence_raw[0]) if confidence_raw else 0.5
                            else:
                                confidence = float(confidence_raw)
                                
                            class_name = ann.get('class_name', 'unknown')
                            
                            # Clean up class name - remove angle brackets and strip whitespace
                            class_name = class_name.strip()
                            if class_name.startswith('>') and class_name.endswith('<'):
                                class_name = class_name[1:-1].strip()
                            elif class_name.startswith('>'):
                                class_name = class_name[1:].strip()
                            elif class_name.endswith('<'):
                                class_name = class_name[:-1].strip()
                            
                            # Ensure we have a valid class name
                            if not class_name:
                                class_name = 'unknown'
                                
                            detection = Detection(
                                bbox=ann.get('bbox', [0, 0, 0, 0]),
                                confidence=confidence,
                                class_name=class_name
                            )
                            detections.append(detection)
                    elif isinstance(data, list):
                        # List of annotations
                        for ann in data:
                            # Handle confidence type conversion
                            confidence_raw = ann.get('score', 0.5)
                            if isinstance(confidence_raw, list):
                                confidence = float(confidence_raw[0]) if confidence_raw else 0.5
                            else:
                                confidence = float(confidence_raw)
                                
                            class_name = ann.get('label', 'unknown')
                            
                            # Clean up class name - remove angle brackets and strip whitespace
                            class_name = class_name.strip()
                            if class_name.startswith('>') and class_name.endswith('<'):
                                class_name = class_name[1:-1].strip()
                            elif class_name.startswith('>'):
                                class_name = class_name[1:].strip()
                            elif class_name.endswith('<'):
                                class_name = class_name[:-1].strip()
                            
                            # Ensure we have a valid class name
                            if not class_name:
                                class_name = 'unknown'
                                
                            detection = Detection(
                                bbox=ann.get('bbox', [0, 0, 0, 0]),
                                confidence=confidence,
                                class_name=class_name
                            )
                            detections.append(detection)
                            
                elif mask_files:
                    # If no JSON but masks exist, create basic detections from masks
                    for i, mask_file in enumerate(mask_files):
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Calculate bounding box from mask
                            bbox = self._mask_to_bbox(mask)
                            
                            detection = Detection(
                                bbox=bbox,
                                confidence=0.5,  # Default confidence
                                class_name=f"object_{i}",
                                mask=mask
                            )
                            detections.append(detection)
                            
        except Exception as e:
            print(f"⚠️ Error parsing detection results: {e}")
            
        return detections
    
    def _rle_to_mask(self, rle_data: dict, height: int, width: int) -> Optional[np.ndarray]:
        """Convert RLE segmentation to binary mask."""
        try:
            import pycocotools.mask as mask_util
            
            # Ensure RLE format is correct
            if isinstance(rle_data, dict) and 'counts' in rle_data and 'size' in rle_data:
                # Standard RLE format
                mask = mask_util.decode(rle_data)
                print(f"🔧 RLE mask decoded - shape: {mask.shape}, dtype: {mask.dtype}, range: {mask.min()}-{mask.max()}")
                
                # Ensure mask is in proper format (0-255 for binary)
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8) * 255
                elif mask.max() <= 1 and mask.dtype != np.uint8:
                    # Convert 0-1 float to 0-255 uint8
                    mask = (mask * 255).astype(np.uint8)
                
                print(f"🔧 Final mask - shape: {mask.shape}, dtype: {mask.dtype}, range: {mask.min()}-{mask.max()}")
                return mask
            else:
                print(f"⚠️ Unknown RLE format: {type(rle_data)}")
                return None
                
        except ImportError:
            print("⚠️ pycocotools not available for mask decoding")
            return None
        except Exception as e:
            print(f"⚠️ Error decoding RLE mask: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_mask_if_exists(self, mask_files: List[Path], index: int) -> Optional[np.ndarray]:
        """Load mask file if it exists for this detection."""
        if index < len(mask_files):
            try:
                mask = cv2.imread(str(mask_files[index]), cv2.IMREAD_GRAYSCALE)
                return mask
            except Exception:
                pass
        return None
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[float]:
        """Convert binary mask to bounding box."""
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return [0, 0, 0, 0]
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return [float(x_min), float(y_min), float(x_max), float(y_max)]
    
    def get_supported_features(self) -> Dict[str, bool]:
        """Get supported features of this detector."""
        return {
            'text_prompts': True,
            'segmentation': True,
            'tracking': False,
            'batch_processing': True
        }
    
    def set_detection_parameters(self, 
                               box_threshold: Optional[float] = None,
                               text_threshold: Optional[float] = None,
                               model_type: Optional[str] = None):
        """Update detection parameters."""
        if box_threshold is not None:
            self.box_threshold = box_threshold
        if text_threshold is not None:
            self.text_threshold = text_threshold
        if model_type is not None:
            self.model_type = model_type
            
        # Clear cache when parameters change
        self._prompt_cache.clear()
    
    async def detect_batch(self, 
                          images: List[Union[str, Path, Image.Image, np.ndarray]],
                          prompts: Optional[List[str]] = None,
                          game_name: Optional[str] = None,
                          **kwargs) -> List[List[Detection]]:
        """
        Detect objects in multiple images efficiently.
        
        Args:
            images: List of input images
            prompts: Text prompts for detection
            game_name: Game name for game-specific prompts
            **kwargs: Additional parameters
            
        Returns:
            List of detection lists (one per image)
        """
        # For now, process individually
        # TODO: Implement true batch processing if Grounded-SAM-2 supports it
        tasks = []
        for image in images:
            task = self.detect(image, prompts, game_name, **kwargs)
            tasks.append(task)
            
        return await asyncio.gather(*tasks)
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"⚠️ Failed to cleanup temp directory: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()


# Convenience function for quick setup
async def create_grounded_sam_detector(grounded_sam_path: Optional[str] = None,
                                     confidence_threshold: float = 0.35,
                                     enable_segmentation: bool = True) -> GroundedSAMDetector:
    """
    Create and initialize a Grounded SAM detector.
    
    Args:
        grounded_sam_path: Path to Grounded-SAM-2 repository
        confidence_threshold: Detection confidence threshold
        enable_segmentation: Whether to enable segmentation
        
    Returns:
        Initialized detector
    """
    config = DetectionConfig(
        confidence_threshold=confidence_threshold,
        enable_segmentation=enable_segmentation
    )
    
    detector = GroundedSAMDetector(config, grounded_sam_path)
    await detector.initialize()
    
    return detector 