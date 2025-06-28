"""
YOLO detector wrapper for compatibility with the detector abstraction.

This module wraps the existing YOLOv11 implementation to work with the
BaseDetector interface, allowing hot-swapping with other detectors.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import sieve
from PIL import Image
from ultralytics import YOLO

from .base_detector import BaseDetector, Detection, DetectionConfig


class YOLODetector(BaseDetector):
    """YOLO detector wrapper implementing the BaseDetector interface."""
    
    def __init__(self, 
                 config: Optional[DetectionConfig] = None,
                 model_name: str = "yolo11m.pt",
                 use_sam_segmentation: bool = True):
        """
        Initialize YOLO detector.
        
        Args:
            config: Detection configuration
            model_name: YOLO model name/path
            use_sam_segmentation: Whether to use SAM for segmentation
        """
        super().__init__(config)
        
        self.model_name = model_name
        self.use_sam_segmentation = use_sam_segmentation
        self.yolo_model = None
        self.sam_model = None
        
        # YOLO-specific parameters
        self.iou_threshold = config.iou_threshold if config else 0.6
        
    async def initialize(self) -> None:
        """Initialize the YOLO model and SAM if needed."""
        if self.is_initialized:
            return
            
        print(f"🔧 Initializing YOLO detector with model: {self.model_name}")
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO(self.model_name)
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
            
        # Initialize SAM if requested
        if self.use_sam_segmentation:
            try:
                import os
                if os.getenv("SIEVE_API_KEY"):
                    sieve.api_key = os.getenv("SIEVE_API_KEY")
                    self.sam_model = sieve.function.get("sieve/sam2")
                    print("✅ SAM segmentation initialized")
                else:
                    print("⚠️ SIEVE_API_KEY not found, SAM segmentation disabled")
                    self.use_sam_segmentation = False
            except Exception as e:
                print(f"⚠️ Failed to initialize SAM: {e}")
                self.use_sam_segmentation = False
                
        self.is_initialized = True
    
    async def detect(self, 
                    image: Union[str, Path, Image.Image, np.ndarray],
                    prompts: Optional[List[str]] = None,
                    **kwargs) -> List[Detection]:
        """
        Detect objects using YOLO.
        
        Args:
            image: Input image
            prompts: Text prompts (ignored for YOLO, used for filtering)
            **kwargs: Additional parameters
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Prepare image
        cv_image = self._prepare_image_for_yolo(image)
        image_path = self._prepare_image_path(image)
        
        # Run YOLO detection
        results = self.yolo_model(
            cv_image,
            conf=self.config.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Convert YOLO results to Detection objects
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extract detection info
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]
                
                # Create detection
                detection = Detection(
                    bbox=box.tolist(),
                    confidence=conf,
                    class_name=cls_name,
                    class_id=cls_id
                )
                
                detections.append(detection)
        
        # Add segmentation masks if SAM is available
        if self.use_sam_segmentation and self.sam_model and detections and image_path:
            detections = await self._add_sam_segmentation(image_path, detections)
        
        # Apply filters
        filtered_detections = self.filter_detections(detections)
        
        # Filter by prompts if provided
        if prompts:
            filtered_detections = self._filter_by_prompts(filtered_detections, prompts)
            
        return filtered_detections
    
    def _prepare_image_for_yolo(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Convert image to format suitable for YOLO."""
        if isinstance(image, (str, Path)):
            return cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _prepare_image_path(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[str]:
        """Get image path for SAM processing."""
        if isinstance(image, (str, Path)):
            return str(image)
        # For other types, we'd need to save to temp file for SAM
        # For now, return None to skip SAM segmentation
        return None
    
    async def _add_sam_segmentation(self, image_path: str, detections: List[Detection]) -> List[Detection]:
        """Add SAM segmentation masks to detections."""
        if not detections:
            return detections
            
        try:
            # Create Sieve File object
            image_file = sieve.File(path=image_path)
            
            # Prepare SAM prompts from YOLO detections
            sam_prompts = []
            for i, det in enumerate(detections):
                bbox = det.bbox
                sam_prompt = {
                    "object_id": i + 1,
                    "frame_index": 0,
                    "box": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                }
                sam_prompts.append(sam_prompt)
            
            # Run SAM segmentation
            sam_job = self.sam_model.push(
                file=image_file,
                prompts=sam_prompts,
                model_type="large"
            )
            
            sam_result = sam_job.result()
            
            # Add masks to detections
            if isinstance(sam_result, tuple) and len(sam_result) >= 2:
                masks_dict = sam_result[1]
                
                for i, (mask_name, mask_file) in enumerate(masks_dict.items()):
                    if i < len(detections):
                        # Load mask
                        mask = cv2.imread(mask_file.path, cv2.IMREAD_GRAYSCALE)
                        detections[i].mask = mask
                        
        except Exception as e:
            print(f"⚠️ SAM segmentation failed: {e}")
            
        return detections
    
    def _filter_by_prompts(self, detections: List[Detection], prompts: List[str]) -> List[Detection]:
        """Filter detections based on text prompts."""
        if not prompts:
            return detections
            
        # Convert prompts to lowercase for matching
        prompt_keywords = []
        for prompt in prompts:
            prompt_keywords.extend(prompt.lower().split())
        
        filtered = []
        for det in detections:
            class_name = det.class_name.lower()
            
            # Check if class name matches any prompt keyword
            for keyword in prompt_keywords:
                if keyword in class_name or class_name in keyword:
                    filtered.append(det)
                    break
                    
        return filtered
    
    def get_supported_features(self) -> Dict[str, bool]:
        """Get supported features of this detector."""
        return {
            'text_prompts': False,  # YOLO doesn't support text prompts natively
            'segmentation': self.use_sam_segmentation,
            'tracking': False,
            'batch_processing': True
        }
    
    def set_yolo_parameters(self, 
                           confidence_threshold: Optional[float] = None,
                           iou_threshold: Optional[float] = None):
        """Update YOLO-specific parameters."""
        if confidence_threshold is not None:
            self.config.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
    
    async def detect_batch(self, 
                          images: List[Union[str, Path, Image.Image, np.ndarray]],
                          prompts: Optional[List[str]] = None,
                          **kwargs) -> List[List[Detection]]:
        """
        Detect objects in multiple images.
        
        Args:
            images: List of input images
            prompts: Text prompts for filtering
            **kwargs: Additional parameters
            
        Returns:
            List of detection lists (one per image)
        """
        tasks = []
        for image in images:
            task = self.detect(image, prompts, **kwargs)
            tasks.append(task)
            
        return await asyncio.gather(*tasks)


# Convenience function for quick setup
async def create_yolo_detector(model_name: str = "yolo11m.pt",
                              confidence_threshold: float = 0.45,
                              use_sam_segmentation: bool = True) -> YOLODetector:
    """
    Create and initialize a YOLO detector.
    
    Args:
        model_name: YOLO model name/path
        confidence_threshold: Detection confidence threshold
        use_sam_segmentation: Whether to use SAM for segmentation
        
    Returns:
        Initialized detector
    """
    config = DetectionConfig(
        confidence_threshold=confidence_threshold,
        enable_segmentation=use_sam_segmentation
    )
    
    detector = YOLODetector(config, model_name, use_sam_segmentation)
    await detector.initialize()
    
    return detector 