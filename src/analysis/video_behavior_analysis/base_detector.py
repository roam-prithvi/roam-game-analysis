"""
Abstract base class for object detectors in game video analysis.

This module provides a standardized interface for different detection methods,
allowing easy hot-swapping between YOLO, Grounding DINO, and future detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class Detection:
    """Standardized detection result."""
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float  # Detection confidence (0.0 to 1.0)
    class_name: str    # Detected object class name
    class_id: Optional[int] = None  # Class ID if available
    mask: Optional[np.ndarray] = None  # Segmentation mask if available
    features: Optional[Dict[str, Any]] = None  # Additional features/metadata


@dataclass
class DetectionConfig:
    """Configuration for detection parameters."""
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.6
    max_detections: int = 100
    min_object_size: int = 10  # Minimum object size in pixels
    max_object_size: Optional[int] = None  # Maximum object size in pixels
    enable_segmentation: bool = True
    filter_classes: Optional[List[str]] = None  # Only detect these classes
    exclude_classes: Optional[List[str]] = None  # Exclude these classes


class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detection configuration parameters
        """
        self.config = config or DetectionConfig()
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the detector (load models, etc.)."""
        pass
    
    @abstractmethod
    async def detect(self, 
                    image: Union[str, Path, Image.Image, np.ndarray],
                    prompts: Optional[List[str]] = None,
                    **kwargs) -> List[Detection]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            prompts: Text prompts for open-vocabulary detection (if supported)
            **kwargs: Additional detector-specific parameters
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def get_supported_features(self) -> Dict[str, bool]:
        """
        Get capabilities of this detector.
        
        Returns:
            Dictionary mapping feature names to availability:
            - 'text_prompts': Supports text-based detection
            - 'segmentation': Provides segmentation masks
            - 'tracking': Supports object tracking
            - 'batch_processing': Can process multiple images
        """
        pass
    
    def filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """Apply configured filters to detections."""
        filtered = []
        
        for det in detections:
            # Filter by class if specified
            if self.config.filter_classes and det.class_name not in self.config.filter_classes:
                continue
            if self.config.exclude_classes and det.class_name in self.config.exclude_classes:
                continue
                
            # Filter by confidence
            if det.confidence < self.config.confidence_threshold:
                continue
                
            # Filter by object size
            bbox = det.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = max(width, height)
            
            if size < self.config.min_object_size:
                continue
            if self.config.max_object_size and size > self.config.max_object_size:
                continue
                
            filtered.append(det)
        
        # Limit number of detections
        if len(filtered) > self.config.max_detections:
            # Sort by confidence and take top N
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:self.config.max_detections]
            
        return filtered
    
    async def detect_batch(self, 
                          images: List[Union[str, Path, Image.Image, np.ndarray]],
                          prompts: Optional[List[str]] = None,
                          **kwargs) -> List[List[Detection]]:
        """
        Detect objects in multiple images.
        
        Args:
            images: List of input images
            prompts: Text prompts for detection
            **kwargs: Additional parameters
            
        Returns:
            List of detection lists (one per image)
        """
        # Default implementation: process each image individually
        results = []
        for image in images:
            detections = await self.detect(image, prompts, **kwargs)
            results.append(detections)
        return results
    
    def benchmark(self, 
                 test_images: List[Union[str, Path, Image.Image, np.ndarray]],
                 prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark detector performance.
        
        Args:
            test_images: Images to test on
            prompts: Text prompts for detection
            
        Returns:
            Performance metrics
        """
        import time
        
        # Warm up
        if test_images:
            asyncio.run(self.detect(test_images[0], prompts))
        
        # Time detection
        start_time = time.time()
        results = []
        for image in test_images:
            detections = asyncio.run(self.detect(image, prompts))
            results.append(detections)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(test_images) if test_images else 0
        total_detections = sum(len(r) for r in results)
        
        return {
            'total_time': total_time,
            'avg_time_per_image': avg_time_per_image,
            'images_processed': len(test_images),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(test_images) if test_images else 0,
            'fps': len(test_images) / total_time if total_time > 0 else 0
        }
    
    def __str__(self) -> str:
        """String representation of the detector."""
        return f"{self.__class__.__name__}(config={self.config})"


# Import asyncio for benchmark method
import asyncio 