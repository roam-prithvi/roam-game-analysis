"""
Simple, practical configuration system for the video behavior analysis pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os
import json


@dataclass
class AnalysisConfig:
    """Configuration for video behavior analysis pipeline."""
    
    # Detector settings
    detector_type: str = "universal"  # "yolo", "grounded_sam", "universal"
    
    # Processing parameters
    max_frames: int = 100
    frame_interval: int = 30
    frame_quality: int = 95
    
    # Model parameters
    sam_model_type: str = "large"  # "tiny", "small", "base", "large"
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    yolo_confidence: float = 0.5
    yolo_iou_threshold: float = 0.7
    nms_threshold: float = 0.5
    
    # Output settings
    save_visualizations: bool = True
    save_masks: bool = True
    save_cutouts: bool = True
    generate_objectives: bool = True
    verbose: bool = True
    
    # Performance
    device: str = "auto"  # "cuda", "mps", "cpu", "auto"
    batch_size: int = 1
    
    # API settings
    enable_video_cache: bool = True
    gemini_model: str = "gemini-2.5-pro"
    
    # Game settings
    game_name: Optional[str] = None
    custom_prompts: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AnalysisConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    @classmethod
    def from_file(cls, path: Path) -> "AnalysisConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    def get_device(self):
        """Get the actual PyTorch device."""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self.device
    
    def get_sam_checkpoint(self) -> str:
        """Get SAM checkpoint filename based on model type."""
        checkpoints = {
            "large": "sam2_hiera_large.pt",
            "base": "sam2_hiera_base_plus.pt",
            "small": "sam2_hiera_small.pt",
            "tiny": "sam2_hiera_tiny.pt"
        }
        return checkpoints.get(self.sam_model_type, "sam2_hiera_large.pt")


# Preset configurations
PRESETS = {
    "fast": AnalysisConfig(
        max_frames=50,
        frame_interval=60,
        sam_model_type="small",
        save_visualizations=False
    ),
    "high_quality": AnalysisConfig(
        max_frames=200,
        frame_interval=15,
        sam_model_type="large",
        box_threshold=0.20,
        text_threshold=0.15,
        frame_quality=100
    ),
    "yolo_basic": AnalysisConfig(
        detector_type="yolo",
        save_masks=False,
        save_cutouts=False
    ),
    "universal_standard": AnalysisConfig(
        detector_type="universal",
        max_frames=100,
        frame_interval=30,
        generate_objectives=True
    )
}


# Global config instance (can be overridden)
_global_config = None


def get_config() -> AnalysisConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = AnalysisConfig()
    return _global_config


def set_config(config: AnalysisConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def load_config(path: Optional[Path] = None) -> AnalysisConfig:
    """Load configuration from file or environment."""
    if path and path.exists():
        config = AnalysisConfig.from_file(path)
    else:
        # Start with default
        config = AnalysisConfig()
        
        # Check for config file in standard locations
        config_locations = [
            Path("config.json"),
            Path.home() / ".roam_analysis" / "config.json",
            Path(os.getenv("ROAM_CONFIG", ""))
        ]
        
        for loc in config_locations:
            if loc and loc.exists():
                config = AnalysisConfig.from_file(loc)
                break
    
    # Override with environment variables
    env_mappings = {
        "ANALYSIS_DETECTOR": "detector_type",
        "ANALYSIS_MAX_FRAMES": ("max_frames", int),
        "ANALYSIS_DEVICE": "device",
        "ANALYSIS_VERBOSE": ("verbose", lambda x: x.lower() == "true"),
        "ANALYSIS_GAME": "game_name"
    }
    
    for env_var, target in env_mappings.items():
        if value := os.getenv(env_var):
            if isinstance(target, tuple):
                attr, converter = target
                setattr(config, attr, converter(value))
            else:
                setattr(config, target, value)
    
    set_config(config)
    return config


# Usage in existing code - minimal changes needed
def integrate_with_analyzer(analyzer_instance, config: Optional[AnalysisConfig] = None):
    """Apply configuration to an analyzer instance."""
    if config is None:
        config = get_config()
    
    # Apply common settings
    if hasattr(analyzer_instance, 'max_frames'):
        analyzer_instance.max_frames = config.max_frames
    if hasattr(analyzer_instance, 'frame_interval'):
        analyzer_instance.frame_interval = config.frame_interval
    if hasattr(analyzer_instance, 'generate_objectives'):
        analyzer_instance.generate_objectives = config.generate_objectives
    
    # Apply detector-specific settings
    if hasattr(analyzer_instance, 'detector'):
        detector = analyzer_instance.detector
        if hasattr(detector, 'box_threshold'):
            detector.box_threshold = config.box_threshold
        if hasattr(detector, 'text_threshold'):
            detector.text_threshold = config.text_threshold
        if hasattr(detector, 'confidence_threshold'):
            detector.confidence_threshold = config.yolo_confidence
    
    return analyzer_instance