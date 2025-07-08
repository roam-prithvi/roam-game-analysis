"""
Centralized configuration management for the video behavior analysis pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os
from pydantic import BaseModel, Field, validator


class DetectorType(str, Enum):
    """Available detector types."""
    YOLO = "yolo"
    GROUNDED_SAM = "grounded_sam"
    UNIVERSAL = "universal"


class AnalysisMode(str, Enum):
    """Analysis modes."""
    SINGLE_SESSION = "single_session"
    MULTI_SESSION = "multi_session"
    BATCH = "batch"


class DeviceType(str, Enum):
    """Compute device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    AUTO = "auto"


# Using Pydantic for validation and serialization
class ModelConfig(BaseModel):
    """Configuration for model paths and parameters."""
    
    # SAM 2 Configuration
    sam_model_type: str = Field(default="large", description="SAM model size: tiny, small, base, large")
    sam_checkpoint_path: Optional[Path] = Field(default=None)
    sam_config_name: Optional[str] = Field(default=None)
    
    # YOLO Configuration
    yolo_model_path: Path = Field(default=Path("yolo11m.pt"))
    yolo_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    yolo_iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Grounding DINO Configuration
    grounding_dino_config_path: Optional[Path] = Field(default=None)
    grounding_dino_checkpoint_path: Optional[Path] = Field(default=None)
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    
    @validator('sam_checkpoint_path', 'grounding_dino_checkpoint_path', pre=True)
    def resolve_paths(cls, v):
        if v and not isinstance(v, Path):
            return Path(v)
        return v
    
    class Config:
        json_encoders = {Path: str}


class ProcessingConfig(BaseModel):
    """Configuration for video processing parameters."""
    
    # Frame extraction
    max_frames: int = Field(default=100, ge=1, description="Maximum frames to extract")
    frame_interval: int = Field(default=30, ge=1, description="Extract every Nth frame")
    frame_quality: int = Field(default=95, ge=1, le=100, description="JPEG quality for extracted frames")
    
    # Video analysis
    enable_touch_events: bool = Field(default=True, description="Process touch event logs")
    sample_touch_events: int = Field(default=20, description="Number of touch events to sample")
    
    # Detection parameters
    min_object_size: int = Field(default=10, description="Minimum object size in pixels")
    max_object_size: Optional[int] = Field(default=None, description="Maximum object size in pixels")
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Non-max suppression threshold")
    
    # Performance
    batch_size: int = Field(default=1, ge=1, description="Batch size for processing")
    num_workers: int = Field(default=0, ge=0, description="Number of worker processes")
    device: DeviceType = Field(default=DeviceType.AUTO, description="Compute device")


class APIConfig(BaseModel):
    """Configuration for external API settings."""
    
    # Gemini API
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key")
    gemini_model: str = Field(default="gemini-2.5-pro", description="Gemini model to use")
    gemini_timeout: int = Field(default=300, description="API timeout in seconds")
    enable_video_cache: bool = Field(default=True, description="Cache uploaded videos")
    
    # Sieve API (for cloud SAM)
    sieve_api_key: Optional[str] = Field(default=None, description="Sieve API key")
    sieve_timeout: int = Field(default=180, description="Sieve API timeout")
    
    @validator('gemini_api_key', 'sieve_api_key', pre=True)
    def load_from_env(cls, v, field):
        if not v:
            env_key = f"{field.name.upper()}"
            return os.getenv(env_key)
        return v


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    
    # Output paths
    output_dir: Path = Field(default=Path("analysis"), description="Base output directory")
    save_visualizations: bool = Field(default=True, description="Save annotated images")
    save_masks: bool = Field(default=True, description="Save segmentation masks")
    save_cutouts: bool = Field(default=True, description="Save object cutouts")
    
    # Output formats
    image_format: str = Field(default="png", description="Output image format")
    mask_format: str = Field(default="npy", description="Mask save format: npy or png")
    
    # Logging
    verbose: bool = Field(default=True, description="Verbose output")
    log_level: str = Field(default="INFO", description="Logging level")


@dataclass
class PipelineConfig:
    """Main pipeline configuration using dataclass for simpler access."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Pipeline settings
    detector_type: DetectorType = DetectorType.UNIVERSAL
    analysis_mode: AnalysisMode = AnalysisMode.SINGLE_SESSION
    generate_objectives: bool = True
    
    # Game-specific settings
    game_name: Optional[str] = None
    game_prompts: Optional[List[str]] = None
    
    @classmethod
    def from_file(cls, config_path: Path) -> "PipelineConfig":
        """Load configuration from JSON/YAML file."""
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                data = json.load(f)
            elif config_path.suffix in ['.yml', '.yaml']:
                import yaml
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Parse sub-configurations
        config = cls()
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'processing' in data:
            config.processing = ProcessingConfig(**data['processing'])
        if 'api' in data:
            config.api = APIConfig(**data['api'])
        if 'output' in data:
            config.output = OutputConfig(**data['output'])
        
        # Parse main settings
        for key in ['detector_type', 'analysis_mode', 'generate_objectives', 'game_name', 'game_prompts']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_file(self, config_path: Path):
        """Save configuration to JSON/YAML file."""
        data = {
            'detector_type': self.detector_type.value,
            'analysis_mode': self.analysis_mode.value,
            'generate_objectives': self.generate_objectives,
            'game_name': self.game_name,
            'game_prompts': self.game_prompts,
            'model': self.model.dict(),
            'processing': self.processing.dict(),
            'api': self.api.dict(),
            'output': self.output.dict()
        }
        
        with open(config_path, 'w') as f:
            if config_path.suffix == '.json':
                json.dump(data, f, indent=2, default=str)
            elif config_path.suffix in ['.yml', '.yaml']:
                import yaml
                yaml.dump(data, f, default_flow_style=False)
    
    def get_device(self):
        """Get the actual device to use based on configuration and availability."""
        import torch
        
        if self.processing.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif self.processing.device == DeviceType.CUDA:
            return torch.device("cuda")
        elif self.processing.device == DeviceType.MPS:
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all relevant paths based on configuration."""
        base_dir = Path(__file__).parent.parent.parent.parent
        
        paths = {
            'grounded_sam_dir': base_dir / "Grounded-SAM-2",
            'output_base': self.output.output_dir,
        }
        
        # Model paths
        if self.model.sam_checkpoint_path:
            paths['sam_checkpoint'] = self.model.sam_checkpoint_path
        else:
            # Default paths
            sam_model_files = {
                'large': 'sam2_hiera_large.pt',
                'base': 'sam2_hiera_base_plus.pt',
                'small': 'sam2_hiera_small.pt',
                'tiny': 'sam2_hiera_tiny.pt'
            }
            checkpoint_name = sam_model_files.get(self.model.sam_model_type, 'sam2_hiera_large.pt')
            paths['sam_checkpoint'] = paths['grounded_sam_dir'] / 'checkpoints' / checkpoint_name
        
        if self.model.grounding_dino_checkpoint_path:
            paths['grounding_dino_checkpoint'] = self.model.grounding_dino_checkpoint_path
        else:
            paths['grounding_dino_checkpoint'] = paths['grounded_sam_dir'] / 'gdino_checkpoints' / 'groundingdino_swinb_cogcoor.pth'
        
        return paths


# Preset configurations for common use cases
class ConfigPresets:
    """Pre-defined configuration presets for common scenarios."""
    
    @staticmethod
    def fast_analysis() -> PipelineConfig:
        """Fast analysis with reduced quality."""
        config = PipelineConfig()
        config.processing.max_frames = 50
        config.processing.frame_interval = 60
        config.model.sam_model_type = "small"
        config.processing.device = DeviceType.AUTO
        config.output.save_visualizations = False
        return config
    
    @staticmethod
    def high_quality() -> PipelineConfig:
        """High quality analysis with maximum detail."""
        config = PipelineConfig()
        config.processing.max_frames = 200
        config.processing.frame_interval = 15
        config.model.sam_model_type = "large"
        config.model.box_threshold = 0.20
        config.model.text_threshold = 0.15
        config.processing.frame_quality = 100
        return config
    
    @staticmethod
    def yolo_only() -> PipelineConfig:
        """YOLO-only detection without SAM."""
        config = PipelineConfig()
        config.detector_type = DetectorType.YOLO
        config.output.save_masks = False
        config.output.save_cutouts = False
        return config
    
    @staticmethod
    def batch_processing() -> PipelineConfig:
        """Optimized for batch processing multiple sessions."""
        config = PipelineConfig()
        config.analysis_mode = AnalysisMode.BATCH
        config.processing.batch_size = 4
        config.processing.num_workers = 4
        config.api.enable_video_cache = True
        config.output.verbose = False
        return config


# Environment-based configuration
def load_config_from_env() -> PipelineConfig:
    """Load configuration from environment variables."""
    config = PipelineConfig()
    
    # Override from environment
    env_mappings = {
        'ANALYSIS_DETECTOR_TYPE': ('detector_type', DetectorType),
        'ANALYSIS_MAX_FRAMES': ('processing.max_frames', int),
        'ANALYSIS_DEVICE': ('processing.device', DeviceType),
        'ANALYSIS_VERBOSE': ('output.verbose', bool),
    }
    
    for env_key, (config_path, type_converter) in env_mappings.items():
        if env_value := os.getenv(env_key):
            # Handle nested attributes
            parts = config_path.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], type_converter(env_value))
    
    return config


# Usage example
if __name__ == "__main__":
    # Create default config
    config = PipelineConfig()
    
    # Or load from file
    # config = PipelineConfig.from_file(Path("config.json"))
    
    # Or use a preset
    # config = ConfigPresets.fast_analysis()
    
    # Or load from environment
    # config = load_config_from_env()
    
    # Save configuration
    config.to_file(Path("example_config.json"))
    
    print(f"Detector: {config.detector_type}")
    print(f"Max frames: {config.processing.max_frames}")
    print(f"Device: {config.get_device()}")
    print(f"Paths: {config.get_paths()}")