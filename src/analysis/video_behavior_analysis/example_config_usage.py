"""
Example of how to use the configuration system in the video analysis pipeline.
"""

from pathlib import Path
from pipeline_config import PipelineConfig, ConfigPresets, DetectorType
from universal_grounded_sam2_pipeline import UniversalGroundedSAM2Pipeline
from improved_game_aware_analyzer import ImprovedGameAwareAnalyzer


# Example 1: Using configuration with UniversalGroundedSAM2Pipeline
class ConfigurableUniversalPipeline(UniversalGroundedSAM2Pipeline):
    """Extended pipeline that uses centralized configuration."""
    
    def __init__(self, session_path: Path, config: PipelineConfig):
        # Initialize base class with config values
        super().__init__(
            session_path=session_path,
            max_frames=config.processing.max_frames,
            frame_interval=config.processing.frame_interval,
            generate_objectives=config.generate_objectives
        )
        
        self.config = config
        self.device = config.get_device()
        self.paths = config.get_paths()
        
    def create_universal_prompt(self, frame_idx: int = 0) -> str:
        """Create prompt using config settings."""
        prompts = set()
        
        # Add custom game prompts from config
        if self.config.game_prompts:
            prompts.update(self.config.game_prompts)
        
        # Add discovered objects
        if self.gemini_objects:
            prompts.update(list(self.gemini_objects)[:15])
        
        # Default behavior
        return super().create_universal_prompt(frame_idx)
    
    def setup_grounded_sam2(self) -> bool:
        """Set up using paths from config."""
        grounded_sam_dir = self.paths['grounded_sam_dir']
        
        # Check for required files
        demo_script = grounded_sam_dir / "grounded_sam2_custom_demo.py"
        if not demo_script.exists():
            print(f"❌ Grounded SAM 2 demo script not found: {demo_script}")
            return False
        
        # Check for model checkpoints using config paths
        if not self.paths['sam_checkpoint'].exists():
            print(f"❌ SAM checkpoint not found: {self.paths['sam_checkpoint']}")
            return False
        
        print(f"✅ Using SAM model: {self.config.model.sam_model_type}")
        print(f"✅ Device: {self.device}")
        return True


# Example 2: Configuration-aware analyzer factory
class AnalyzerFactory:
    """Factory for creating analyzers with configuration."""
    
    @staticmethod
    def create_analyzer(session_path: Path, config: PipelineConfig):
        """Create appropriate analyzer based on configuration."""
        
        if config.detector_type == DetectorType.UNIVERSAL:
            return ConfigurableUniversalPipeline(session_path, config)
        
        elif config.detector_type in [DetectorType.YOLO, DetectorType.GROUNDED_SAM]:
            analyzer = ImprovedGameAwareAnalyzer(
                session_path=session_path,
                detector_type=config.detector_type.value
            )
            
            # Apply configuration
            analyzer.max_frames = config.processing.max_frames
            analyzer.frame_interval = config.processing.frame_interval
            
            if config.game_name:
                analyzer.game_name = config.game_name
            
            # Set detection parameters
            if hasattr(analyzer, 'detector'):
                if config.detector_type == DetectorType.GROUNDED_SAM:
                    analyzer.detector.box_threshold = config.model.box_threshold
                    analyzer.detector.text_threshold = config.model.text_threshold
                elif config.detector_type == DetectorType.YOLO:
                    analyzer.detector.confidence_threshold = config.model.yolo_confidence
            
            return analyzer
        
        else:
            raise ValueError(f"Unknown detector type: {config.detector_type}")


# Example 3: Usage patterns
async def example_usage():
    """Show different ways to use the configuration system."""
    
    # 1. Default configuration
    config = PipelineConfig()
    session_path = Path("data/subway_surfers/24-06-25_at_23.51.52")
    analyzer = AnalyzerFactory.create_analyzer(session_path, config)
    
    # 2. Load from file
    config = PipelineConfig.from_file(Path("my_config.json"))
    analyzer = AnalyzerFactory.create_analyzer(session_path, config)
    
    # 3. Use a preset
    config = ConfigPresets.fast_analysis()
    analyzer = AnalyzerFactory.create_analyzer(session_path, config)
    
    # 4. Programmatic configuration
    config = PipelineConfig()
    config.detector_type = DetectorType.YOLO
    config.processing.max_frames = 150
    config.processing.device = "mps"  # Use Apple Silicon
    config.model.yolo_confidence = 0.6
    config.output.save_visualizations = True
    analyzer = AnalyzerFactory.create_analyzer(session_path, config)
    
    # 5. Override specific settings
    config = ConfigPresets.high_quality()
    config.game_name = "subway_surfers"
    config.game_prompts = ["train", "character", "coin", "barrier"]
    analyzer = AnalyzerFactory.create_analyzer(session_path, config)
    
    # Run analysis
    result = await analyzer.analyze_video()
    return result


# Example 4: CLI integration
def create_cli_with_config():
    """Example of integrating config with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("session_path", help="Path to session")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--preset", choices=["fast", "high_quality", "batch"])
    parser.add_argument("--detector", choices=["yolo", "grounded_sam", "universal"])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"])
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config:
        config = PipelineConfig.from_file(Path(args.config))
    elif args.preset:
        preset_map = {
            "fast": ConfigPresets.fast_analysis,
            "high_quality": ConfigPresets.high_quality,
            "batch": ConfigPresets.batch_processing
        }
        config = preset_map[args.preset]()
    else:
        config = PipelineConfig()
    
    # Override with CLI arguments
    if args.detector:
        config.detector_type = DetectorType(args.detector)
    if args.max_frames:
        config.processing.max_frames = args.max_frames
    if args.device:
        config.processing.device = args.device
    
    return config


# Example 5: Config validation and defaults
def validate_and_setup_config(config: PipelineConfig) -> PipelineConfig:
    """Validate configuration and set up defaults."""
    
    # Auto-detect game from prompts if not specified
    if not config.game_name and config.game_prompts:
        # Try to infer game from prompts
        prompt_text = " ".join(config.game_prompts).lower()
        if "train" in prompt_text and "subway" in prompt_text:
            config.game_name = "subway_surfers"
        elif "temple" in prompt_text and "monkey" in prompt_text:
            config.game_name = "temple_run"
    
    # Set up API keys from environment if not in config
    if not config.api.gemini_api_key:
        config.api.gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Validate paths exist
    paths = config.get_paths()
    if config.detector_type == DetectorType.GROUNDED_SAM:
        if not paths['grounded_sam_dir'].exists():
            raise ValueError(f"Grounded SAM directory not found: {paths['grounded_sam_dir']}")
    
    return config


if __name__ == "__main__":
    # Create example configuration file
    config = PipelineConfig()
    config.detector_type = DetectorType.UNIVERSAL
    config.processing.max_frames = 100
    config.model.sam_model_type = "large"
    config.game_name = "subway_surfers"
    config.to_file(Path("example_pipeline_config.json"))
    print("Created example_pipeline_config.json")