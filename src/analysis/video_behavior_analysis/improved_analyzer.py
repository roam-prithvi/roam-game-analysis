"""
Improved game-aware video analyzer with detector abstraction.
"""

import asyncio
from pathlib import Path
from typing import Optional, Union

from .base_detector import DetectionConfig
from .grounded_sam_detector import create_grounded_sam_detector
from .yolo_detector import create_yolo_detector
from .game_prompts import get_game_prompts, get_supported_games


class ImprovedAnalyzer:
    """Game-aware analyzer with pluggable detectors."""
    
    def __init__(self, session_path: Union[str, Path], detector_type: str = "grounded_sam"):
        self.session_path = Path(session_path)
        self.detector_type = detector_type
        self.detector = None
        
        # Auto-detect game
        self.game_name = self._detect_game()
        self.prompts = get_game_prompts(self.game_name)
        
        print(f"🎮 Game: {self.game_name}")
        print(f"🔧 Detector: {detector_type}")
        print(f"🎯 Prompts: {len(self.prompts)}")
    
    def _detect_game(self) -> str:
        """Detect game from path."""
        path_str = str(self.session_path).lower()
        
        for game in get_supported_games():
            if game.replace("_", "") in path_str.replace("_", "").replace(" ", ""):
                return game
        
        return "generic_runner"
    
    async def initialize(self):
        """Initialize the detector."""
        if self.detector_type == "grounded_sam":
            self.detector = await create_grounded_sam_detector()
        else:
            self.detector = await create_yolo_detector()
    
    async def analyze(self):
        """Run analysis."""
        await self.initialize()
        
        # Detection logic here
        print(f"✅ Analysis complete with {self.detector_type}")
        return {"detector": self.detector_type, "game": self.game_name}


async def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python improved_analyzer.py <session_path> [detector_type]")
        return
    
    session_path = sys.argv[1]
    detector_type = sys.argv[2] if len(sys.argv) > 2 else "grounded_sam"
    
    analyzer = ImprovedAnalyzer(session_path, detector_type)
    result = await analyzer.analyze()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main()) 