#!/usr/bin/env python3
"""
Runner script for video behavior analysis.
Usage: python -m src.analysis.video_behavior_analysis.run_video_analysis
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.video_behavior_analysis.video_analyzer import main

if __name__ == "__main__":
    main() 