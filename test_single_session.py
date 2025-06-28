#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.video_behavior_analysis.video_analyzer import VideoGameplayAnalyzer

def main():
    # Test the improved object-focused analysis
    session_path = Path("data/PvZ2/25-06-25_at_00.48.15")
    
    print(f"🎮 Testing improved object-focused analysis on: {session_path}")
    
    try:
        # Initialize with single session
        analyzer = VideoGameplayAnalyzer([session_path])
        analysis_text = analyzer.run_analysis()
        
        print(f"\n🎉 Object-focused analysis completed successfully!")
        print(f"📁 Results saved in: {analyzer.video_analysis_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 