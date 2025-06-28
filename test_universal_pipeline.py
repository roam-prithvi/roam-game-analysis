#!/usr/bin/env python3
"""
Test script for the Universal Grounded SAM 2 Pipeline.

This script demonstrates how to use the new universal object detection pipeline
that works with any mobile game without requiring game-specific configuration.

Usage:
    python test_universal_pipeline.py [session_path]
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.analysis.video_behavior_analysis.universal_grounded_sam2_pipeline import UniversalGroundedSAM2Pipeline
from src.analysis.video_behavior_analysis.video_analyzer import VideoGameplayAnalyzer


async def test_video_analysis_only(session_path: Path):
    """Test just the video analysis part (Gemini object extraction)."""
    print("🧠 Testing Gemini Video Analysis (Object Extraction Only)")
    print("=" * 60)
    
    try:
        # Initialize video analyzer
        analyzer = VideoGameplayAnalyzer([session_path])
        
        # Run analysis
        print("📤 Running Gemini analysis...")
        analysis_text = analyzer.run_analysis()
        
        # Show extracted objects
        extracted_objects = analyzer.extracted_objects
        
        print(f"\n🎯 Analysis Results:")
        print(f"✅ Objects extracted: {len(extracted_objects)}")
        if extracted_objects:
            print("📝 Discovered objects:")
            for obj in sorted(extracted_objects):
                print(f"   - {obj}")
            
            print(f"\n🔧 Generated universal prompt:")
            print(f"   {analyzer.create_universal_grounded_sam_prompt()}")
        else:
            print("⚠️ No specific objects extracted - will use universal prompts")
        
        return True
        
    except Exception as e:
        print(f"❌ Video analysis test failed: {e}")
        return False


async def test_full_pipeline(session_path: Path):
    """Test the complete universal pipeline."""
    print("\n🚀 Testing Complete Universal Pipeline")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = UniversalGroundedSAM2Pipeline(
            session_path, 
            max_frames=5,  # Limit to 5 frames for testing
            frame_interval=30,
            generate_objectives=True  # Include objective generation
        )
        
        # Run complete pipeline
        summary_file = await pipeline.run_complete_pipeline()
        
        print(f"\n🎉 Pipeline test completed!")
        print(f"📊 Results summary: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False


async def demonstrate_universal_vs_traditional():
    """Demonstrate the difference between universal and traditional approaches."""
    print("\n📋 Universal vs Traditional Approach Comparison")
    print("=" * 60)
    
    print("🔧 Traditional Game-Specific Approach:")
    print("   ❌ Requires manual prompt engineering for each game")
    print("   ❌ Limited to pre-configured games (PvZ2, Subway Surfers, etc.)")
    print("   ❌ Must update prompts when encountering new object types")
    print("   ❌ Cannot analyze unknown games")
    
    print("\n🌟 Universal Approach:")
    print("   ✅ Works with ANY mobile game automatically")
    print("   ✅ Uses Gemini to discover objects from video content")
    print("   ✅ Combines AI analysis with computer vision detection")
    print("   ✅ Self-adapting and scalable")
    print("   ✅ No manual configuration required")
    
    print("\n🎯 Use Cases:")
    print("   • Analyzing new games without prior knowledge")
    print("   • Research on mobile game mechanics across genres")
    print("   • Automated asset extraction for any game")
    print("   • Building comprehensive game object databases")


def validate_session(session_path: Path) -> bool:
    """Validate that the session path is valid for testing."""
    if not session_path.exists():
        print(f"❌ Session path not found: {session_path}")
        return False
    
    video_file = session_path / "screen_recording.mp4"
    if not video_file.exists():
        print(f"❌ Video file not found: {video_file}")
        return False
    
    touch_file = session_path / "touch_events.log"
    if not touch_file.exists():
        print(f"❌ Touch events file not found: {touch_file}")
        return False
    
    print(f"✅ Session validation passed: {session_path.name}")
    return True


async def main():
    """Main test function."""
    print("🎮 Universal Grounded SAM 2 Pipeline Test")
    print("=" * 60)
    
    # Get session path from command line or use example
    if len(sys.argv) > 1:
        session_path = Path(sys.argv[1])
    else:
        # Look for any available session
        data_dir = Path("data")
        if data_dir.exists():
            # Find first available session
            for game_dir in data_dir.iterdir():
                if game_dir.is_dir():
                    for session_dir in game_dir.iterdir():
                        if session_dir.is_dir() and (session_dir / "screen_recording.mp4").exists():
                            session_path = session_dir
                            break
                    else:
                        continue
                    break
            else:
                print("❌ No valid sessions found in data directory")
                print("Usage: python test_universal_pipeline.py [session_path]")
                return
        else:
            print("❌ Data directory not found")
            print("Usage: python test_universal_pipeline.py [session_path]")
            return
    
    # Validate session
    if not validate_session(session_path):
        return
    
    print(f"🎯 Testing with session: {session_path}")
    
    # Show comparison
    await demonstrate_universal_vs_traditional()
    
    # Test video analysis only (faster test)
    print(f"\n" + "="*60)
    video_success = await test_video_analysis_only(session_path)
    
    if video_success:
        # Ask if user wants to run full pipeline
        print(f"\n" + "="*60)
        response = input("🔍 Run full pipeline test (includes frame extraction and detection)? [y/N]: ").strip().lower()
        
        if response in ['y', 'yes']:
            pipeline_success = await test_full_pipeline(session_path)
            
            if pipeline_success:
                print("\n🎉 All tests passed! Universal pipeline is working correctly.")
                print(f"📁 Check results in: {session_path}/analysis/universal_grounded_sam2/")
            else:
                print("\n⚠️ Full pipeline test failed - check setup and dependencies")
        else:
            print("\n✅ Video analysis test completed successfully!")
            print("💡 You can run the full pipeline anytime with:")
            print(f"   python -m src.analysis.video_behavior_analysis.universal_grounded_sam2_pipeline {session_path}")
    else:
        print("\n❌ Video analysis test failed - check Gemini API setup")


if __name__ == "__main__":
    asyncio.run(main()) 