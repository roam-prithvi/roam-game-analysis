#!/usr/bin/env python3
"""
Example usage of the YOLO-based video analyzers.

This script demonstrates how to use:
1. YOLOVideoAnalyzer - Basic YOLO + SAM segmentation
2. GameAwareAnalyzer - Game-specific object detection and segmentation
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.video_behavior_analysis.yolo_video_analyzer import YOLOVideoAnalyzer
from src.analysis.video_behavior_analysis.game_aware_analyzer import GameAwareAnalyzer
from src.util import list_sessions


async def example_basic_yolo_analysis():
    """Example of using the basic YOLO video analyzer."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic YOLO Video Analysis")
    print("="*60)
    
    # Find a sample session
    game_dir = Path("data") / "subway surfers"
    if not game_dir.exists():
        print(f"❌ No data found for 'subway surfers'. Please record some gameplay first.")
        return
    
    sessions = list_sessions(str(game_dir))
    if not sessions:
        print(f"❌ No sessions found for 'subway surfers'")
        return
    
    # Use the most recent session
    latest_session = sessions[0]
    print(f"\n📁 Using session: {Path(latest_session).name}")
    
    # Create analyzer and run analysis
    analyzer = YOLOVideoAnalyzer(latest_session)
    results = await analyzer.analyze_video()
    
    print(f"\n📊 Analysis Summary:")
    print(f"   - Frames processed: {results['frames_processed']}")
    print(f"   - Total objects detected: {results['total_objects_detected']}")
    print(f"   - Object classes found: {', '.join(results['object_classes'].keys())}")


async def example_game_aware_analysis():
    """Example of using the game-aware analyzer."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Game-Aware Video Analysis")
    print("="*60)
    
    # Example with Subway Surfers
    game_name = "subway surfers"
    game_dir = Path("data") / game_name
    
    if not game_dir.exists():
        print(f"❌ No data found for '{game_name}'. Trying with any available game...")
        # Find any game with data
        data_dir = Path("data")
        game_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if not game_dirs:
            print("❌ No game data found. Please record some gameplay first.")
            return
        game_dir = game_dirs[0]
        game_name = game_dir.name
        print(f"✅ Found game data for: {game_name}")
    
    sessions = list_sessions(str(game_dir))
    if not sessions:
        print(f"❌ No sessions found for '{game_name}'")
        return
    
    # Use the most recent session
    latest_session = sessions[0]
    print(f"\n📁 Using session: {Path(latest_session).name}")
    
    # Create game-aware analyzer
    analyzer = GameAwareAnalyzer(latest_session, game_name=game_name)
    
    # Run analysis with fewer frames for demo
    results = await analyzer.analyze_video(max_frames=20)
    
    print(f"\n📊 Game-Aware Analysis Summary:")
    print(f"   - Game: {results['game']}")
    print(f"   - Frames processed: {results['frames_processed']}")
    print(f"   - Total game objects: {results['total_objects']}")
    
    if results['categories']:
        print(f"\n📦 Object Categories:")
        for category, count in results['categories'].items():
            print(f"   - {category}: {count} objects")


async def compare_analyzers():
    """Compare results from both analyzers on the same session."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Comparing Both Analyzers")
    print("="*60)
    
    # Find a sample session
    game_name = "subway surfers"
    game_dir = Path("data") / game_name
    
    if not game_dir.exists():
        print(f"❌ No data found for '{game_name}'")
        return
    
    sessions = list_sessions(str(game_dir))
    if not sessions:
        print(f"❌ No sessions found")
        return
    
    session_path = sessions[0]
    print(f"\n📁 Analyzing session: {Path(session_path).name}")
    
    # Run basic YOLO analysis
    print("\n🔍 Running Basic YOLO Analysis...")
    basic_analyzer = YOLOVideoAnalyzer(session_path)
    basic_results = await basic_analyzer.analyze_video()
    
    # Run game-aware analysis
    print("\n🎮 Running Game-Aware Analysis...")
    game_analyzer = GameAwareAnalyzer(session_path, game_name=game_name)
    game_results = await game_analyzer.analyze_video(max_frames=20)
    
    # Compare results
    print("\n📊 Comparison:")
    print(f"\nBasic YOLO Analysis:")
    print(f"   - Total objects: {basic_results['total_objects_detected']}")
    print(f"   - Classes: {', '.join(list(basic_results['object_classes'].keys())[:5])}...")
    
    print(f"\nGame-Aware Analysis:")
    print(f"   - Total objects: {game_results['total_objects']}")
    print(f"   - Categories: {', '.join(game_results['categories'].keys())}")
    
    print(f"\n💡 The game-aware analyzer filters and categorizes objects")
    print(f"   specifically for {game_name}, providing more relevant results.")


async def process_multiple_sessions():
    """Example of processing multiple sessions in batch."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing Multiple Sessions")
    print("="*60)
    
    game_name = "subway surfers"
    game_dir = Path("data") / game_name
    
    if not game_dir.exists():
        print(f"❌ No data found for '{game_name}'")
        return
    
    sessions = list_sessions(str(game_dir))[:3]  # Process up to 3 sessions
    if not sessions:
        print(f"❌ No sessions found")
        return
    
    print(f"\n📁 Found {len(sessions)} sessions to process")
    
    # Process each session
    for i, session_path in enumerate(sessions, 1):
        print(f"\n{'='*40}")
        print(f"Processing session {i}/{len(sessions)}: {Path(session_path).name}")
        print('='*40)
        
        try:
            analyzer = GameAwareAnalyzer(session_path, game_name=game_name)
            results = await analyzer.analyze_video(max_frames=10)  # Fewer frames for speed
            
            print(f"✅ Completed: {results['total_objects']} objects found")
            
        except Exception as e:
            print(f"❌ Error processing session: {e}")
            continue


async def main():
    """Run all examples."""
    print("\n🎮 YOLO Video Analysis Examples")
    print("================================\n")
    
    # Check for required environment variables
    import os
    if not os.getenv("SIEVE_API_KEY"):
        print("⚠️  Warning: SIEVE_API_KEY not set. SAM segmentation will fail.")
        print("   Please set your Sieve API key: export SIEVE_API_KEY='your-key'")
    
    # Run examples
    try:
        # Example 1: Basic YOLO analysis
        await example_basic_yolo_analysis()
        
        # Example 2: Game-aware analysis
        await example_game_aware_analysis()
        
        # Example 3: Compare both analyzers
        await compare_analyzers()
        
        # Example 4: Batch processing
        # await process_multiple_sessions()  # Commented out by default (takes longer)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Examples completed!")
    print("\n💡 Tips:")
    print("   - Use YOLOVideoAnalyzer for general object detection")
    print("   - Use GameAwareAnalyzer for game-specific asset extraction")
    print("   - Check the 'analysis' folder in each session for results")
    print("   - Segmented assets are saved as transparent PNGs")


if __name__ == "__main__":
    asyncio.run(main()) 