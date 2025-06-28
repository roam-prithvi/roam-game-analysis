#!/usr/bin/env python3
"""
Main entry point for YOLO-based video analysis.

This script provides a user-friendly interface to run either:
1. Basic YOLO + SAM segmentation 
2. Game-aware object detection and segmentation

Usage:
    python -m src.analysis.video_behavior_analysis.run_yolo_analysis
    python -m src.analysis.video_behavior_analysis.run_yolo_analysis --game "subway surfers"
    python -m src.analysis.video_behavior_analysis.run_yolo_analysis data/subway_surfers/08-06-25_at_19.33.00
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.video_behavior_analysis.yolo_video_analyzer import YOLOVideoAnalyzer
from src.analysis.video_behavior_analysis.game_aware_analyzer import GameAwareAnalyzer
from src.util import list_sessions
from src.streaming.android_streamer import sanitize_path_component


def select_analyzer_type() -> str:
    """Let user choose which analyzer to use."""
    print("\n🔍 Select Analysis Type:")
    print("1. Basic YOLO Detection (all objects)")
    print("2. Game-Aware Detection (game-specific objects)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            return "basic"
        elif choice == "2":
            return "game_aware"
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def select_game() -> str:
    """Select a game from available data."""
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("❌ No data directory found. Please record some gameplay first.")
        sys.exit(1)
    
    # Find all game directories
    game_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not game_dirs:
        print("❌ No game data found. Please record some gameplay first.")
        sys.exit(1)
    
    if len(game_dirs) == 1:
        game_name = game_dirs[0].name
        print(f"\n🎮 Found data for: {game_name}")
        return game_name
    
    # Multiple games - let user choose
    print("\n🎮 Available games:")
    for i, game_dir in enumerate(game_dirs, 1):
        sessions = list_sessions(game_dir)
        print(f"  {i}. {game_dir.name} ({len(sessions)} sessions)")
    
    while True:
        try:
            choice = int(input("\nSelect game number: ")) - 1
            if 0 <= choice < len(game_dirs):
                return game_dirs[choice].name
        except ValueError:
            pass
        print("❌ Invalid selection. Please try again.")


def select_session(game_name: str) -> Optional[Path]:
    """Select a session for the given game."""
    game_dir = Path("data") / game_name
    sessions = list_sessions(game_dir)
    
    if not sessions:
        print(f"❌ No sessions found for {game_name}")
        sys.exit(1)
    
    if len(sessions) == 1:
        print(f"\n📁 Found 1 session: {Path(sessions[0]).name}")
        return Path(sessions[0])
    
    # Multiple sessions - show list
    print(f"\n📁 Available sessions for {game_name}:")
    for i, session in enumerate(sessions[:10], 1):  # Show max 10 recent sessions
        session_path = Path(session)
        video_path = session_path / "screen_recording.mp4"
        
        # Get video size if available
        size_str = ""
        if video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            size_str = f" ({size_mb:.1f} MB)"
        
        print(f"  {i}. {session_path.name}{size_str}")
    
    if len(sessions) > 10:
        print(f"  ... and {len(sessions) - 10} more sessions")
    
    while True:
        choice = input("\nSelect session number (or 'all' for batch processing): ").strip()
        
        if choice.lower() == 'all':
            return None  # Signal batch processing
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < min(len(sessions), 10):
                return Path(sessions[idx])
        except ValueError:
            pass
        
        print("❌ Invalid selection. Please try again.")


async def run_single_analysis(session_path: Path, analyzer_type: str, game_name: str):
    """Run analysis on a single session."""
    print(f"\n{'='*60}")
    print(f"📹 Analyzing: {session_path.name}")
    print(f"🎮 Game: {game_name}")
    print(f"🔍 Analyzer: {'Game-Aware' if analyzer_type == 'game_aware' else 'Basic YOLO'}")
    print('='*60)
    
    try:
        if analyzer_type == "game_aware":
            analyzer = GameAwareAnalyzer(session_path, game_name=game_name)
            results = await analyzer.analyze_video()
        else:
            analyzer = YOLOVideoAnalyzer(session_path)
            results = await analyzer.analyze_video()
        
        print("\n✅ Analysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure the session has a screen_recording.mp4 file")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


async def run_batch_analysis(game_name: str, analyzer_type: str, max_sessions: int = 5):
    """Run analysis on multiple sessions."""
    game_dir = Path("data") / game_name
    sessions = list_sessions(game_dir)[:max_sessions]
    
    print(f"\n📦 Batch processing {len(sessions)} sessions for {game_name}")
    
    for i, session_path in enumerate(sessions, 1):
        print(f"\n{'='*60}")
        print(f"Processing session {i}/{len(sessions)}")
        
        await run_single_analysis(Path(session_path), analyzer_type, game_name)
        
        # Add a small delay between sessions
        if i < len(sessions):
            await asyncio.sleep(1)
    
    print(f"\n🎉 Batch processing complete! Processed {len(sessions)} sessions.")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run YOLO-based video analysis for game recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m src.analysis.video_behavior_analysis.run_yolo_analysis
  
  # Specify game name
  python -m src.analysis.video_behavior_analysis.run_yolo_analysis --game "subway surfers"
  
  # Analyze specific session
  python -m src.analysis.video_behavior_analysis.run_yolo_analysis data/subway_surfers/08-06-25_at_19.33.00
  
  # Use basic YOLO analyzer
  python -m src.analysis.video_behavior_analysis.run_yolo_analysis --analyzer basic
  
  # Batch process with custom limit
  python -m src.analysis.video_behavior_analysis.run_yolo_analysis --batch --max-sessions 10
        """
    )
    
    parser.add_argument("path", nargs="?", help="Direct path to session directory")
    parser.add_argument("--game", "-g", help="Game name")
    parser.add_argument("--analyzer", "-a", choices=["basic", "game_aware"], 
                       help="Analyzer type (default: prompt user)")
    parser.add_argument("--batch", "-b", action="store_true", 
                       help="Process all sessions for the selected game")
    parser.add_argument("--max-sessions", "-m", type=int, default=5,
                       help="Maximum sessions to process in batch mode (default: 5)")
    
    args = parser.parse_args()
    
    # Check environment
    if not os.getenv("SIEVE_API_KEY"):
        print("⚠️  Warning: SIEVE_API_KEY environment variable not set!")
        print("   SAM 2 segmentation will fail without it.")
        print("   Get your API key from: https://www.sievedata.com/")
        print("   Then set it: export SIEVE_API_KEY='your-key-here'")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    print("\n🎮 YOLO-Based Video Game Analysis")
    print("==================================")
    
    # Direct path mode
    if args.path:
        session_path = Path(args.path)
        if not session_path.exists():
            print(f"❌ Path not found: {session_path}")
            return
        
        # Try to detect game name
        game_name = args.game
        if not game_name:
            # Try to infer from path
            parts = session_path.parts
            if "data" in parts:
                idx = parts.index("data")
                if idx + 1 < len(parts):
                    game_name = parts[idx + 1]
        
        if not game_name:
            game_name = input("🎮 Enter game name: ").strip()
        
        # Get analyzer type
        analyzer_type = args.analyzer or select_analyzer_type()
        
        await run_single_analysis(session_path, analyzer_type, game_name)
        
    else:
        # Interactive mode
        game_name = args.game or select_game()
        analyzer_type = args.analyzer or select_analyzer_type()
        
        if args.batch:
            await run_batch_analysis(game_name, analyzer_type, args.max_sessions)
        else:
            session_path = select_session(game_name)
            if session_path:
                await run_single_analysis(session_path, analyzer_type, game_name)
            else:
                # User selected 'all' - run batch
                await run_batch_analysis(game_name, analyzer_type, args.max_sessions)
    
    print("\n✨ Done! Check the analysis folders for results:")
    print("   - Basic YOLO: analysis/yolo_segmentation/")
    print("   - Game-Aware: analysis/game_aware_segmentation/")
    print("   - Segmented assets are saved as transparent PNGs")


if __name__ == "__main__":
    asyncio.run(main()) 