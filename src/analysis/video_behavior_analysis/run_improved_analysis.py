"""
Enhanced CLI for running video analysis with multiple detector options.

This script provides a unified interface for running game-aware video analysis
with either YOLO or Grounding DINO + SAM 2 detectors.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .improved_game_aware_analyzer import ImprovedGameAwareAnalyzer as ImprovedAnalyzer
from .detector_evaluation import DetectorEvaluator
from .game_prompts import get_supported_games
from .universal_grounded_sam2_pipeline import UniversalGroundedSAM2Pipeline
from src.util import list_sessions
from src.streaming.android_streamer import sanitize_path_component


def get_video_file(session_path: Path) -> Optional[Path]:
    """Get the video file for a session, preferring trimmed version if available."""
    trimmed_video = session_path / "trimmed_screen_recording.mp4"
    if trimmed_video.exists():
        return trimmed_video

    original_video = session_path / "screen_recording.mp4"
    if original_video.exists():
        return original_video

    return None


def print_banner():
    """Print application banner."""
    print(
        """
🎮 Enhanced Game Video Analysis System
=====================================
🔧 Analysis Options:
   - grounded_sam: Grounding DINO + SAM 2 (Open-vocabulary detection)
   - yolo: YOLOv11 + SAM 2 (Traditional object detection)
   - universal: Universal pipeline with Gemini + Grounded SAM 2 (ANY GAME!)

🎯 Supported Games:
   {}

🚀 Features:
   - Game-specific object detection prompts
   - Universal object detection (no game labels needed)
   - Gemini-driven object discovery
   - Hot-swappable detector backends
   - Comprehensive evaluation tools
   - Asset extraction with segmentation

🌟 NEW: Universal mode works with ANY game without pre-configured labels!
""".format(", ".join(get_supported_games()))
    )


async def analyze_single_session(
    session_path: Path, detector_type: str, game_name: Optional[str] = None
):
    """Analyze a single session with the specified detector."""
    video_file = get_video_file(session_path)
    if video_file:
        video_name = video_file.name
        print(f"\n📁 Analyzing session: {session_path.name} (using {video_name})")
    else:
        print(f"\n📁 Analyzing session: {session_path.name}")

    try:
        if detector_type == "universal":
            # Use universal pipeline that works with any game
            pipeline = UniversalGroundedSAM2Pipeline(
                session_path, generate_objectives=True
            )
            result = await pipeline.run_complete_pipeline()
            print(f"✅ Universal analysis completed successfully")
            return result
        else:
            # Use traditional game-aware analyzer
            analyzer = ImprovedAnalyzer(session_path, detector_type)
            if game_name:
                analyzer.game_name = game_name

            # Run full video analysis (defaults: max 100 frames sampled every 30)
            result = await analyzer.analyze_video()
            print(f"✅ Analysis completed successfully")
            return result

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


async def analyze_multiple_sessions(
    sessions: List[Path], detector_type: str, game_name: Optional[str] = None
):
    """Analyze multiple sessions with the specified detector."""
    print(f"\n📁 Analyzing {len(sessions)} sessions with {detector_type} detector")

    results = []
    for i, session in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] Processing: {session.name}")

        result = await analyze_single_session(session, detector_type, game_name)
        if result:
            results.append(result)

        # Brief pause between sessions to avoid overwhelming the system
        if i < len(sessions):
            await asyncio.sleep(1)

    print(f"\n✅ Completed analysis of {len(results)}/{len(sessions)} sessions")
    return results


async def run_detector_comparison(
    test_images_dir: Path, game_name: str, output_dir: Path
):
    """Run comprehensive detector comparison."""
    print(f"\n🔬 Running detector comparison")
    print(f"🎮 Game: {game_name}")
    print(f"📸 Test images: {test_images_dir}")
    print(f"📁 Output: {output_dir}")

    try:
        evaluator = DetectorEvaluator(test_images_dir, game_name)
        results = await evaluator.run_comparison(output_dir)
        return results
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return None


def interactive_session_selection(game_dir: Path) -> List[Path]:
    """Interactive session selection from a game directory."""
    sessions = list_sessions(game_dir)

    if not sessions:
        print(f"❌ No sessions found in {game_dir}")
        return []

    print(f"\n📂 Found {len(sessions)} session(s) in {game_dir.name}:")
    for i, session in enumerate(sessions, 1):
        print(f"  [{i}] {session.name}")

    if len(sessions) == 1:
        print(f"\n🎯 Auto-selecting single session: {sessions[0].name}")
        return sessions

    print(f"  [A] Analyze ALL {len(sessions)} sessions")
    print(f"  [Q] Quit")

    while True:
        choice = (
            input(f"\n🔢 Select option [1-{len(sessions)}, A, Q]: ").strip().upper()
        )

        if choice == "Q":
            return []
        elif choice == "A":
            return sessions
        else:
            try:
                session_idx = int(choice) - 1
                if 0 <= session_idx < len(sessions):
                    return [sessions[session_idx]]
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number, 'A', or 'Q'.")


def scan_data_directory():
    """Scan the data directory and return game directories and their sessions."""
    data_dir = Path("data")
    if not data_dir.exists():
        return {}

    games = {}
    for game_dir in data_dir.iterdir():
        if game_dir.is_dir() and not game_dir.name.startswith("."):
            sessions = list_sessions(game_dir)
            if sessions:
                games[game_dir.name] = {"path": game_dir, "sessions": sessions}

    return games


def interactive_game_and_session_selection():
    """Interactive selection of game and sessions from data directory."""
    games = scan_data_directory()

    if not games:
        print("❌ No games found in data/ directory")
        return None, []

    # Sort games alphabetically
    sorted_games = sorted(games.keys())

    print("\n🎮 Available Games:")
    for i, game_name in enumerate(sorted_games, 1):
        session_count = len(games[game_name]["sessions"])
        print(
            f"  [{i}] {game_name} ({session_count} session{'s' if session_count != 1 else ''})"
        )

    # Game selection
    while True:
        choice = input(f"\n🔢 Select game [1-{len(sorted_games)}]: ").strip()
        try:
            game_idx = int(choice) - 1
            if 0 <= game_idx < len(sorted_games):
                selected_game = sorted_games[game_idx]
                break
            else:
                print("❌ Invalid selection. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

    game_data = games[selected_game]
    sessions = game_data["sessions"]

    # Sort sessions by timestamp (newest first)
    sessions.sort(key=lambda x: x.name, reverse=True)

    print(f"\n📂 Sessions for {selected_game}:")
    for i, session in enumerate(sessions, 1):
        # Try to get session timestamp from folder name
        print(f"  [{i}] {session.name}")

    if len(sessions) == 1:
        print(f"\n🎯 Auto-selecting single session: {sessions[0].name}")
        return selected_game, sessions

    print(f"  [A] Analyze ALL {len(sessions)} sessions")
    print(f"  [L] Analyze LATEST session")
    print(f"  [Q] Quit")

    # Session selection
    while True:
        choice = (
            input(f"\n🔢 Select option [1-{len(sessions)}, A, L, Q]: ").strip().upper()
        )

        if choice == "Q":
            return selected_game, []
        elif choice == "A":
            return selected_game, sessions
        elif choice == "L":
            return selected_game, [sessions[0]]  # Latest session
        else:
            try:
                session_idx = int(choice) - 1
                if 0 <= session_idx < len(sessions):
                    return selected_game, [sessions[session_idx]]
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number, 'A', 'L', or 'Q'.")


async def interactive_mode():
    """Interactive mode for analysis configuration."""
    print_banner()
    print("🔧 Interactive Analysis Mode")
    print("=" * 40)

    # Get detector type
    print("\n🎯 Analysis Method Selection:")
    print("  [1] Grounding DINO + SAM 2 (Recommended for known games)")
    print("  [2] YOLO + SAM 2 (Faster, traditional detection)")
    print("  [3] Universal Pipeline (🌟 Works with ANY game!)")
    print("  [4] Compare detectors (1 & 2)")

    while True:
        detector_choice = input("\n🔢 Select method [1-4]: ").strip()
        if detector_choice == "1":
            detector_type = "grounded_sam"
            break
        elif detector_choice == "2":
            detector_type = "yolo"
            break
        elif detector_choice == "3":
            detector_type = "universal"
            break
        elif detector_choice == "4":
            detector_type = "comparison"
            break
        else:
            print("❌ Please enter 1, 2, 3, or 4.")

    # Get analysis type
    if detector_type != "comparison":
        # Automatically scan data directory and let user select
        game_name, selected_sessions = interactive_game_and_session_selection()

        if not selected_sessions:
            print("❌ No sessions selected")
            return

        if len(selected_sessions) == 1:
            await analyze_single_session(selected_sessions[0], detector_type, game_name)
        else:
            await analyze_multiple_sessions(selected_sessions, detector_type, game_name)

    else:  # Comparison mode
        test_dir = input("\n📸 Enter test images directory: ").strip()
        if not test_dir:
            print("❌ Test images directory is required")
            return

        test_dir = Path(test_dir)
        if not test_dir.exists():
            print(f"❌ Directory not found: {test_dir}")
            return

        game_name = input(
            f"\n🎮 Enter game name ({', '.join(get_supported_games())}): "
        ).strip()
        if not game_name:
            game_name = "generic_runner"

        output_dir = input(
            "\n📁 Enter output directory (default: ./comparison_results): "
        ).strip()
        if not output_dir:
            output_dir = "./comparison_results"

        await run_detector_comparison(test_dir, game_name, Path(output_dir))


async def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced game video analysis with multiple detector backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single session with Grounding DINO
  python run_improved_analysis.py session_path --detector grounded_sam

  # Analyze all sessions in a game directory with YOLO
  python run_improved_analysis.py game_dir --detector yolo --all-sessions

  # Universal analysis (works with ANY game!)
  python run_improved_analysis.py session_path --detector universal

  # Compare detectors on test images
  python run_improved_analysis.py --compare test_images_dir --game subway_surfers

  # Interactive mode
  python run_improved_analysis.py
        """,
    )

    parser.add_argument("path", nargs="?", help="Path to session or game directory")
    parser.add_argument(
        "--detector",
        choices=["grounded_sam", "yolo", "universal"],
        default="grounded_sam",
        help="Analysis method to use",
    )
    parser.add_argument("--game", help="Game name for context-specific analysis")
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Analyze all sessions in the specified directory",
    )
    parser.add_argument(
        "--compare",
        metavar="TEST_DIR",
        help="Run detector comparison on test images directory",
    )
    parser.add_argument(
        "--output",
        default="./comparison_results",
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    # Check if in comparison mode
    if args.compare:
        test_dir = Path(args.compare)
        if not test_dir.exists():
            print(f"❌ Test images directory not found: {test_dir}")
            return

        game_name = args.game or "generic_runner"
        output_dir = Path(args.output)

        await run_detector_comparison(test_dir, game_name, output_dir)
        return

    # Check if path provided
    if not args.path:
        # Interactive mode
        await interactive_mode()
        return

    # Path-based analysis
    path = Path(args.path)
    if not path.exists():
        print(f"❌ Path not found: {path}")
        return

    if args.all_sessions:
        # Analyze all sessions in directory
        sessions = list_sessions(path)
        if not sessions:
            print(f"❌ No sessions found in {path}")
            return

        await analyze_multiple_sessions(sessions, args.detector, args.game)
    else:
        # Check if path is a session or game directory
        if get_video_file(path) is not None:
            # It's a session directory
            await analyze_single_session(path, args.detector, args.game)
        else:
            # It's likely a game directory
            sessions = interactive_session_selection(path)
            if sessions:
                await analyze_multiple_sessions(sessions, args.detector, args.game)


if __name__ == "__main__":
    asyncio.run(main())
