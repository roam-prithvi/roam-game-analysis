"""
Universal Grounded SAM 2 pipeline that works with any game.

This module integrates:
1. Video behavior analysis with Gemini to extract objects
2. Universal object detection using Grounded DINO + SAM 2
3. Frame-by-frame object extraction and segmentation
4. Scalable object detection without game-specific prompts
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

# Import our modules
from .video_analyzer import VideoGameplayAnalyzer
from .universal_object_extractor import UniversalObjectExtractor
from .objective_generator import VideoAnalysisObjectiveGenerator

# Add Grounded-SAM-2 to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "Grounded-SAM-2"))


class UniversalGroundedSAM2Pipeline:
    """Universal pipeline for game object detection and segmentation."""

    def __init__(
        self,
        session_path: Path,
        max_frames: int = 100,
        frame_interval: int = 30,
        generate_objectives: bool = True,
    ):
        """
        Initialize the universal pipeline.

        Args:
            session_path: Path to game session directory
            max_frames: Maximum number of frames to analyze
            frame_interval: Interval between frames (every N frames)
            generate_objectives: Whether to generate game objectives from video analysis
        """
        self.session_path = Path(session_path)
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.generate_objectives = generate_objectives

        # Paths
        # Check for trimmed video first, fallback to original
        trimmed_video = self.session_path / "trimmed_screen_recording.mp4"
        if trimmed_video.exists():
            self.video_path = trimmed_video
        else:
            self.video_path = self.session_path / "screen_recording.mp4"
        self.frames_dir = self.session_path / "frames"
        self.analysis_dir = self.session_path / "analysis" / "universal_grounded_sam2"

        # Initialize components
        self.video_analyzer = None
        self.object_extractor = None
        self.gemini_objects: Set[str] = set()

        # Results
        self.analysis_results = {}

        # Ensure directories exist
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎮 Universal Grounded SAM 2 Pipeline initialized")
        print(f"📁 Session: {self.session_path.name}")
        print(f"🎯 Max frames: {max_frames}, Interval: {frame_interval}")

    def extract_frames(self) -> List[Path]:
        """Extract frames from the video at specified intervals."""
        print(f"🎞️ Extracting frames from video...")

        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Use FFmpeg to extract frames
        frame_paths = []

        try:
            # Get video info
            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"📊 Video info: {total_frames} frames @ {fps} FPS")

            # Calculate frame extraction parameters
            frames_to_extract = min(
                self.max_frames, total_frames // self.frame_interval
            )
            frame_step = max(1, total_frames // frames_to_extract)

            print(f"🎯 Extracting {frames_to_extract} frames with step {frame_step}")

            # Extract frames using OpenCV for precise control
            cap = cv2.VideoCapture(str(self.video_path))
            frame_count = 0
            extracted_count = 0

            while cap.isOpened() and extracted_count < frames_to_extract:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_step == 0:
                    # Save frame
                    frame_filename = f"frame_{extracted_count:04d}.jpg"
                    frame_path = self.frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1

                frame_count += 1

            cap.release()

            print(f"✅ Extracted {len(frame_paths)} frames to {self.frames_dir}")
            return frame_paths

        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            return []

    async def run_video_analysis(self) -> str:
        """Run Gemini video analysis to extract objects."""
        print(f"🧠 Running Gemini video analysis...")

        try:
            # Initialize video analyzer
            self.video_analyzer = VideoGameplayAnalyzer([self.session_path])

            # Run analysis
            analysis_text = self.video_analyzer.run_analysis()

            # Extract objects from analysis
            self.gemini_objects = self.video_analyzer.extracted_objects.copy()

            print(f"✅ Video analysis complete!")
            print(f"🎯 Extracted {len(self.gemini_objects)} objects from analysis")

            return analysis_text

        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return ""

    async def generate_game_objectives(self) -> Optional[str]:
        """Generate game objectives from the video analysis."""
        if not self.generate_objectives:
            return None

        print(f"🎯 Generating game objectives...")

        try:
            # Find the most recent analysis file
            analysis_dir = self.session_path / "analysis" / "video_behavior_analysis"
            if not analysis_dir.exists():
                print(f"⚠️ No video analysis directory found for objective generation")
                return None

            # Look for detailed analysis files
            analysis_files = list(analysis_dir.glob("detailed_analysis_*.txt"))
            if not analysis_files:
                print(f"⚠️ No detailed analysis files found for objective generation")
                return None

            # Use the most recent analysis file
            latest_analysis = max(analysis_files, key=lambda f: f.stat().st_mtime)
            print(f"📄 Using analysis file: {latest_analysis.name}")

            # Generate objectives
            objective_generator = VideoAnalysisObjectiveGenerator(latest_analysis)
            result = objective_generator.generate_objectives()

            print(f"✅ Objective generation complete!")

            return result.get("natural_language_prompt")

        except Exception as e:
            print(f"❌ Objective generation failed: {e}")
            return None

    def setup_grounded_sam2(self) -> bool:
        """Set up Grounded SAM 2 environment."""
        print(f"🔧 Setting up Grounded SAM 2...")

        try:
            grounded_sam_dir = (
                Path(__file__).parent.parent.parent.parent / "Grounded-SAM-2"
            )

            # Check if required files exist
            demo_script = grounded_sam_dir / "grounded_sam2_custom_demo.py"
            if not demo_script.exists():
                print(f"❌ Grounded SAM 2 demo script not found: {demo_script}")
                return False

            # Check for model checkpoints
            checkpoints_dir = grounded_sam_dir / "checkpoints"
            if not checkpoints_dir.exists():
                print(f"⚠️ Checkpoints directory not found: {checkpoints_dir}")
                print(f"📥 Please download SAM 2 checkpoints first")
                return False

            print(f"✅ Grounded SAM 2 setup verified")
            return True

        except Exception as e:
            print(f"❌ Grounded SAM 2 setup failed: {e}")
            return False

    def create_universal_prompt(self, frame_idx: int = 0) -> str:
        """Create a universal detection prompt."""
        # Combine multiple strategies
        prompts = set()

        # Strategy 1: Add Gemini-discovered objects
        if self.gemini_objects:
            prompts.update(
                list(self.gemini_objects)[:15]
            )  # Limit to avoid overly long prompts

        # Strategy 2: Add universal game object prompts
        universal_prompts = [
            "character",
            "player",
            "enemy",
            "monster",
            "opponent",
            "obstacle",
            "barrier",
            "wall",
            "platform",
            "block",
            "collectible",
            "coin",
            "gem",
            "item",
            "powerup",
            "button",
            "menu",
            "icon",
            "score",
            "health",
            "vehicle",
            "car",
            "train",
            "ship",
            "plane",
            "weapon",
            "tool",
            "door",
            "key",
            "chest",
        ]
        prompts.update(universal_prompts[:10])  # Add top universal prompts

        # Strategy 3: Add context-specific prompts based on frame analysis
        # (This could be enhanced with frame-specific analysis)

        # Create final prompt
        prompt_list = list(prompts)[:25]  # Limit to 25 prompts maximum
        return " . ".join(prompt_list) + " ."

    def run_grounded_sam2_on_frame(self, frame_path: Path, prompt: str) -> Dict:
        """Run Grounded SAM 2 detection on a single frame."""
        grounded_sam_dir = Path(__file__).parent.parent.parent.parent / "Grounded-SAM-2"
        demo_script = grounded_sam_dir / "grounded_sam2_custom_demo.py"

        # Create output directory for this frame
        frame_output_dir = self.analysis_dir / "frame_detections" / frame_path.stem
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize temp_link_path
        temp_link_path = None

        try:
            # Create absolute path to the file
            abs_frame_path = frame_path.absolute()
            abs_output_dir = frame_output_dir.absolute()

            # WORKAROUND: If path contains spaces, create a temporary symlink
            if " " in str(abs_frame_path):
                # Create a temporary directory for symlinks
                temp_dir = self.analysis_dir / "temp_links"
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Create a symlink without spaces
                temp_link_path = temp_dir / f"frame_{frame_path.stem}.jpg"
                try:
                    # Remove old symlink if it exists
                    if temp_link_path.exists() or temp_link_path.is_symlink():
                        temp_link_path.unlink()
                    # Create new symlink
                    temp_link_path.symlink_to(abs_frame_path)
                    abs_frame_path = temp_link_path.absolute()
                    print(f"  Created temp symlink: {temp_link_path} -> {frame_path}")
                except Exception as e:
                    print(f"  Warning: Could not create symlink: {e}")
                    temp_link_path = None

            # Run Grounded SAM 2
            cmd = [
                sys.executable,
                str(demo_script),
                "--text-prompt",
                prompt,
                "--img-path",
                str(abs_frame_path),
                "--output-dir",
                str(abs_output_dir),
                "--box-threshold",
                "0.25",  # Lower threshold for more detections
                "--text-threshold",
                "0.20",
            ]

            print(
                f"🔍 Running detection on {frame_path.name} with prompt: {prompt[:100]}..."
            )

            # Debug: Print the exact command being run
            print(f"  Command: {' '.join(cmd)}")

            # Use absolute path for cwd as well
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(grounded_sam_dir.absolute()),
            )

            if result.returncode != 0:
                print(f"⚠️ Detection failed for {frame_path.name}: {result.stderr}")
                # Also print stdout in case there's useful info there
                if result.stdout:
                    print(f"  Stdout: {result.stdout[:500]}...")

                # Clean up temp symlink if created
                if temp_link_path and (
                    temp_link_path.exists() or temp_link_path.is_symlink()
                ):
                    temp_link_path.unlink()

                return {"success": False, "error": result.stderr}

            # Load results
            results_file = frame_output_dir / "grounded_sam2_custom_demo_results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    detection_results = json.load(f)

                print(
                    f"✅ Found {len(detection_results.get('annotations', []))} objects in {frame_path.name}"
                )

                # Clean up temp symlink if created
                if temp_link_path and (
                    temp_link_path.exists() or temp_link_path.is_symlink()
                ):
                    temp_link_path.unlink()

                return {
                    "success": True,
                    "frame_path": str(frame_path),
                    "output_dir": str(frame_output_dir),
                    "detections": detection_results,
                    "prompt_used": prompt,
                }
            else:
                print(f"⚠️ No results file found for {frame_path.name}")
                result_dict = {"success": False, "error": "No results file generated"}

            # Clean up temp symlink if created
            if temp_link_path and (
                temp_link_path.exists() or temp_link_path.is_symlink()
            ):
                temp_link_path.unlink()

            return result_dict

        except Exception as e:
            print(f"❌ Detection error for {frame_path.name}: {e}")

            # Clean up temp symlink if created
            if temp_link_path and (
                temp_link_path.exists() or temp_link_path.is_symlink()
            ):
                temp_link_path.unlink()

            return {"success": False, "error": str(e)}

    def categorize_detections(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize detections into game object types."""
        categories = {
            "player": [],
            "enemies": [],
            "obstacles": [],
            "collectibles": [],
            "ui": [],
            "environment": [],
            "interactive": [],
            "unknown": [],
        }

        for detection in detections:
            class_name = detection.get("class_name", "").lower()
            bbox = detection.get("bbox", [0, 0, 0, 0])

            # Simple categorization based on class name
            category = "unknown"

            if any(
                keyword in class_name
                for keyword in ["character", "player", "hero", "avatar"]
            ):
                category = "player"
            elif any(
                keyword in class_name
                for keyword in ["enemy", "monster", "opponent", "zombie", "alien"]
            ):
                category = "enemies"
            elif any(
                keyword in class_name
                for keyword in ["obstacle", "barrier", "wall", "block", "train", "car"]
            ):
                category = "obstacles"
            elif any(
                keyword in class_name
                for keyword in [
                    "coin",
                    "gem",
                    "collectible",
                    "item",
                    "powerup",
                    "bonus",
                ]
            ):
                category = "collectibles"
            elif any(
                keyword in class_name
                for keyword in ["button", "menu", "icon", "score", "health", "timer"]
            ):
                category = "ui"
            elif any(
                keyword in class_name
                for keyword in ["building", "tree", "platform", "ground", "background"]
            ):
                category = "environment"
            elif any(
                keyword in class_name
                for keyword in ["door", "lever", "switch", "tool", "weapon", "key"]
            ):
                category = "interactive"

            categories[category].append(detection)

        return categories

    def save_frame_analysis(
        self, frame_results: List[Dict], objectives_prompt: Optional[str] = None
    ) -> str:
        """Save comprehensive frame analysis results."""
        summary_file = self.analysis_dir / "universal_analysis_summary.json"

        # Compile statistics
        total_detections = 0
        category_totals = {}
        all_classes = set()

        for frame_result in frame_results:
            if frame_result.get("success"):
                detections = frame_result.get("detections", {}).get("annotations", [])
                total_detections += len(detections)

                # Categorize this frame's detections
                categorized = self.categorize_detections(detections)

                for category, objects in categorized.items():
                    if category not in category_totals:
                        category_totals[category] = 0
                    category_totals[category] += len(objects)

                    for obj in objects:
                        all_classes.add(obj.get("class_name", "unknown"))

        # Create summary
        summary = {
            "analysis_type": "universal_grounded_sam2",
            "session_path": str(self.session_path),
            "video_path": str(self.video_path),
            "frames_analyzed": len(frame_results),
            "successful_frames": sum(1 for r in frame_results if r.get("success")),
            "total_objects_detected": total_detections,
            "unique_object_classes": len(all_classes),
            "categories_found": len(
                [c for c, count in category_totals.items() if count > 0]
            ),
            "gemini_discovered_objects": list(self.gemini_objects),
            "generated_objectives": objectives_prompt,
            "category_statistics": category_totals,
            "detected_classes": list(all_classes),
            "frame_results": frame_results,
            "processing_parameters": {
                "max_frames": self.max_frames,
                "frame_interval": self.frame_interval,
                "generate_objectives": self.generate_objectives,
                "detection_thresholds": {"box_threshold": 0.25, "text_threshold": 0.20},
            },
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📊 Analysis summary saved to: {summary_file}")
        return str(summary_file)

    async def run_complete_pipeline(self) -> str:
        """Run the complete universal pipeline."""
        print(f"\n🚀 Starting Universal Grounded SAM 2 Pipeline")
        print("=" * 60)

        try:
            # Step 1: Extract frames
            frame_paths = self.extract_frames()
            if not frame_paths:
                raise Exception("No frames extracted from video")

            # Step 2: Run video analysis to discover objects
            analysis_text = await self.run_video_analysis()

            # Step 2.5: Generate game objectives (optional)
            objectives_prompt = await self.generate_game_objectives()

            # Step 3: Set up Grounded SAM 2
            if not self.setup_grounded_sam2():
                raise Exception("Failed to set up Grounded SAM 2")

            # Step 4: Process each frame
            frame_results = []

            print(f"\n🔍 Processing {len(frame_paths)} frames...")

            for i, frame_path in enumerate(frame_paths):
                print(f"\n[{i + 1}/{len(frame_paths)}] Processing {frame_path.name}")

                # Debug: Print current working directory and frame path
                print(f"  Current CWD: {os.getcwd()}")
                print(f"  Frame path (relative): {frame_path}")
                print(f"  Frame path (absolute): {frame_path.absolute()}")
                print(f"  Frame exists: {frame_path.exists()}")

                # Create prompt for this frame
                prompt = self.create_universal_prompt(i)

                # Run detection
                result = self.run_grounded_sam2_on_frame(frame_path, prompt)
                frame_results.append(result)

                # Brief pause between frames
                await asyncio.sleep(0.5)

            # Step 5: Save results
            summary_file = self.save_frame_analysis(frame_results, objectives_prompt)

            # Step 6: Print summary
            successful_frames = sum(1 for r in frame_results if r.get("success"))
            total_detections = sum(
                len(r.get("detections", {}).get("annotations", []))
                for r in frame_results
                if r.get("success")
            )

            print(f"\n🎉 Universal Pipeline Complete!")
            print("=" * 60)
            print(f"✅ Frames processed: {successful_frames}/{len(frame_paths)}")
            print(f"✅ Total objects detected: {total_detections}")
            print(f"✅ Gemini objects discovered: {len(self.gemini_objects)}")
            if objectives_prompt:
                print(f"✅ Game objectives generated")
            print(f"✅ Results saved to: {self.analysis_dir}")
            print(f"📊 Summary file: {summary_file}")

            if objectives_prompt and self.generate_objectives:
                print(f"\n🎯 Generated Objectives Prompt:")
                print("=" * 60)
                print(objectives_prompt)
                print("=" * 60)

            return summary_file

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise


async def main():
    """CLI entry point for the universal pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Grounded SAM 2 pipeline for game analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single session (includes objective generation)
  python universal_grounded_sam2_pipeline.py data/GameName/session_folder

  # Analyze with custom parameters
  python universal_grounded_sam2_pipeline.py data/GameName/session_folder --max-frames 50 --frame-interval 15

  # Skip objective generation for faster processing
  python universal_grounded_sam2_pipeline.py data/GameName/session_folder --no-objectives
        """,
    )

    parser.add_argument("session_path", help="Path to game session directory")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to analyze (default: 100)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Frame interval for extraction (default: 30)",
    )
    parser.add_argument(
        "--no-objectives",
        action="store_true",
        help="Skip objective generation (default: generate objectives)",
    )

    args = parser.parse_args()

    # Validate session path
    session_path = Path(args.session_path)
    if not session_path.exists():
        print(f"❌ Session path not found: {session_path}")
        return

    # Check for video file (trimmed or original)
    trimmed_video = session_path / "trimmed_screen_recording.mp4"
    original_video = session_path / "screen_recording.mp4"

    if not trimmed_video.exists() and not original_video.exists():
        print(f"❌ No video file found in session: {session_path}")
        return

    # Run pipeline
    try:
        pipeline = UniversalGroundedSAM2Pipeline(
            session_path,
            max_frames=args.max_frames,
            frame_interval=args.frame_interval,
            generate_objectives=not args.no_objectives,
        )

        summary_file = await pipeline.run_complete_pipeline()
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Check results in: {pipeline.analysis_dir}")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
