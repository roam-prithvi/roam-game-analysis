"""
Improved game-aware video analyzer using the detector abstraction layer.

This module can use either YOLO or Grounding DINO + SAM 2 for detection,
with game-specific prompts and intelligent filtering.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

# Import detector abstractions
from .base_detector import BaseDetector, Detection, DetectionConfig
from .grounded_sam_detector import GroundedSAMDetector, create_grounded_sam_detector
from .yolo_detector import YOLODetector, create_yolo_detector
from .game_prompts import (
    get_game_prompts,
    get_primary_game_prompts,
    get_supported_games,
)

# Import utilities
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.processing.frame_cutter import process_directory as extract_frames
from src.util import list_sessions


class ImprovedGameAwareAnalyzer:
    """Advanced game-aware analyzer supporting multiple detection backends."""

    def __init__(
        self,
        session_path: Union[str, Path],
        detector_type: str = "grounded_sam",
        game_name: Optional[str] = None,
        detector_config: Optional[DetectionConfig] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            session_path: Path to session directory
            detector_type: Type of detector ("grounded_sam", "yolo")
            game_name: Name of the game (auto-detect if None)
            detector_config: Custom detector configuration
        """
        self.session_path = Path(session_path)

        # Check for trimmed video first, fallback to original
        trimmed_video = self.session_path / "trimmed_screen_recording.mp4"
        if trimmed_video.exists():
            self.video_path = trimmed_video
        else:
            self.video_path = self.session_path / "screen_recording.mp4"
        self.frames_dir = self.session_path / "frames"
        self.analysis_dir = (
            self.session_path / "analysis" / f"{detector_type}_game_aware"
        )

        # Detect or set game name
        self.game_name = game_name or self._detect_game_name()
        print(f"🎮 Game detected: {self.game_name}")

        # Detector configuration
        self.detector_type = detector_type
        self.detector_config = detector_config or DetectionConfig(
            confidence_threshold=0.35,
            max_detections=50,
            min_object_size=10,
            enable_segmentation=True,
        )

        # Initialize detector
        self.detector: Optional[BaseDetector] = None

        # Game-specific prompts
        self.game_prompts = get_game_prompts(self.game_name)
        self.primary_prompts = get_primary_game_prompts(self.game_name)

        print(f"🎯 Using {len(self.game_prompts)} game-specific prompts")
        print(f"🔧 Detector type: {detector_type}")

        # Create directories
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def _detect_game_name(self) -> str:
        """Auto-detect game name from session path."""
        # Check parent directory names for game hints
        path_parts = [p.name.lower() for p in self.session_path.parents] + [
            self.session_path.name.lower()
        ]

        # Check against known games
        supported_games = get_supported_games()
        for game in supported_games:
            game_words = game.replace("_", " ").split()
            for part in path_parts:
                if any(word in part for word in game_words):
                    return game

        # Fallback detection based on common patterns
        for part in path_parts:
            if "subway" in part or "surfer" in part:
                return "subway_surfers"
            elif "temple" in part or "run" in part:
                return "temple_run"
            elif "crossy" in part or "road" in part:
                return "crossy_road"
            elif "plant" in part or "zombie" in part or "pvz" in part:
                if "2" in part:
                    return "plants_vs_zombies_2"
                else:
                    return "plants_vs_zombies"

        print("⚠️ Could not auto-detect game, using generic prompts")
        return "generic_runner"

    async def initialize_detector(self):
        """Initialize the selected detector."""
        if self.detector and self.detector.is_initialized:
            return

        print(f"🔧 Initializing {self.detector_type} detector...")

        if self.detector_type == "grounded_sam":
            self.detector = await create_grounded_sam_detector(
                confidence_threshold=self.detector_config.confidence_threshold,
                enable_segmentation=self.detector_config.enable_segmentation,
            )
        elif self.detector_type == "yolo":
            self.detector = await create_yolo_detector(
                confidence_threshold=self.detector_config.confidence_threshold,
                use_sam_segmentation=self.detector_config.enable_segmentation,
            )
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

        print("✅ Detector initialized successfully")

    async def detect_objects_in_frame(self, frame_path: str) -> List[Detection]:
        """Detect game objects in a single frame."""
        if not self.detector:
            await self.initialize_detector()

        # Use game-specific prompts for text-based detectors
        if self.detector.get_supported_features().get("text_prompts", False):
            # For Grounding DINO, use primary prompts to avoid false positives
            detections = await self.detector.detect(
                frame_path, prompts=self.primary_prompts, game_name=self.game_name
            )
        else:
            # For YOLO, detect all objects then filter
            detections = await self.detector.detect(
                frame_path,
                prompts=self.game_prompts,  # Used for filtering
            )

        return detections

    def categorize_detections(
        self, detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """Categorize detections by game object type."""
        categories = {
            "player": [],
            "enemies": [],
            "obstacles": [],
            "collectibles": [],
            "ui": [],
            "environment": [],
            "unknown": [],
        }

        # Simple categorization based on class names
        # This can be expanded with more sophisticated game-specific logic
        for det in detections:
            class_name = det.class_name.lower()
            categorized = False

            # Player detection
            if any(
                word in class_name for word in ["character", "player", "person", "hero"]
            ):
                categories["player"].append(det)
                categorized = True

            # Enemy detection
            elif any(
                word in class_name
                for word in ["zombie", "enemy", "monster", "opponent"]
            ):
                categories["enemies"].append(det)
                categorized = True

            # Obstacle detection
            elif any(
                word in class_name
                for word in ["obstacle", "barrier", "train", "car", "truck"]
            ):
                categories["obstacles"].append(det)
                categorized = True

            # Collectible detection
            elif any(
                word in class_name
                for word in ["coin", "gem", "collectible", "power", "bonus"]
            ):
                categories["collectibles"].append(det)
                categorized = True

            # UI elements
            elif any(
                word in class_name
                for word in ["button", "menu", "score", "ui", "interface"]
            ):
                categories["ui"].append(det)
                categorized = True

            # Environment
            elif any(
                word in class_name
                for word in ["background", "platform", "ground", "building"]
            ):
                categories["environment"].append(det)
                categorized = True

            if not categorized:
                categories["unknown"].append(det)

        return categories

    def merge_similar_detections(
        self, detections: List[Detection], iou_threshold: float = 0.3
    ) -> List[Detection]:
        """Merge overlapping detections of similar objects."""
        if len(detections) <= 1:
            return detections

        # Group by class name
        class_groups = {}
        for det in detections:
            class_name = det.class_name
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)

        merged = []

        # Merge within each class
        for class_name, group in class_groups.items():
            if len(group) == 1:
                merged.extend(group)
                continue

            # Sort by confidence
            group.sort(key=lambda x: x.confidence, reverse=True)

            kept = []
            for det in group:
                should_merge = False
                merge_target = None

                # Check overlap with kept detections
                for kept_det in kept:
                    iou = self._calculate_iou(det.bbox, kept_det.bbox)
                    if iou > iou_threshold:
                        should_merge = True
                        merge_target = kept_det
                        break

                if should_merge and merge_target:
                    # Merge bounding boxes
                    merged_bbox = self._merge_bboxes(det.bbox, merge_target.bbox)
                    merge_target.bbox = merged_bbox
                    merge_target.confidence = max(
                        det.confidence, merge_target.confidence
                    )
                else:
                    kept.append(det)

            merged.extend(kept)

        return merged

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """Merge two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        return [
            min(x1_min, x2_min),
            min(y1_min, y2_min),
            max(x1_max, x2_max),
            max(y1_max, y2_max),
        ]

    def save_detection_cutouts(
        self,
        frame_path: str,
        categorized_detections: Dict[str, List[Detection]],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        """Save cutouts of detected objects organized by category."""
        cutout_info = []
        frame_name = Path(frame_path).stem

        for category, detections in categorized_detections.items():
            if not detections:
                continue

            # Create category directory
            category_dir = self.analysis_dir / "segmented_assets" / category
            category_dir.mkdir(parents=True, exist_ok=True)

            for i, det in enumerate(detections):
                try:
                    if det.mask is not None:
                        # Extract cutout using mask
                        output_name = f"{frame_name}_{category}_{det.class_name}_{i}_{det.confidence:.2f}.png"
                        output_path = category_dir / output_name

                        if self._extract_cutout_with_mask(
                            frame_path, det.mask, str(output_path)
                        ):
                            cutout_info.append(
                                {
                                    "category": category,
                                    "class_name": det.class_name,
                                    "confidence": det.confidence,
                                    "bbox": det.bbox,
                                    "cutout_path": str(output_path),
                                    "frame_idx": frame_idx,
                                }
                            )
                            print(f"✅ Saved {category}/{det.class_name} cutout")
                    else:
                        # Extract using bounding box only
                        output_name = f"{frame_name}_{category}_{det.class_name}_{i}_{det.confidence:.2f}.png"
                        output_path = category_dir / output_name

                        if self._extract_cutout_with_bbox(
                            frame_path, det.bbox, str(output_path)
                        ):
                            cutout_info.append(
                                {
                                    "category": category,
                                    "class_name": det.class_name,
                                    "confidence": det.confidence,
                                    "bbox": det.bbox,
                                    "cutout_path": str(output_path),
                                    "frame_idx": frame_idx,
                                }
                            )
                            print(f"✅ Saved {category}/{det.class_name} bbox cutout")

                except Exception as e:
                    print(f"⚠️ Failed to save cutout for {det.class_name}: {e}")

        return cutout_info

    def _extract_cutout_with_mask(
        self, image_path: str, mask: np.ndarray, output_path: str
    ) -> bool:
        """Extract cutout using segmentation mask."""
        try:
            # Load original image
            original = Image.open(image_path).convert("RGBA")

            # Debug: Print mask info
            print(
                f"🔧 Debug mask - shape: {mask.shape}, dtype: {mask.dtype}, range: {mask.min()}-{mask.max()}"
            )

            # Check if mask is completely empty
            if mask.sum() == 0:
                print(f"⚠️ Mask is completely empty (all zeros), skipping cutout")
                return False

            # Convert mask to proper format
            if mask.dtype == bool:
                # Boolean mask - convert to 0-255
                mask = mask.astype(np.uint8) * 255
            elif mask.dtype == np.float32 or mask.dtype == np.float64:
                # Float mask - ensure it's in 0-1 range then scale to 0-255
                mask = np.clip(mask, 0, 1)
                mask = (mask * 255).astype(np.uint8)
            elif mask.dtype == np.uint8:
                # Check if mask values are in 0-1 range (should be 0-255)
                if mask.max() <= 1:
                    print(f"⚠️ Mask appears to be in 0-1 range, scaling to 0-255")
                    mask = mask * 255
                elif mask.max() <= 10:
                    # Sometimes masks are in 0-10 range or similar low values
                    print(
                        f"⚠️ Mask appears to be in low range (max={mask.max()}), scaling to 0-255"
                    )
                    mask = (mask / mask.max() * 255).astype(np.uint8)

            # Final check - ensure mask has meaningful values
            if mask.max() == 0:
                print(
                    f"⚠️ Mask has no positive values after processing, skipping cutout"
                )
                return False

            print(f"🔧 After processing - mask range: {mask.min()}-{mask.max()}")

            # Convert to PIL Image
            mask_img = Image.fromarray(mask, mode="L")

            # Resize mask if needed
            if mask_img.size != original.size:
                mask_img = mask_img.resize(original.size, Image.Resampling.LANCZOS)

            # Apply mask as alpha channel
            original_array = np.array(original)
            mask_array = np.array(mask_img)

            # Ensure we're working with the right dimensions
            if len(mask_array.shape) == 2:
                original_array[:, :, 3] = mask_array
            else:
                print(f"⚠️ Unexpected mask dimensions: {mask_array.shape}")
                return False

            # Debug final alpha stats
            final_alpha = original_array[:, :, 3]
            non_transparent = (final_alpha > 0).sum()
            total_pixels = final_alpha.shape[0] * final_alpha.shape[1]
            non_transparent_pct = 100 * non_transparent / total_pixels

            print(
                f"🔧 Final alpha - min: {final_alpha.min()}, max: {final_alpha.max()}, "
                f"non-transparent: {non_transparent}/{total_pixels} ({non_transparent_pct:.1f}%)"
            )

            # Skip cutouts that are too small (likely noise)
            if non_transparent_pct < 0.01:  # Less than 0.01% of image
                print(
                    f"⚠️ Cutout too small ({non_transparent_pct:.3f}%), likely noise - skipping"
                )
                return False

            # Save cutout
            cutout = Image.fromarray(original_array, "RGBA")
            cutout.save(output_path, "PNG")

            return True

        except Exception as e:
            print(f"⚠️ Mask cutout extraction failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _extract_cutout_with_bbox(
        self, image_path: str, bbox: List[float], output_path: str
    ) -> bool:
        """Extract cutout using bounding box."""
        try:
            # Load image
            image = Image.open(image_path)

            # Crop to bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cropped = image.crop((x1, y1, x2, y2))

            # Convert to RGBA for consistency
            if cropped.mode != "RGBA":
                cropped = cropped.convert("RGBA")

            # Save
            cropped.save(output_path, "PNG")

            return True

        except Exception as e:
            print(f"⚠️ Bbox cutout extraction failed: {e}")
            return False

    async def process_frame(self, frame_path: str, frame_idx: int) -> Dict[str, Any]:
        """Process a single frame with advanced game-aware detection."""
        print(f"\n📸 Processing frame {frame_idx}: {Path(frame_path).name}")

        # Detect objects
        detections = await self.detect_objects_in_frame(frame_path)
        print(f"🔍 Detected {len(detections)} objects")

        if not detections:
            return {
                "frame": frame_path,
                "frame_idx": frame_idx,
                "timestamp": frame_idx / 30.0,
                "detections": 0,
                "categories": {},
                "objects": [],
            }

        # Merge similar detections
        merged_detections = self.merge_similar_detections(detections)
        print(f"🔗 Merged to {len(merged_detections)} unique objects")

        # Categorize detections
        categorized = self.categorize_detections(merged_detections)
        print(
            f"📂 Categorized into {sum(1 for cats in categorized.values() if cats)} categories"
        )

        # Save cutouts
        cutout_info = self.save_detection_cutouts(frame_path, categorized, frame_idx)

        # Compile category statistics
        category_stats = {cat: len(dets) for cat, dets in categorized.items() if dets}

        return {
            "frame": frame_path,
            "frame_idx": frame_idx,
            "timestamp": frame_idx / 30.0,
            "detections": len(merged_detections),
            "categories": category_stats,
            "objects": cutout_info,
        }

    async def analyze_video(self, max_frames: int = 1000, frame_interval: int = 1):
        """Analyze the entire video with improved game-aware detection."""
        print(f"\n🎮 Starting improved game-aware analysis")
        print(f"🎯 Game: {self.game_name}")
        print(f"🔧 Detector: {self.detector_type}")
        print(f"📋 Using {len(self.game_prompts)} game-specific prompts")

        # Ensure detector is initialized
        await self.initialize_detector()

        # Extract frames if needed
        if not self.frames_dir.exists() or not list(self.frames_dir.glob("*.png")):
            print("\n📸 Extracting frames from video...")
            extract_frames(self.session_path)

        # Get frame files
        frame_files = sorted(self.frames_dir.glob("*_time.png"))
        print(f"📁 Found {len(frame_files)} total frames")

        # Sample frames
        if frame_interval == 1:
            sampled_frames = frame_files[:max_frames]
            print(f"📌 Processing {len(sampled_frames)} frames (all frames)")
        else:
            sampled_frames = frame_files[::frame_interval][:max_frames]
            print(
                f"📌 Processing {len(sampled_frames)} sampled frames (every {frame_interval}th frame)"
            )

        # Process frames concurrently (with limit to avoid overwhelming)
        batch_size = 5  # Process 5 frames at a time
        all_results = []

        for i in range(0, len(sampled_frames), batch_size):
            batch = sampled_frames[i : i + batch_size]
            tasks = []

            for idx, frame_path in enumerate(batch):
                global_idx = i + idx
                task = self.process_frame(str(frame_path), global_idx)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)

            print(
                f"⏳ Completed batch {i // batch_size + 1}/{(len(sampled_frames) + batch_size - 1) // batch_size}"
            )

        # Compile final statistics
        total_objects = sum(r["detections"] for r in all_results)
        overall_categories = {}
        class_counts = {}

        for result in all_results:
            for category, count in result["categories"].items():
                overall_categories[category] = (
                    overall_categories.get(category, 0) + count
                )

            for obj in result["objects"]:
                class_name = obj["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Save analysis results
        analysis_output = {
            "game": self.game_name,
            "detector_type": self.detector_type,
            "session": str(self.session_path),
            "frames_processed": len(all_results),
            "total_objects_detected": total_objects,
            "category_distribution": overall_categories,
            "class_distribution": class_counts,
            "game_prompts_used": self.game_prompts,
            "detector_features": self.detector.get_supported_features(),
            "frame_results": all_results,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        output_file = (
            self.analysis_dir
            / f"improved_game_aware_analysis_{self.detector_type}.json"
        )
        with open(output_file, "w") as f:
            json.dump(analysis_output, f, indent=2)

        print(f"\n✅ Analysis complete!")
        print(f"📊 Total objects detected: {total_objects}")
        print(f"📈 Category distribution:")
        for category, count in sorted(
            overall_categories.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   - {category}: {count}")
        print(f"💾 Results saved to: {output_file}")
        print(f"🖼️ Cutouts saved to: {self.analysis_dir / 'segmented_assets'}")

        return analysis_output


async def main():
    """CLI entry point for improved game-aware analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Improved game-aware video analysis")
    parser.add_argument("path", nargs="?", help="Path to session or game directory")
    parser.add_argument(
        "--detector",
        choices=["grounded_sam", "yolo"],
        default="grounded_sam",
        help="Detector type to use",
    )
    parser.add_argument("--game", help="Game name (auto-detect if not specified)")
    parser.add_argument(
        "--max-frames", type=int, default=1000, help="Maximum frames to process"
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Frame sampling interval (1 = all frames)",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Process all sessions in game directory",
    )

    args = parser.parse_args()

    if args.path:
        path = Path(args.path)

        if args.all_sessions and path.is_dir():
            # Process all sessions
            sessions = list_sessions(path)
            if not sessions:
                print(f"❌ No sessions found in {path}")
                return

            print(
                f"📁 Processing {len(sessions)} sessions with {args.detector} detector"
            )

            for session in sessions:
                try:
                    analyzer = ImprovedGameAwareAnalyzer(
                        session, detector_type=args.detector, game_name=args.game
                    )
                    await analyzer.analyze_video(args.max_frames, args.frame_interval)
                except Exception as e:
                    print(f"❌ Error processing {session}: {e}")
                    continue
        else:
            # Process single session
            analyzer = ImprovedGameAwareAnalyzer(
                path, detector_type=args.detector, game_name=args.game
            )
            await analyzer.analyze_video(args.max_frames, args.frame_interval)
    else:
        # Interactive mode
        print(f"🎮 Improved Game-Aware Video Analysis")
        print(f"Supported games: {', '.join(get_supported_games())}")

        game_name = input(
            "\n🎮 Enter game name (or press Enter to auto-detect): "
        ).strip()
        if not game_name:
            game_name = None

        detector_type = input(
            "🔧 Choose detector [grounded_sam/yolo] (default: grounded_sam): "
        ).strip()
        if not detector_type:
            detector_type = "grounded_sam"

        session_path = input("📁 Enter session path: ").strip()
        if not session_path:
            print("❌ Session path is required")
            return

        try:
            analyzer = ImprovedGameAwareAnalyzer(
                session_path, detector_type=detector_type, game_name=game_name
            )
            await analyzer.analyze_video()
        except Exception as e:
            print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
