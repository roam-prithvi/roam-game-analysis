1
"""
Game-aware video analyzer that combines YOLOv11 with game-specific object knowledge.

This module improves upon basic YOLO detection by:
1. Using game-specific object names and categories
2. Filtering YOLO detections based on game context
3. Providing better bounding boxes for SAM 2 segmentation
4. Organizing assets by game-relevant categories
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import sieve
from PIL import Image
from ultralytics import YOLO

# Import utilities
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.processing.frame_cutter import process_directory as extract_frames
from src.util import list_sessions

# Initialize Sieve
sieve.api_key = os.getenv("SIEVE_API_KEY")

# Game-specific object mappings
GAME_OBJECT_MAPPINGS = {
    "subway surfers": {
        "player": ["person", "skateboard"],
        "obstacles": ["train", "barrier", "fence", "traffic light", "bus"],
        "collectibles": ["coin", "star", "gift", "magnet"],
        "environment": ["railroad", "track", "platform", "building"],
        "ui": ["score", "button", "icon"]
    },
    "temple run": {
        "player": ["person", "character"],
        "obstacles": ["tree", "rock", "fire", "gap", "cliff"],
        "collectibles": ["coin", "gem", "powerup"],
        "environment": ["path", "bridge", "temple", "jungle"],
        "ui": ["score", "meter", "button"]
    },
    "crossy road": {
        "player": ["person", "chicken", "character"],
        "obstacles": ["car", "truck", "train", "log", "river"],
        "collectibles": ["coin", "gift"],
        "environment": ["road", "grass", "water", "railroad"],
        "ui": ["score", "button"]
    },
    "pvz": {
        "player": ["plant", "sunflower", "peashooter"],
        "enemies": ["zombie", "person"],
        "projectiles": ["ball", "pea", "projectile"],
        "collectibles": ["sun", "coin"],
        "environment": ["lawn", "grass", "house"],
        "ui": ["button", "meter", "icon"]
    },
    "pvz 2": {
        "player": ["plant", "sunflower", "peashooter"],
        "enemies": ["zombie", "person"],
        "projectiles": ["ball", "pea", "projectile"],
        "collectibles": ["sun", "coin"],
        "environment": ["lawn", "grass", "house"],
        "ui": ["button", "meter", "icon"]
    }
}

# YOLO class names that are relevant for games
RELEVANT_YOLO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
}


class GameAwareAnalyzer:
    """Analyzes game videos with game-specific object detection and segmentation."""
    
    def __init__(self, session_path: str | Path, game_name: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            session_path: Path to session directory
            game_name: Name of the game (will try to auto-detect if not provided)
        """
        self.session_path = Path(session_path)
        self.video_path = self.session_path / "screen_recording.mp4"
        self.frames_dir = self.session_path / "frames"
        self.analysis_dir = self.session_path / "analysis" / "game_aware_segmentation"
        
        # Detect game name from path if not provided
        if game_name is None:
            self.game_name = self._detect_game_name()
        else:
            self.game_name = game_name.lower()
            
        print(f"🎮 Detected game: {self.game_name}")
        
        # Get game-specific mappings
        self.object_mappings = GAME_OBJECT_MAPPINGS.get(
            self.game_name, 
            self._get_default_mappings()
        )
        
        # Create directories
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        self._load_models()
        
    def _detect_game_name(self) -> str:
        """Detect game name from the session path."""
        # Try to get game name from parent directory
        parents = list(self.session_path.parents)
        if len(parents) >= 2:
            potential_game = parents[0].name
            # Check if this matches any known game
            for known_game in GAME_OBJECT_MAPPINGS.keys():
                if known_game in potential_game.lower():
                    return known_game
        return "unknown"
    
    def _get_default_mappings(self) -> Dict[str, List[str]]:
        """Get default object mappings for unknown games."""
        return {
            "player": ["person", "character"],
            "obstacles": ["car", "truck", "barrier"],
            "collectibles": ["coin", "star"],
            "environment": ["road", "building", "tree"],
            "ui": ["button", "icon"]
        }
    
    def _load_models(self):
        """Load YOLOv11 and initialize SAM."""
        # Load YOLO
        model_name = "yolo11m.pt"  # Corrected filename
        model_path = Path(__file__).parent / model_name
        print(f"🔧 Loading YOLOv11 ({model_name})...")
        
        if not model_path.exists():
             print(f"📥 {model_name} not found. YOLO will attempt to download it.")
        
        try:
            self.yolo = YOLO(model_name)
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("💡 Please ensure you have an internet connection for the first-time download.")
            raise
        
        # Initialize SAM
        self.sam = sieve.function.get("sieve/sam2")
        print("✅ Models loaded successfully")
    
    def categorize_detection(self, yolo_class: str) -> Optional[str]:
        """
        Categorize a YOLO detection based on game context.
        
        Args:
            yolo_class: YOLO class name
            
        Returns:
            Game category (player, obstacles, etc.) or None if not relevant
        """
        # Check each category
        for category, class_list in self.object_mappings.items():
            if yolo_class in class_list:
                return category
                
        # For unknown games, use heuristics
        if self.game_name == "unknown":
            if yolo_class == "person":
                return "player"
            elif yolo_class in ["car", "truck", "bus", "train"]:
                return "obstacles"
                
        return None
    
    def filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter YOLO detections based on game relevance.
        
        Args:
            detections: Raw YOLO detections
            
        Returns:
            Filtered detections with game categories
        """
        filtered = []
        
        for det in detections:
            class_name = det['class_name']
            
            # Skip if not a relevant class
            if class_name not in RELEVANT_YOLO_CLASSES:
                continue
                
            # Get game category
            category = self.categorize_detection(class_name)
            if category:
                det['game_category'] = category
                filtered.append(det)
                
        return filtered
    
    def merge_overlapping_boxes(self, 
                               detections: List[Dict[str, Any]], 
                               iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Merge overlapping bounding boxes of the same category.
        
        Args:
            detections: List of detections with bboxes
            iou_threshold: IoU threshold for merging
            
        Returns:
            Merged detections
        """
        if not detections:
            return []
            
        # Group by category
        category_groups = {}
        for det in detections:
            cat = det.get('game_category', 'unknown')
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(det)
        
        merged = []
        
        # Merge within each category
        for category, group in category_groups.items():
            # Sort by confidence
            group.sort(key=lambda x: x['confidence'], reverse=True)
            
            kept = []
            for det in group:
                # Check if this box overlaps with any kept box
                should_merge = False
                merge_target = None
                
                for kept_det in kept:
                    iou = self._calculate_iou(det['bbox'], kept_det['bbox'])
                    if iou > iou_threshold:
                        should_merge = True
                        merge_target = kept_det
                        break
                
                if should_merge and merge_target:
                    # Merge boxes
                    merged_box = self._merge_boxes(det['bbox'], merge_target['bbox'])
                    merge_target['bbox'] = merged_box
                    # Keep higher confidence
                    merge_target['confidence'] = max(det['confidence'], merge_target['confidence'])
                else:
                    kept.append(det)
            
            merged.extend(kept)
            
        return merged
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
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
    
    def _merge_boxes(self, box1: List[float], box2: List[float]) -> List[float]:
        """Merge two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        return [
            min(x1_min, x2_min),
            min(y1_min, y2_min),
            max(x1_max, x2_max),
            max(y1_max, y2_max)
        ]
    
    async def process_frame(self, 
                           frame_path: str, 
                           frame_idx: int) -> Dict[str, Any]:
        """Process a single frame with game-aware detection and segmentation."""
        print(f"\n📸 Processing frame {frame_idx}: {Path(frame_path).name}")
        
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            return {"frame": frame_path, "objects": []}
        
        # Run YOLO detection
        print("🔍 Running YOLOv11 detection...")
        results = self.yolo(frame, conf=0.2, iou=0.45, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = results[0].names[cls_id]
                
                detections.append({
                    'bbox': box,
                    'confidence': conf,
                    'class_name': cls_name,
                    'class_id': cls_id
                })
        
        # Filter and categorize detections
        filtered_detections = self.filter_detections(detections)
        print(f"✅ Filtered {len(filtered_detections)} game-relevant objects from {len(detections)} total")
        
        # Merge overlapping boxes
        merged_detections = self.merge_overlapping_boxes(filtered_detections)
        
        # Run SAM segmentation
        if merged_detections:
            # Prepare SAM prompts
            image_file = sieve.File(path=frame_path)
            sam_prompts = []
            
            for i, det in enumerate(merged_detections):
                bbox = det['bbox']
                sam_prompts.append({
                    "object_id": i + 1,
                    "frame_index": 0,
                    "box": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                })
            
            try:
                # Run SAM
                print(f"🎯 Running SAM segmentation on {len(sam_prompts)} objects...")
                sam_job = self.sam.push(
                    file=image_file,
                    prompts=sam_prompts,
                    model_type="large"
                )
                sam_result = sam_job.result()
                
                # Extract cutouts
                objects_info = []
                if isinstance(sam_result, tuple) and len(sam_result) >= 2:
                    masks_dict = sam_result[1]
                    
                    for i, (mask_name, mask_file) in enumerate(masks_dict.items()):
                        if i < len(merged_detections):
                            det = merged_detections[i]
                            category = det.get('game_category', 'unknown')
                            
                            # Create category directory
                            cat_dir = self.analysis_dir / "segmented_assets" / category
                            cat_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save cutout
                            frame_name = Path(frame_path).stem
                            output_name = f"{frame_name}_{det['class_name']}_{i}.png"
                            output_path = cat_dir / output_name
                            
                            if self._extract_cutout(frame_path, mask_file, str(output_path)):
                                objects_info.append({
                                    'category': category,
                                    'class': det['class_name'],
                                    'confidence': det['confidence'],
                                    'bbox': det['bbox'],
                                    'cutout_path': str(output_path)
                                })
                                print(f"✅ Saved {category}/{det['class_name']} cutout")
                
                return {
                    "frame": frame_path,
                    "timestamp": frame_idx / 30.0,
                    "objects": objects_info
                }
                
            except Exception as e:
                print(f"⚠️ SAM segmentation failed: {e}")
                
        return {"frame": frame_path, "timestamp": frame_idx / 30.0, "objects": []}
    
    def _extract_cutout(self, 
                       image_path: str, 
                       mask_file: sieve.File, 
                       output_path: str) -> bool:
        """Extract cutout using mask."""
        try:
            # Load images
            original = Image.open(image_path).convert("RGBA")
            mask = Image.open(mask_file.path).convert("L")
            
            # Resize mask if needed
            if mask.size != original.size:
                mask = mask.resize(original.size, Image.Resampling.LANCZOS)
            
            # Apply mask
            arr = np.array(original)
            arr[:, :, 3] = np.array(mask)
            
            # Save
            Image.fromarray(arr, "RGBA").save(output_path, "PNG")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to extract cutout: {e}")
            return False
    
    async def analyze_video(self, max_frames: int = 50):
        """Analyze the video with game-aware object detection."""
        print(f"\n🎮 Starting game-aware analysis for: {self.game_name}")
        print(f"📹 Video: {self.video_path.name}")
        
        # Extract frames if needed
        if not self.frames_dir.exists() or not list(self.frames_dir.glob("*.png")):
            print("\n📸 Extracting frames...")
            extract_frames(self.session_path)
        
        # Get timeline frames
        frame_files = sorted(self.frames_dir.glob("*_time.png"))
        
        # Sample frames
        sample_interval = max(1, len(frame_files) // max_frames)
        sampled_frames = frame_files[::sample_interval][:max_frames]
        
        print(f"📊 Processing {len(sampled_frames)} frames")
        
        # Process frames concurrently
        tasks = []
        for idx, frame_path in enumerate(sampled_frames):
            task = self.process_frame(str(frame_path), idx)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Compile statistics
        category_counts = {}
        class_counts = {}
        total_objects = 0
        
        for result in results:
            for obj in result['objects']:
                total_objects += 1
                cat = obj['category']
                cls = obj['class']
                
                category_counts[cat] = category_counts.get(cat, 0) + 1
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Save results
        output = {
            "game": self.game_name,
            "session": str(self.session_path),
            "frames_processed": len(results),
            "total_objects": total_objects,
            "categories": category_counts,
            "classes": class_counts,
            "object_mappings": self.object_mappings,
            "frame_results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        output_file = self.analysis_dir / "game_aware_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Analysis complete!")
        print(f"🎮 Game: {self.game_name}")
        print(f"📊 Total objects: {total_objects}")
        print(f"📦 Category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"   - {cat}: {count}")
        print(f"💾 Results: {output_file}")
        print(f"🖼️ Assets: {self.analysis_dir / 'segmented_assets'}")
        
        return output


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Game-aware video analysis")
    parser.add_argument("path", nargs="?", help="Path to session or game directory")
    parser.add_argument("--game", help="Override game detection")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to process")
    
    args = parser.parse_args()
    
    if args.path:
        path = Path(args.path)
        analyzer = GameAwareAnalyzer(path, game_name=args.game)
        await analyzer.analyze_video(max_frames=args.max_frames)
    else:
        # Interactive mode
        game_name = input("🎮 Enter game name: ").strip()
        if not game_name:
            return
            
        game_dir = Path("data") / game_name
        if not game_dir.exists():
            print(f"❌ No data for: {game_name}")
            return
            
        sessions = list_sessions(str(game_dir))
        if not sessions:
            print(f"❌ No sessions found")
            return
            
        print(f"\n📁 Available sessions:")
        for i, session in enumerate(sessions):
            print(f"  {i+1}. {Path(session).name}")
            
        try:
            selection = int(input("\n📌 Select session: ")) - 1
            if 0 <= selection < len(sessions):
                analyzer = GameAwareAnalyzer(sessions[selection], game_name=game_name)
                await analyzer.analyze_video(max_frames=args.max_frames)
        except ValueError:
            print("❌ Invalid selection")


if __name__ == "__main__":
    asyncio.run(main()) 