"""
Video analysis pipeline using YOLOv11 for object detection and SAM 2 for segmentation.

This module processes game recordings from android_streamer to:
1. Detect game objects using YOLOv11
2. Generate high-quality segmentations using SAM 2 (via Sieve)
3. Extract and save game assets as transparent PNGs
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import sieve
from PIL import Image
from ultralytics import YOLO

# Import utilities from existing modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.processing.frame_cutter import process_directory as extract_frames
from src.processing.visualize import parse_touch_log
from src.util import list_sessions

# Initialize Sieve with API key
sieve.api_key = os.getenv("SIEVE_API_KEY")

# Constants
YOLO_MODEL = "yolo11m.pt"  # Corrected filename
YOLO_CONFIDENCE_THRESHOLD = 0.45  # Lower threshold to catch more objects
YOLO_IOU_THRESHOLD = 0.6  # Non-max suppression threshold
SAM_MODEL_TYPE = "large"  # Use large SAM model for best quality
FRAME_SAMPLE_INTERVAL = 30  # Sample every 30 frames (1 second at 30fps)
MAX_FRAMES_TO_PROCESS = 100  # Limit total frames to process


class YOLOVideoAnalyzer:
    """Analyzes game videos using YOLOv11 for detection and SAM 2 for segmentation."""
    
    def __init__(self, session_path: str | Path):
        """
        Initialize the analyzer with a session path.
        
        Args:
            session_path: Path to a session directory containing screen_recording.mp4
        """
        self.session_path = Path(session_path)
        self.video_path = self.session_path / "screen_recording.mp4"
        self.touch_log_path = self.session_path / "touch_events.log"
        self.frames_dir = self.session_path / "frames"
        self.analysis_dir = self.session_path / "analysis" / "yolo_segmentation"
        
        # Ensure required files exist
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Create output directories
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Initialize SAM via Sieve
        self.sam = sieve.function.get("sieve/sam2")
        
    def _load_yolo_model(self):
        """Load YOLOv11 model, downloading if necessary."""
        # The model is expected to be in the same directory as this script.
        # YOLO() can handle the download automatically if the file isn't found
        # but we provide a clear message.
        model_path = Path(__file__).parent / YOLO_MODEL
        
        if not model_path.exists():
            print(f"📥 {YOLO_MODEL} not found. YOLO will attempt to download it.")
            
        print(f"🔧 Loading YOLO model: {YOLO_MODEL}...")
        try:
            self.yolo = YOLO(YOLO_MODEL)
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("💡 Please ensure you have an internet connection for the first-time download.")
            raise
        
    def detect_objects_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame using YOLOv11.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            List of detections with format:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_name': str,
                'class_id': int
            }
        """
        # Run YOLO inference
        results = self.yolo(
            frame, 
            conf=YOLO_CONFIDENCE_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extract detection info
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]
                
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': conf,
                    'class_name': cls_name,
                    'class_id': cls_id
                })
                
        return detections
    
    async def segment_with_sam(self, 
                              image_path: str, 
                              detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use SAM 2 to segment objects based on YOLO detections.
        
        Args:
            image_path: Path to the image file
            detections: List of YOLO detections
            
        Returns:
            List of segmentation results with masks
        """
        if not detections:
            return []
            
        # Create Sieve File object
        image_file = sieve.File(path=image_path)
        
        # Prepare SAM prompts from YOLO detections
        sam_prompts = []
        for i, det in enumerate(detections):
            # Convert bbox to SAM format [x_min, y_min, x_max, y_max]
            bbox = det['bbox']
            sam_prompt = {
                "object_id": i + 1,
                "frame_index": 0,  # For images, always 0
                "box": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            }
            sam_prompts.append(sam_prompt)
        
        try:
            # Run SAM segmentation
            print(f"🎯 Running SAM segmentation on {len(sam_prompts)} objects...")
            sam_job = self.sam.push(
                file=image_file,
                prompts=sam_prompts,
                model_type=SAM_MODEL_TYPE
            )
            
            # Get results
            sam_result = sam_job.result()
            
            # Process results
            segmentations = []
            if isinstance(sam_result, tuple) and len(sam_result) >= 2:
                masks_dict = sam_result[1]  # Second element contains masks
                
                for i, (mask_name, mask_file) in enumerate(masks_dict.items()):
                    if i < len(detections):
                        segmentations.append({
                            'detection': detections[i],
                            'mask_file': mask_file,
                            'mask_name': mask_name
                        })
                        
            return segmentations
            
        except Exception as e:
            print(f"⚠️ SAM segmentation failed: {e}")
            return []
    
    def extract_cutout(self, 
                      original_image_path: str, 
                      mask_file: sieve.File,
                      output_path: str) -> bool:
        """
        Extract a cutout using the segmentation mask.
        
        Args:
            original_image_path: Path to original image
            mask_file: Sieve File object containing the mask
            output_path: Where to save the cutout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load original image
            original_img = Image.open(original_image_path).convert("RGBA")
            
            # Load mask
            mask_path = mask_file.path
            mask_img = Image.open(mask_path).convert("L")
            
            # Resize mask if needed
            if mask_img.size != original_img.size:
                mask_img = mask_img.resize(original_img.size, Image.Resampling.LANCZOS)
            
            # Apply mask to create cutout
            original_array = np.array(original_img)
            mask_array = np.array(mask_img)
            
            # Set alpha channel based on mask
            cutout_array = original_array.copy()
            cutout_array[:, :, 3] = mask_array
            
            # Save cutout
            cutout_img = Image.fromarray(cutout_array, "RGBA")
            cutout_img.save(output_path, "PNG")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to extract cutout: {e}")
            return False
    
    async def process_frame(self, 
                           frame_path: str, 
                           frame_idx: int) -> Dict[str, Any]:
        """
        Process a single frame: detect objects and segment them.
        
        Args:
            frame_path: Path to the frame image
            frame_idx: Frame index for naming
            
        Returns:
            Dictionary with processing results
        """
        print(f"\n📸 Processing frame {frame_idx}: {Path(frame_path).name}")
        
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"⚠️ Failed to load frame: {frame_path}")
            return {"frame": frame_path, "objects": []}
        
        # Detect objects with YOLO
        print("🔍 Running YOLOv11 object detection...")
        detections = self.detect_objects_in_frame(frame)
        print(f"✅ Detected {len(detections)} objects")
        
        # Group detections by class for better organization
        class_groups = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        # Segment objects with SAM
        if detections:
            segmentations = await self.segment_with_sam(frame_path, detections)
            
            # Extract cutouts
            frame_name = Path(frame_path).stem
            objects_info = []
            
            for seg in segmentations:
                det = seg['detection']
                class_name = det['class_name']
                
                # Create output directory for this class
                class_dir = self.analysis_dir / "segmented_assets" / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate output filename
                output_name = f"{frame_name}_{class_name}_{det['confidence']:.2f}.png"
                output_path = class_dir / output_name
                
                # Extract cutout
                if self.extract_cutout(frame_path, seg['mask_file'], str(output_path)):
                    print(f"✅ Saved {class_name} cutout to {output_path.name}")
                    objects_info.append({
                        'class': class_name,
                        'confidence': det['confidence'],
                        'bbox': det['bbox'],
                        'cutout_path': str(output_path)
                    })
                    
            return {
                "frame": frame_path,
                "timestamp": frame_idx / 30.0,  # Assuming 30fps
                "detections": len(detections),
                "objects": objects_info
            }
        else:
            return {
                "frame": frame_path,
                "timestamp": frame_idx / 30.0,
                "detections": 0,
                "objects": []
            }
    
    async def analyze_video(self):
        """Main method to analyze the entire video."""
        print(f"\n🎮 Starting video analysis for: {self.session_path}")
        print(f"📹 Video: {self.video_path.name}")
        
        # Extract frames if not already done
        if not self.frames_dir.exists() or not list(self.frames_dir.glob("*.png")):
            print("\n📸 Extracting frames from video...")
            extract_frames(self.session_path)
        
        # Get all frame files
        frame_files = sorted(self.frames_dir.glob("*_time.png"))
        print(f"\n📊 Found {len(frame_files)} timeline frames to process")
        
        # Sample frames based on interval
        sampled_frames = frame_files[::FRAME_SAMPLE_INTERVAL]
        if len(sampled_frames) > MAX_FRAMES_TO_PROCESS:
            sampled_frames = sampled_frames[:MAX_FRAMES_TO_PROCESS]
        
        print(f"📌 Processing {len(sampled_frames)} sampled frames")
        
        # Process frames
        results = []
        tasks = []
        
        for idx, frame_path in enumerate(sampled_frames):
            task = self.process_frame(str(frame_path), idx)
            tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Compile summary statistics
        total_objects = sum(len(r['objects']) for r in results)
        class_counts = {}
        
        for result in results:
            for obj in result['objects']:
                class_name = obj['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Save analysis results
        analysis_output = {
            "session": str(self.session_path),
            "video": str(self.video_path),
            "frames_processed": len(results),
            "total_objects_detected": total_objects,
            "object_classes": class_counts,
            "frame_results": results,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        output_file = self.analysis_dir / "yolo_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_output, f, indent=2)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Total objects detected: {total_objects}")
        print(f"📦 Object class distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {class_name}: {count}")
        print(f"💾 Results saved to: {output_file}")
        print(f"🖼️ Segmented assets saved to: {self.analysis_dir / 'segmented_assets'}")
        
        return analysis_output


async def main():
    """Main entry point for the video analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze game videos with YOLOv11 and SAM 2")
    parser.add_argument("path", nargs="?", help="Path to session directory or game directory")
    parser.add_argument("--all-sessions", action="store_true", 
                       help="Process all sessions in a game directory")
    
    args = parser.parse_args()
    
    if args.path:
        path = Path(args.path)
        
        if args.all_sessions and path.is_dir():
            # Process all sessions in the directory
            sessions = list_sessions(path)
            if not sessions:
                print(f"❌ No sessions found in {path}")
                return
                
            print(f"📁 Found {len(sessions)} sessions to process")
            
            for session in sessions:
                try:
                    analyzer = YOLOVideoAnalyzer(session)
                    await analyzer.analyze_video()
                except Exception as e:
                    print(f"❌ Error processing {session}: {e}")
                    continue
        else:
            # Process single session
            analyzer = YOLOVideoAnalyzer(path)
            await analyzer.analyze_video()
    else:
        # Interactive mode
        game_name = input("🎮 Enter the game name: ").strip()
        if not game_name:
            print("❌ Game name cannot be empty")
            return
            
        # List available sessions
        game_dir = Path("data") / game_name
        if not game_dir.exists():
            print(f"❌ No data found for game: {game_name}")
            return
            
        sessions = list_sessions(game_dir)
        if not sessions:
            print(f"❌ No sessions found for {game_name}")
            return
            
        print(f"\n📁 Available sessions for {game_name}:")
        for i, session in enumerate(sessions):
            session_name = Path(session).name
            print(f"  {i+1}. {session_name}")
            
        # Get user selection
        try:
            selection = int(input("\n📌 Select session number: ")) - 1
            if 0 <= selection < len(sessions):
                analyzer = YOLOVideoAnalyzer(sessions[selection])
                await analyzer.analyze_video()
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")


if __name__ == "__main__":
    asyncio.run(main()) 