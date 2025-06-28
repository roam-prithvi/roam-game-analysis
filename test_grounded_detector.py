#!/usr/bin/env python3
"""
Test script for the improved Grounding DINO + SAM 2 detector.
Tests the script invocation and detection pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from analysis.video_behavior_analysis.grounded_sam_detector import create_grounded_sam_detector


async def test_grounded_detector():
    """Test the Grounding DINO + SAM 2 detector."""
    print("🔬 Testing Grounded SAM Detector")
    print("=" * 50)
    
    try:
        # Create detector
        print("🔧 Creating Grounded SAM detector...")
        detector = await create_grounded_sam_detector(
            confidence_threshold=0.3,
            enable_segmentation=True
        )
        
        print("✅ Detector created successfully!")
        print(f"📁 Repository path: {detector.grounded_sam_path}")
        print(f"🎯 Features: {detector.get_supported_features()}")
        
        # Test with a simple prompt on a test image (if available)
        test_image_path = detector.grounded_sam_path / "notebooks/images/truck.jpg"
        
        if test_image_path.exists():
            print(f"\n🖼️ Testing detection on: {test_image_path}")
            
            # Test detection with simple prompts
            test_prompts = ["car", "vehicle", "truck"]
            
            print(f"💬 Testing prompts: {test_prompts}")
            detections = await detector.detect(
                str(test_image_path),
                prompts=test_prompts
            )
            
            print(f"✅ Detection completed!")
            print(f"📊 Found {len(detections)} objects:")
            
            for i, detection in enumerate(detections):
                print(f"  [{i+1}] {detection.class_name} "
                      f"(confidence: {detection.confidence:.2f}, "
                      f"bbox: {[round(x, 1) for x in detection.bbox]}, "
                      f"has_mask: {detection.mask is not None})")
            
            # Test game-specific prompts
            print(f"\n🎮 Testing game-specific prompts...")
            game_detections = await detector.detect(
                str(test_image_path),
                game_name="subway_surfers"
            )
            
            print(f"📊 Game-specific detection found {len(game_detections)} objects:")
            for i, detection in enumerate(game_detections):
                print(f"  [{i+1}] {detection.class_name} "
                      f"(confidence: {detection.confidence:.2f})")
            
        else:
            print(f"⚠️ Test image not found at {test_image_path}")
            print("💡 You can test detection manually by providing an image path")
        
        # Cleanup
        detector.cleanup()
        print("\n🧹 Cleanup completed")
        
        print("\n🎉 All tests passed!")
        
    except FileNotFoundError as e:
        print(f"❌ Repository not found: {e}")
        print("\n💡 To fix this:")
        print("1. Clone: git clone https://github.com/IDEA-Research/Grounded-SAM-2.git")
        print("2. Setup: cd Grounded-SAM-2 && pip install -r requirements.txt")
        print("3. Download models: Follow the INSTALL.md instructions")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_grounded_detector()) 