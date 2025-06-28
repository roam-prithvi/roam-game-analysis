#!/usr/bin/env python3
"""
Test script to verify YOLO-based video analysis setup.

This script checks:
1. Required dependencies are installed
2. YOLO model can be loaded
3. Sieve API key is configured
4. Sample data is available
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'pillow',
        'ultralytics': 'ultralytics',
        'sieve': 'sievedata'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def check_yolo_model():
    """Check if YOLO model can be loaded."""
    print("\n🔍 Checking YOLOv11 model...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load the model (will download if needed)
        print("📥 Loading YOLOv11 model (may download on first use)...")
        model = YOLO('yolo11m.pt')
        print("✅ YOLOv11 model loaded successfully")
        
        # Test inference on a dummy image
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        print("✅ YOLO inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        return False


def check_sieve_api():
    """Check if Sieve API is configured."""
    print("\n🔍 Checking Sieve API configuration...")
    
    api_key = os.getenv("SIEVE_API_KEY")
    if not api_key:
        print("❌ SIEVE_API_KEY environment variable not set")
        print("   Get your API key from: https://www.sievedata.com/")
        print("   Then set it: export SIEVE_API_KEY='your-key-here'")
        return False
    
    print("✅ SIEVE_API_KEY is set")
    
    try:
        import sieve
        sieve.api_key = api_key
        
        # Try to get SAM function
        sam = sieve.function.get("sieve/sam2")
        print("✅ Successfully connected to Sieve SAM 2")
        return True
        
    except Exception as e:
        print(f"⚠️  Sieve connection test failed: {e}")
        print("   This might be due to network issues or invalid API key")
        return False


def check_sample_data():
    """Check if sample data is available."""
    print("\n🔍 Checking for sample data...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ No data directory found")
        print("   Record some gameplay using android_streamer first")
        return False
    
    # Find game directories
    game_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not game_dirs:
        print("❌ No game data found")
        print("   Record some gameplay using android_streamer first")
        return False
    
    print(f"✅ Found {len(game_dirs)} game(s) with data:")
    
    total_sessions = 0
    for game_dir in game_dirs[:5]:  # Show max 5 games
        sessions = list(game_dir.glob("*/screen_recording.mp4"))
        total_sessions += len(sessions)
        print(f"   - {game_dir.name}: {len(sessions)} sessions")
    
    if len(game_dirs) > 5:
        print(f"   ... and {len(game_dirs) - 5} more games")
    
    print(f"\n📊 Total: {total_sessions} recording sessions available")
    
    return total_sessions > 0


def main():
    """Run all checks."""
    print("🧪 YOLO Video Analysis Setup Test")
    print("=" * 40)
    
    all_good = True
    
    # Check dependencies
    if not check_dependencies():
        all_good = False
    
    # Check YOLO
    if not check_yolo_model():
        all_good = False
    
    # Check Sieve
    if not check_sieve_api():
        all_good = False
    
    # Check data
    if not check_sample_data():
        all_good = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_good:
        print("✅ All checks passed! You're ready to run YOLO analysis.")
        print("\nTry running:")
        print("  python -m src.analysis.video_behavior_analysis.run_yolo_analysis")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nKey steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set Sieve API key: export SIEVE_API_KEY='your-key'")
        print("3. Record gameplay data using android_streamer")


if __name__ == "__main__":
    main() 