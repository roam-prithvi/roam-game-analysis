#!/usr/bin/env python3
"""
Convert an image to 3D using Meshy AI API
"""

import requests
import os
import time
import base64
from pathlib import Path
import json
from datetime import datetime

# Configuration
IMAGE_PATH = "data/brawl stars/25-06-25_at_02.38.41/analysis/universal_grounded_sam2/frame_detections/frame_0035/cutouts/rgba_cutout_000_player_0.31.png"
OUTPUT_DIR = "meshy_3d_outputs"

# Meshy API configuration
MESHY_API_BASE_URL = "https://api.meshy.ai"
MESHY_API_KEY = "msy_K4rFu228LjI9ol2uZ1ZcZPgwQzxt1vHH1axg"

def load_api_key():
    """Load API key from environment or .env file"""
    api_key = os.environ.get('MESHY_API_KEY')
    
    if not api_key:
        # Try to load from .env file
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('MESHY_API_KEY='):
                        api_key = line.strip().split('=', 1)[1]
                        break
    
    if not api_key:
        print("No API key found. Please:")
        print("1. Set MESHY_API_KEY environment variable, or")
        print("2. Create a .env file with MESHY_API_KEY=your_api_key, or")
        print("3. Enter your API key now (or press Enter to use test mode)")
        api_key = input("API Key: ").strip()
        
        if not api_key:
            api_key = "msy_dummy_api_key_for_test_mode_12345678"
            print("Using test mode API key. This will not generate real results.")
    
    return api_key


def image_to_base64_data_uri(image_path):
    """Convert image to base64 data URI"""
    with open(image_path, 'rb') as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine the image type
    if image_path.lower().endswith('.png'):
        mime_type = 'image/png'
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = 'image/jpeg'
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
    
    return f"data:{mime_type};base64,{base64_string}"


def create_image_to_3d_task(api_key, image_path):
    """Create an Image to 3D task"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Convert image to base64 data URI
    image_data_uri = image_to_base64_data_uri(image_path)
    
    # Prepare the request payload
    payload = {
        "image_url": image_data_uri,
        "ai_model": "meshy-5",  # Using the latest model
        "topology": "quad",  # Quad-dominant mesh for better quality
        "target_polycount": 50000,  # Higher poly count for detail
        "symmetry_mode": "auto",
        "should_remesh": True,
        "should_texture": True,
        "texture_prompt": "colorful cartoon game character with vibrant colors",
        "moderation": False
    }
    
    print(f"Creating Image to 3D task...")
    print(f"Image: {image_path}")
    print(f"Model: {payload['ai_model']}")
    print(f"Topology: {payload['topology']}")
    print(f"Target polycount: {payload['target_polycount']}")
    
    response = requests.post(
        f"{MESHY_API_BASE_URL}/openapi/v1/image-to-3d",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        print(f"Error creating task: {response.status_code}")
        print(response.text)
        return None
    
    task_id = response.json()["result"]
    print(f"Task created successfully! Task ID: {task_id}")
    return task_id


def get_task_status(api_key, task_id):
    """Get the status of a task"""
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.get(
        f"{MESHY_API_BASE_URL}/openapi/v1/image-to-3d/{task_id}",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error getting task status: {response.status_code}")
        return None
    
    return response.json()


def wait_for_task_completion(api_key, task_id, check_interval=5):
    """Wait for a task to complete, polling at regular intervals"""
    print(f"\nWaiting for task to complete...")
    start_time = time.time()
    last_progress = -1
    
    while True:
        task = get_task_status(api_key, task_id)
        if not task:
            print("Failed to get task status")
            return None
        
        status = task.get("status")
        progress = task.get("progress", 0)
        
        # Only print if progress changed
        if progress != last_progress:
            elapsed = time.time() - start_time
            print(f"Status: {status} | Progress: {progress}% | Elapsed: {elapsed:.1f}s")
            last_progress = progress
        
        if status == "SUCCEEDED":
            print(f"\nTask completed successfully in {time.time() - start_time:.1f} seconds!")
            return task
        elif status == "FAILED":
            error_msg = task.get("task_error", {}).get("message", "Unknown error")
            print(f"\nTask failed: {error_msg}")
            return None
        elif status == "CANCELED":
            print("\nTask was canceled")
            return None
        
        time.sleep(check_interval)


def download_3d_model(url, output_path):
    """Download a file from URL"""
    print(f"Downloading: {output_path}")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {output_path}")
        return True
    else:
        print(f"Failed to download: {response.status_code}")
        return False


def save_task_results(task, output_dir):
    """Save all the generated assets"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory for this specific task
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = os.path.join(output_dir, f"brawlstars_character_{timestamp}")
    os.makedirs(task_dir, exist_ok=True)
    
    # Save task metadata
    metadata_path = os.path.join(task_dir, "task_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(task, f, indent=2)
    print(f"\nSaved task metadata to: {metadata_path}")
    
    # Download 3D models
    model_urls = task.get("model_urls", {})
    downloaded_files = []
    
    for format_name, url in model_urls.items():
        if url:
            output_path = os.path.join(task_dir, f"model.{format_name}")
            if download_3d_model(url, output_path):
                downloaded_files.append(output_path)
    
    # Download thumbnail
    thumbnail_url = task.get("thumbnail_url")
    if thumbnail_url:
        thumbnail_path = os.path.join(task_dir, "thumbnail.png")
        download_3d_model(thumbnail_url, thumbnail_path)
        downloaded_files.append(thumbnail_path)
    
    # Download textures
    texture_urls = task.get("texture_urls", [])
    if texture_urls:
        textures_dir = os.path.join(task_dir, "textures")
        os.makedirs(textures_dir, exist_ok=True)
        
        for i, texture_set in enumerate(texture_urls):
            for texture_type, url in texture_set.items():
                if url:
                    texture_path = os.path.join(textures_dir, f"texture_{i}_{texture_type}.png")
                    download_3d_model(url, texture_path)
                    downloaded_files.append(texture_path)
    
    return task_dir, downloaded_files


def main():
    """Main function"""
    print("=== Meshy AI Image to 3D Converter ===\n")
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return
    
    # Get API key
    api_key = load_api_key()
    
    # Create the task
    task_id = create_image_to_3d_task(api_key, IMAGE_PATH)
    if not task_id:
        return
    
    # Wait for completion
    completed_task = wait_for_task_completion(api_key, task_id)
    if not completed_task:
        return
    
    # Save results
    output_dir, files = save_task_results(completed_task, OUTPUT_DIR)
    
    print(f"\n=== Conversion Complete! ===")
    print(f"Output directory: {output_dir}")
    print(f"Downloaded {len(files)} files:")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    
    # Print model URLs for reference
    print(f"\nModel URLs (valid for limited time):")
    for format_name, url in completed_task.get("model_urls", {}).items():
        if url:
            print(f"  {format_name}: {url[:80]}...")


if __name__ == "__main__":
    main() 