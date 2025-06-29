"""
Video chunking tool for preprocessing gameplay recordings.
Creates 5-second video chunks with overlaps for spatial understanding pipeline.
"""

import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import logging

from src.util import list_sessions, get_latest_session

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting video duration: {e}")
        raise
    except ValueError:
        logger.error(f"Could not parse duration from ffprobe output")
        raise


def calculate_chunk_times(total_duration: float, chunk_duration: int, overlap_seconds: int) -> List[Tuple[float, float]]:
    """
    Calculate start and end times for each chunk with overlap.
    
    Args:
        total_duration: Total video duration in seconds
        chunk_duration: Duration of each chunk in seconds
        overlap_seconds: Overlap between consecutive chunks in seconds
        
    Returns:
        List of (start_time, end_time) tuples for each chunk
    """
    chunks = []
    stride = chunk_duration - overlap_seconds
    
    if stride <= 0:
        raise ValueError("Overlap must be less than chunk duration")
    
    start_time = 0.0
    while start_time < total_duration:
        end_time = min(start_time + chunk_duration, total_duration)
        chunks.append((start_time, end_time))
        
        # If this chunk ends at the video end, we're done
        if end_time >= total_duration:
            break
            
        start_time += stride
    
    return chunks


def create_video_chunks(
    video_path: Path,
    output_dir: Path,
    chunk_duration: int = 5,
    overlap_seconds: int = 2
) -> Dict[str, Any]:
    """
    Create video chunks using ffmpeg with sliding window overlap.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save chunks
        chunk_duration: Duration of each chunk in seconds (default: 5)
        overlap_seconds: Overlap between consecutive chunks in seconds (default: 2)
        
    Returns:
        Dictionary containing operation status and metadata
    """
    try:
        # Validate input video exists
        if not video_path.exists():
            return {
                "status": "error",
                "error": f"Video file not found: {video_path}"
            }
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Get video duration
        total_duration = get_video_duration(str(video_path))
        logger.info(f"Video duration: {total_duration:.2f} seconds")
        
        # Calculate chunk times
        chunk_times = calculate_chunk_times(total_duration, chunk_duration, overlap_seconds)
        logger.info(f"Creating {len(chunk_times)} chunks")
        
        # Create chunks using ffmpeg
        chunk_files = []
        for i, (start_time, end_time) in enumerate(chunk_times):
            output_file = output_dir / f"{i}.mp4"
            duration = end_time - start_time
            
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-ss", str(start_time),      # Start time
                "-t", str(duration),         # Duration
                "-c", "copy",                # Copy codec (fast)
                "-avoid_negative_ts", "make_zero",
                "-y",                        # Overwrite output
                str(output_file)
            ]
            
            logger.info(f"Creating chunk {i}: {start_time:.2f}s - {end_time:.2f}s")
            subprocess.run(cmd, capture_output=True, check=True)
            chunk_files.append(str(output_file))
        
        # Save metadata
        metadata = {
            "source_video": str(video_path),
            "total_duration": total_duration,
            "chunk_duration": chunk_duration,
            "overlap_seconds": overlap_seconds,
            "num_chunks": len(chunk_times),
            "chunk_times": chunk_times,
            "created_at": datetime.now().isoformat()
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully created {len(chunk_files)} video chunks")
        
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "chunks": chunk_files,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error creating video chunks: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def chunk_session(session_path: Path, chunk_duration: int = 5, overlap_seconds: int = 2) -> bool:
    """
    Create chunks for a single session.
    
    Args:
        session_path: Path to session directory
        chunk_duration: Duration of each chunk in seconds
        overlap_seconds: Overlap between chunks
        
    Returns:
        True if successful, False otherwise
    """
    # Look for the overlay video first, fallback to original
    video_file = session_path / "screen_recording_overlay.mp4"
    if not video_file.exists():
        video_file = session_path / "screen_recording.mp4"
        if not video_file.exists():
            logger.error(f"No video file found in {session_path}")
            return False
    
    # Create chunked directory
    output_dir = session_path / "chunked"
    
    # Skip if already chunked
    if output_dir.exists() and (output_dir / "metadata.json").exists():
        logger.info(f"Session already chunked: {session_path}")
        return True
    
    logger.info(f"Chunking video: {video_file}")
    result = create_video_chunks(video_file, output_dir, chunk_duration, overlap_seconds)
    
    return result["status"] == "success"


def main():
    parser = argparse.ArgumentParser(description="Create video chunks for spatial understanding pipeline")
    parser.add_argument("--game", type=str, help="Game name (e.g., 'subway surfers')")
    parser.add_argument("--session", type=str, help="Specific session timestamp to chunk")
    parser.add_argument("--all", action="store_true", help="Chunk all sessions for the game")
    parser.add_argument("--chunk-duration", type=int, default=5, help="Duration of each chunk in seconds (default: 5)")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between chunks in seconds (default: 2)")
    
    args = parser.parse_args()
    
    if not args.game:
        # Interactive mode
        game_name = input("Enter game name: ").strip()
    else:
        game_name = args.game
    
    base_dir = Path("data")
    game_dir = base_dir / game_name
    
    if not game_dir.exists():
        logger.error(f"Game directory not found: {game_dir}")
        return
    
    if args.all:
        # Chunk all sessions
        sessions = list_sessions(game_dir)
        logger.info(f"Found {len(sessions)} sessions to process")
        
        success_count = 0
        for session in sessions:
            if chunk_session(session, args.chunk_duration, args.overlap):
                success_count += 1
        
        logger.info(f"Successfully chunked {success_count}/{len(sessions)} sessions")
    
    elif args.session:
        # Chunk specific session
        session_path = game_dir / args.session
        if not session_path.exists():
            logger.error(f"Session not found: {session_path}")
            return
        
        if chunk_session(session_path, args.chunk_duration, args.overlap):
            logger.info("Chunking completed successfully")
        else:
            logger.error("Chunking failed")
    
    else:
        # Interactive: chunk latest session
        latest_session = get_latest_session(game_dir)
        if not latest_session:
            logger.error(f"No sessions found in {game_dir}")
            return
        
        logger.info(f"Chunking latest session: {latest_session}")
        if chunk_session(latest_session, args.chunk_duration, args.overlap):
            logger.info("Chunking completed successfully")
        else:
            logger.error("Chunking failed")


if __name__ == "__main__":
    main()