"""
Parallel file upload manager for Google GenAI.
Handles concurrent uploads of video chunks with progress tracking.
"""

import os
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import time

from google import genai
from google.genai import types

# Only import tqdm if available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")


class ChunkUploader:
    """Manages parallel uploads of video chunks to Google GenAI."""
    
    def __init__(self, client: Optional[genai.Client] = None, max_workers: int = 5):
        """
        Initialize the chunk uploader.
        
        Args:
            client: Google GenAI client instance. If None, will create one.
            max_workers: Maximum number of parallel upload threads.
        """
        self.client = client or genai.Client()
        self.max_workers = max_workers
        self.upload_stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def find_chunks(self, base_dir: str = "data") -> List[Tuple[str, str]]:
        """
        Find all video chunks in the specified directory.
        
        Args:
            base_dir: Base directory to search for chunks.
            
        Returns:
            List of tuples (chunk_path, session_name)
        """
        chunks = []
        base_path = Path(base_dir)
        
        # Look for chunked directories
        for chunked_dir in base_path.rglob("chunked"):
            if chunked_dir.is_dir():
                # Get session name from parent directories
                session_parts = chunked_dir.relative_to(base_path).parts[:-1]
                session_name = "/".join(session_parts)
                
                # Find all video files
                for video_file in chunked_dir.glob("*.mp4"):
                    chunks.append((str(video_file), session_name))
        
        # Sort by filename to ensure consistent ordering
        chunks.sort(key=lambda x: (x[1], int(Path(x[0]).stem) if Path(x[0]).stem.isdigit() else x[0]))
        
        return chunks
    
    def list_chunks(self, base_dir: str = "data") -> Dict[str, Dict[str, Any]]:
        """
        List all available chunked video sessions with metadata.
        
        Args:
            base_dir: Base directory to search for chunks.
            
        Returns:
            Dictionary mapping session paths to metadata.
        """
        sessions = {}
        base_path = Path(base_dir)
        
        # Find all chunked directories
        for chunked_dir in base_path.rglob("chunked"):
            if chunked_dir.is_dir():
                # Check for metadata.json
                metadata_file = chunked_dir / "metadata.json"
                if metadata_file.exists():
                    # Get relative path for display
                    rel_path = chunked_dir.relative_to(base_path)
                    session_key = str(rel_path)
                    
                    # Load metadata
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        # Extract game and session info
                        parts = rel_path.parts
                        game_name = parts[0] if len(parts) > 0 else "unknown"
                        session_name = parts[1] if len(parts) > 1 else "unknown"
                        
                        # Count actual chunk files
                        chunk_files = list(chunked_dir.glob("*.mp4"))
                        
                        sessions[session_key] = {
                            "path": str(chunked_dir),
                            "game": game_name,
                            "session": session_name,
                            "duration": metadata.get("total_duration", 0),
                            "num_chunks": len(chunk_files),
                            "expected_chunks": metadata.get("num_chunks", 0),
                            "chunk_duration": metadata.get("chunk_duration", 0),
                            "overlap": metadata.get("overlap_seconds", 0),
                            "created_at": metadata.get("created_at", ""),
                            "metadata": metadata
                        }
                    except Exception as e:
                        print(f"Warning: Could not load metadata for {chunked_dir}: {e}")
        
        return sessions
    
    def select_sessions_interactive(self, base_dir: str = "data", game_filter: Optional[str] = None) -> List[str]:
        """
        Interactive UI to select which sessions to upload.
        
        Args:
            base_dir: Base directory to search for chunks.
            game_filter: Optional game name to filter sessions.
            
        Returns:
            List of selected session paths.
        """
        sessions = self.list_chunks(base_dir)
        
        if not sessions:
            print("No chunked video sessions found!")
            return []
        
        # Group sessions by game
        games_dict = {}
        for session_key, info in sessions.items():
            game = info["game"]
            if game_filter and game_filter.lower() not in game.lower():
                continue
            if game not in games_dict:
                games_dict[game] = []
            games_dict[game].append((session_key, info))
        
        if not games_dict:
            print(f"No sessions found{' for game: ' + game_filter if game_filter else ''}")
            return []
        
        # Create flat list with game headers
        display_items = []
        session_to_idx = {}  # Map session path to display index
        idx = 1
        
        print("\nAvailable chunked videos:")
        print("=" * 80)
        
        for game in sorted(games_dict.keys()):
            print(f"\n🎮 {game.upper()}")
            print("-" * 80)
            
            # Sort sessions by timestamp/name
            game_sessions = sorted(games_dict[game], key=lambda x: x[1]["session"])
            
            for session_key, info in game_sessions:
                display_items.append((session_key, info))
                session_to_idx[info["path"]] = idx
                
                # Show session info
                print(f"{idx:3d}. {info['session']}")
                print(f"      📁 Path: {info['path']}")
                print(f"      ⏱️  Duration: {info['duration']:.1f}s")
                print(f"      🎬 Chunks: {info['num_chunks']} ({info['chunk_duration']}s each, {info['overlap']}s overlap)")
                
                if info['created_at']:
                    # Parse and format the date nicely
                    try:
                        created_dt = datetime.fromisoformat(info['created_at'])
                        created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"      📅 Created: {created_str}")
                    except:
                        print(f"      📅 Created: {info['created_at']}")
                
                # Check if chunks match expected
                if info['num_chunks'] != info['expected_chunks']:
                    print(f"      ⚠️  Warning: Found {info['num_chunks']} chunks, expected {info['expected_chunks']}")
                
                print()
                idx += 1
        
        print("=" * 80)
        print("\nSelection Options:")
        print("  📌 Numbers: 1,3,5 or ranges: 1-3")
        print("  📌 Game shortcuts: 'game:subway' to select all Subway Surfers sessions")
        print("  📌 'all' to select everything")
        print("  📌 'list' to see the selection again")
        print("  📌 'q' or empty to cancel")
        print()
        
        selected_paths = []
        
        while True:
            selection = input("Your choice: ").strip().lower()
            
            if not selection or selection == 'q':
                return []
            
            if selection == 'list':
                # Redisplay the list
                for game in sorted(games_dict.keys()):
                    print(f"\n🎮 {game.upper()}")
                    for session_key, info in games_dict[game]:
                        idx = session_to_idx[info["path"]]
                        print(f"{idx:3d}. {info['session']} ({info['duration']:.1f}s, {info['num_chunks']} chunks)")
                continue
            
            if selection == 'all':
                selected_paths = [info["path"] for _, info in display_items]
                break
            
            # Handle game shortcuts
            if selection.startswith('game:'):
                game_name = selection[5:].strip()
                found_game = None
                for game in games_dict.keys():
                    if game_name.lower() in game.lower():
                        found_game = game
                        break
                
                if found_game:
                    for _, info in games_dict[found_game]:
                        selected_paths.append(info["path"])
                    print(f"Selected all sessions for {found_game}")
                    break
                else:
                    print(f"No game found matching '{game_name}'")
                    continue
            
            # Parse numeric selection
            selected_indices = set()
            parts = selection.split(',')
            
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle range
                    try:
                        start, end = part.split('-')
                        start_idx = int(start.strip())
                        end_idx = int(end.strip())
                        for idx in range(start_idx, end_idx + 1):
                            if 1 <= idx <= len(display_items):
                                selected_indices.add(idx)
                    except ValueError:
                        print(f"Warning: Invalid range '{part}', skipping")
                else:
                    # Handle single number
                    try:
                        idx = int(part)
                        if 1 <= idx <= len(display_items):
                            selected_indices.add(idx)
                        else:
                            print(f"Warning: Index {idx} out of range (1-{len(display_items)})")
                    except ValueError:
                        print(f"Warning: Invalid number '{part}', skipping")
            
            if selected_indices:
                # Convert indices to paths
                for idx in sorted(selected_indices):
                    _, info = display_items[idx - 1]
                    selected_paths.append(info["path"])
                break
            else:
                print("No valid selections made. Try again or press 'q' to quit.")
        
        # Display selected sessions
        if selected_paths:
            print(f"\n✅ Selected {len(selected_paths)} session(s):")
            for path in selected_paths:
                # Find the info for this path
                for _, info in display_items:
                    if info["path"] == path:
                        print(f"  - {info['game']} / {info['session']}")
                        break
        
        return selected_paths
    
    def upload_single_chunk(self, chunk_path: str, retry_count: int = 3) -> Dict:
        """
        Upload a single video chunk.
        
        Args:
            chunk_path: Path to the video chunk file.
            retry_count: Number of retries on failure.
            
        Returns:
            Dictionary with upload result.
        """
        chunk_name = Path(chunk_path).stem
        result = {
            "path": chunk_path,
            "display_name": chunk_name,
            "status": "pending",
            "file_object": None,
            "error": None,
            "attempts": 0
        }
        
        for attempt in range(retry_count):
            try:
                result["attempts"] = attempt + 1
                
                # Upload the file
                file_obj = self.client.files.upload(
                    file=chunk_path,
                    config=types.UploadFileConfig(display_name=chunk_name)
                )
                
                result["status"] = "success"
                result["file_object"] = file_obj
                result["file_name"] = file_obj.name  # Store the server-side name
                break
                
            except Exception as e:
                result["error"] = str(e)
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    result["status"] = "failed"
        
        return result
    
    def upload_chunks(self, 
                     chunks_dir: Optional[str] = None,
                     chunks_dirs: Optional[List[str]] = None,
                     show_progress: Optional[bool] = None,
                     save_results: bool = True) -> Dict[str, Dict]:
        """
        Upload video chunks in parallel.
        
        Args:
            chunks_dir: Single directory containing chunks.
            chunks_dirs: List of directories containing chunks (overrides chunks_dir).
            show_progress: Whether to show progress bar. Auto-detects if None.
            save_results: Whether to save upload results to JSON file.
            
        Returns:
            Dictionary mapping chunk paths to upload results.
        """
        # Auto-detect progress display
        if show_progress is None:
            # Show progress if running as main script and tqdm is available
            show_progress = (__name__ == "__main__" or "ipykernel" in sys.modules) and TQDM_AVAILABLE
        
        # Find chunks based on provided directories
        chunks = []
        
        if chunks_dirs:
            # Multiple directories specified
            for dir_path in chunks_dirs:
                chunk_path = Path(dir_path)
                if chunk_path.exists() and chunk_path.is_dir():
                    session_name = f"{chunk_path.parent.parent.name}/{chunk_path.parent.name}"
                    dir_chunks = [(str(f), session_name) for f in chunk_path.glob("*.mp4")]
                    chunks.extend(dir_chunks)
                else:
                    print(f"Warning: Directory not found: {dir_path}")
        elif chunks_dir:
            # Single directory specified
            chunk_path = Path(chunks_dir)
            if chunk_path.exists() and chunk_path.is_dir():
                session_name = f"{chunk_path.parent.parent.name}/{chunk_path.parent.name}"
                chunks = [(str(f), session_name) for f in chunk_path.glob("*.mp4")]
            else:
                print(f"Error: Directory not found: {chunks_dir}")
                return {}
        else:
            # No directories specified, find all
            chunks = self.find_chunks()
        
        if not chunks:
            print("No video chunks found!")
            return {}
        
        # Initialize stats
        self.upload_stats["total"] = len(chunks)
        self.upload_stats["start_time"] = datetime.now()
        
        print(f"Found {len(chunks)} chunks to upload")
        print(f"Using {self.max_workers} parallel workers")
        
        # Upload results
        results = {}
        
        # Create progress bar if needed
        pbar = None
        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=len(chunks), desc="Uploading chunks", unit="chunk")
        
        # Parallel upload
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.upload_single_chunk, chunk[0]): chunk[0]
                for chunk in chunks
            }
            
            # Process completed uploads
            for future in as_completed(future_to_chunk):
                chunk_path = future_to_chunk[future]
                
                try:
                    result = future.result()
                    results[chunk_path] = result
                    
                    if result["status"] == "success":
                        self.upload_stats["successful"] += 1
                    else:
                        self.upload_stats["failed"] += 1
                    
                    # Update progress
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            "success": self.upload_stats["successful"],
                            "failed": self.upload_stats["failed"]
                        })
                    
                except Exception as e:
                    results[chunk_path] = {
                        "path": chunk_path,
                        "status": "failed",
                        "error": f"Unexpected error: {str(e)}"
                    }
                    self.upload_stats["failed"] += 1
                    
                    if pbar:
                        pbar.update(1)
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Update stats
        self.upload_stats["end_time"] = datetime.now()
        duration = (self.upload_stats["end_time"] - self.upload_stats["start_time"]).total_seconds()
        
        # Print summary
        print(f"\nUpload complete!")
        print(f"Total: {self.upload_stats['total']}")
        print(f"Successful: {self.upload_stats['successful']}")
        print(f"Failed: {self.upload_stats['failed']}")
        print(f"Time taken: {duration:.2f} seconds")
        print(f"Average: {duration/self.upload_stats['total']:.2f} seconds per chunk")
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Dict]):
        """Save upload results to JSON file."""
        output_file = f"upload_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for path, result in results.items():
            serializable_result = result.copy()
            # Remove non-serializable file object
            if "file_object" in serializable_result:
                if serializable_result["file_object"]:
                    serializable_result["file_info"] = {
                        "name": serializable_result.get("file_name", ""),
                        "display_name": serializable_result["display_name"]
                    }
                del serializable_result["file_object"]
            serializable_results[path] = serializable_result
        
        # Add metadata
        output_data = {
            "upload_stats": self.upload_stats,
            "results": serializable_results
        }
        
        # Save to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
    
    def get_successful_uploads(self, results: Dict[str, Dict]) -> Dict[str, any]:
        """
        Extract successful upload file objects.
        
        Args:
            results: Upload results dictionary.
            
        Returns:
            Dictionary mapping display names to file objects.
        """
        successful = {}
        for path, result in results.items():
            if result["status"] == "success" and result["file_object"]:
                display_name = result["display_name"]
                successful[display_name] = result["file_object"]
        
        return successful


def main():
    """CLI interface for chunk uploader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload video chunks to Google GenAI in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive selection of sessions to upload
  python -m src.analysis.spatial.chunk_uploader --interactive
  
  # Upload all sessions for a specific game
  python -m src.analysis.spatial.chunk_uploader --interactive --game "subway surfers"
  
  # Upload a specific directory
  python -m src.analysis.spatial.chunk_uploader --dir "data/subway surfers/08-06-25_at_19.33.00/chunked"
  
  # Just list available sessions without uploading
  python -m src.analysis.spatial.chunk_uploader --list
"""
    )
    parser.add_argument("--dir", "-d", help="Directory containing chunks", default=None)
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode to select which sessions to upload")
    parser.add_argument("--game", "-g", help="Filter sessions by game name (use with --interactive)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available sessions without uploading")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        return 1
    
    # Create uploader
    uploader = ChunkUploader(max_workers=args.workers)
    
    # Handle list mode
    if args.list:
        sessions = uploader.list_chunks()
        if not sessions:
            print("No chunked video sessions found!")
            return 0
        
        # Group by game
        games_dict = {}
        for session_key, info in sessions.items():
            game = info["game"]
            if args.game and args.game.lower() not in game.lower():
                continue
            if game not in games_dict:
                games_dict[game] = []
            games_dict[game].append(info)
        
        # Display sessions
        total_sessions = 0
        total_chunks = 0
        total_duration = 0
        
        for game in sorted(games_dict.keys()):
            print(f"\n🎮 {game.upper()}")
            print("-" * 60)
            
            game_sessions = sorted(games_dict[game], key=lambda x: x["session"])
            for info in game_sessions:
                print(f"  📁 {info['session']}")
                print(f"     Path: {info['path']}")
                print(f"     Duration: {info['duration']:.1f}s, Chunks: {info['num_chunks']}")
                if info['created_at']:
                    print(f"     Created: {info['created_at']}")
                print()
                
                total_sessions += 1
                total_chunks += info['num_chunks']
                total_duration += info['duration']
        
        print("=" * 60)
        print(f"Total: {total_sessions} sessions, {total_chunks} chunks, {total_duration:.1f}s")
        return 0
    
    # Handle interactive mode
    if args.interactive:
        selected_dirs = uploader.select_sessions_interactive(game_filter=args.game)
        if not selected_dirs:
            print("No sessions selected.")
            return 0
        
        # Confirm before uploading
        print(f"\n🚀 Ready to upload {len(selected_dirs)} session(s)")
        confirm = input("Proceed with upload? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Upload cancelled.")
            return 0
        
        # Upload selected directories
        results = uploader.upload_chunks(
            chunks_dirs=selected_dirs,
            show_progress=not args.no_progress,
            save_results=not args.no_save
        )
    else:
        # Normal mode - single directory or all
        if args.dir:
            # Validate directory exists
            if not Path(args.dir).exists():
                print(f"Error: Directory not found: {args.dir}")
                return 1
            if not (Path(args.dir) / "metadata.json").exists():
                print(f"Warning: No metadata.json found in {args.dir}")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    return 0
        
        results = uploader.upload_chunks(
            chunks_dir=args.dir,
            show_progress=not args.no_progress,
            save_results=not args.no_save
        )
    
    return 0 if uploader.upload_stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())