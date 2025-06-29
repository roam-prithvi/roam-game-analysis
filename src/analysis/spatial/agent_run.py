"""
Process uploaded video chunks sequentially through the spatial reasoning agent.
Reads from upload results and processes each chunk in order by display_name.
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from src.analysis.spatial.file_tools import read_file, write_file, edit_file
from src.analysis.spatial.prompts import SYSTEM_PROMPT, SUBWAY_SURFERS, BRAWL_STARS


# Set up logging
def setup_logging():
    """Configure logging for spatial reasoning agent."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"spatial_reasoning_{timestamp}.log"
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up handlers
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Console output
    ]
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    # Create logger
    logger = logging.getLogger("spatial_reasoning")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger, log_file


# Initialize logging
logger, log_file_path = setup_logging()


def find_upload_results(directory: str = ".") -> List[str]:
    """
    Find all upload results JSON files in the specified directory.
    
    Args:
        directory: Directory to search for upload results files.
        
    Returns:
        List of upload results file paths.
    """
    results = []
    for file in Path(directory).glob("upload_results_*.json"):
        results.append(str(file))
    
    # Sort by modification time (newest first)
    results.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return results


def validate_chunks_and_results(chunks_dir: str, upload_results_file: str) -> Tuple[bool, str]:
    """
    Validate that the chunks directory matches the upload results.
    
    Args:
        chunks_dir: Path to the chunks directory.
        upload_results_file: Path to the upload results file.
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if chunks directory exists
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        return False, f"Chunks directory not found: {chunks_dir}"
    
    # Check for metadata.json
    metadata_file = chunks_path / "metadata.json"
    if not metadata_file.exists():
        return False, f"No metadata.json found in chunks directory: {chunks_dir}"
    
    # Load upload results
    try:
        with open(upload_results_file, "r") as f:
            upload_data = json.load(f)
    except Exception as e:
        return False, f"Failed to load upload results: {e}"
    
    # Get the first result path to check consistency
    if not upload_data.get("results"):
        return False, "No results found in upload file"
    
    # Check if any result paths match the chunks directory
    chunks_dir_norm = os.path.normpath(chunks_dir)
    for path in upload_data["results"].keys():
        path_dir = os.path.normpath(os.path.dirname(path))
        if path_dir == chunks_dir_norm:
            return True, ""
    
    return False, f"Upload results do not contain files from {chunks_dir}"


def get_game_name(chunks_dir: str) -> str:
    """
    Extract game name from chunks directory path.
    
    Args:
        chunks_dir: Path to chunks directory (e.g., "data/brawl stars/1/chunked")
        
    Returns:
        Game name in lowercase (e.g., "brawl stars")
    """
    parts = Path(chunks_dir).parts
    if len(parts) >= 2 and parts[0] == "data":
        return parts[1].lower()
    return "unknown"


def get_game_and_session(chunks_dir: str) -> Tuple[str, str]:
    """
    Extract game name and session from chunks directory path.
    
    Args:
        chunks_dir: Path to chunks directory (e.g., "data/brawl stars/1/chunked")
        
    Returns:
        Tuple of (game_name, session_name) with spaces replaced by underscores
    """
    parts = Path(chunks_dir).parts
    if len(parts) >= 3 and parts[0] == "data":
        game = parts[1].lower().replace(" ", "_")
        session = parts[2]
        return game, session
    return "unknown", "unknown"


def select_chunks_interactive() -> Tuple[Optional[str], Optional[str]]:
    """
    Interactive selection of chunks directory and upload results.
    
    Returns:
        Tuple of (chunks_dir, upload_results_file) or (None, None) if cancelled.
    """
    # Find all chunked directories
    chunked_dirs = []
    base_path = Path("data")
    
    for chunked_dir in base_path.rglob("chunked"):
        if chunked_dir.is_dir() and (chunked_dir / "metadata.json").exists():
            chunked_dirs.append(str(chunked_dir))
    
    if not chunked_dirs:
        logger.error("No chunked video directories found!")
        return None, None
    
    # Sort for consistent display
    chunked_dirs.sort()
    
    print("\nAvailable chunked video directories:")
    print("-" * 60)
    for i, chunk_dir in enumerate(chunked_dirs, 1):
        # Load metadata to show info
        try:
            with open(os.path.join(chunk_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            duration = metadata.get("total_duration", 0)
            num_chunks = metadata.get("num_chunks", 0)
            print(f"{i:3d}. {chunk_dir}")
            print(f"      Duration: {duration:.1f}s, Chunks: {num_chunks}")
        except:
            print(f"{i:3d}. {chunk_dir}")
        print()
    
    # Get user selection
    print("-" * 60)
    selection = input("\nSelect chunk directory (number) or 'q' to quit: ").strip()
    
    if selection.lower() == 'q' or not selection:
        return None, None
    
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(chunked_dirs):
            selected_chunks = chunked_dirs[idx]
        else:
            logger.error("Invalid selection")
            return None, None
    except ValueError:
        logger.error("Invalid selection")
        return None, None
    
    # Find upload results files
    upload_files = find_upload_results()
    
    if not upload_files:
        logger.error("No upload results files found!")
        return None, None
    
    # Try to find matching upload results automatically
    matching_upload = None
    for upload_file in upload_files:
        is_valid, _ = validate_chunks_and_results(selected_chunks, upload_file)
        if is_valid:
            matching_upload = upload_file
            break
    
    if matching_upload:
        print(f"\nFound matching upload results: {matching_upload}")
        use_it = input("Use this file? (y/n): ").strip().lower()
        if use_it == 'y':
            return selected_chunks, matching_upload
    
    # Show available upload results
    print("\nAvailable upload results files:")
    print("-" * 60)
    for i, upload_file in enumerate(upload_files, 1):
        # Show file info
        file_time = datetime.fromtimestamp(os.path.getmtime(upload_file))
        print(f"{i:3d}. {upload_file}")
        print(f"      Modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("-" * 60)
    selection = input("\nSelect upload results file (number) or 'q' to quit: ").strip()
    
    if selection.lower() == 'q' or not selection:
        return None, None
    
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(upload_files):
            selected_upload = upload_files[idx]
            return selected_chunks, selected_upload
        else:
            logger.error("Invalid selection")
            return None, None
    except ValueError:
        logger.error("Invalid selection")
        return None, None


# Retry configuration for API calls
retry_config = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=32),
    retry=retry_if_exception_type(Exception),  # Retry on all exceptions
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True
)


@retry_config
def retry_file_get(client, file_name: str):
    """Get file from GenAI with retry logic."""
    logger.debug(f"Attempting to get file: {file_name}")
    return client.files.get(name=file_name)


@retry_config
def retry_generate_content(client, model: str, config: types.GenerateContentConfig, contents: List[Any]):
    """Generate content with retry logic."""
    logger.debug(f"Attempting to generate content with model: {model}")
    return client.models.generate_content(model=model, config=config, contents=contents)


def log_function_calls(response, chunk_id: int) -> Dict[str, int]:
    """
    Extract and log function calls from a response.
    
    Returns a dictionary with function call counts.
    """
    function_counts = {}
    
    if not response or not response.candidates:
        return function_counts
    
    # Try to access function calls
    candidate = response.candidates[0]
    
    # Check if the response has function calls
    if hasattr(candidate, 'function_calls') and candidate.function_calls:
        logger.info(f"[Chunk {chunk_id}] Function calls detected:")
        for fc in candidate.function_calls:
            func_name = fc.name
            func_args = fc.args
            
            # Log the function call
            logger.info(f"  → {func_name}({', '.join(f'{k}={v}' for k, v in func_args.items())})")
            
            # Count function calls
            function_counts[func_name] = function_counts.get(func_name, 0) + 1
    
    # Also check content parts for function calls (in case of manual calling)
    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
        for part in candidate.content.parts:
            if hasattr(part, 'function_call'):
                fc = part.function_call
                func_name = fc.name
                func_args = fc.args
                
                logger.info(f"  → {func_name}({', '.join(f'{k}={v}' for k, v in func_args.items())})")
                function_counts[func_name] = function_counts.get(func_name, 0) + 1
    
    return function_counts


def main():
    """Main entry point for the spatial reasoning processor."""
    parser = argparse.ArgumentParser(
        description="Process uploaded video chunks through spatial reasoning agent"
    )
    parser.add_argument(
        "--chunks-dir", 
        type=str, 
        help="Directory containing video chunks to process"
    )
    parser.add_argument(
        "--upload-results", 
        type=str, 
        help="Upload results JSON file to use"
    )
    parser.add_argument(
        "--interactive", 
        "-i", 
        action="store_true",
        help="Interactive mode to select chunks and upload results"
    )
    parser.add_argument(
        "--game",
        type=str,
        help="Explicitly specify the game name (overrides auto-detection)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set")
        return 1
    
    # Determine chunks directory and upload results file
    chunks_dir = None
    upload_results_file = None
    
    if args.chunks_dir and args.upload_results:
        # Both specified explicitly
        chunks_dir = args.chunks_dir
        upload_results_file = args.upload_results
        
        # Validate
        is_valid, error_msg = validate_chunks_and_results(chunks_dir, upload_results_file)
        if not is_valid:
            logger.error(f"Validation failed: {error_msg}")
            return 1
            
    elif args.interactive or (not args.chunks_dir and not args.upload_results):
        # Interactive mode
        chunks_dir, upload_results_file = select_chunks_interactive()
        if not chunks_dir or not upload_results_file:
            logger.info("Selection cancelled")
            return 0
            
    else:
        # Partial specification not allowed
        logger.error("Please specify both --chunks-dir and --upload-results, or use --interactive")
        return 1
    
    logger.info(f"Using chunks directory: {chunks_dir}")
    logger.info(f"Using upload results: {upload_results_file}")
    
    # Determine game name
    if args.game:
        game_name = args.game.lower()
        logger.info(f"Game explicitly specified: {game_name}")
    else:
        game_name = get_game_name(chunks_dir)
        logger.info(f"Auto-detected game: {game_name}")
    
    # Game-specific prompt mapping
    GAME_PROMPTS = {
        "subway surfers": SUBWAY_SURFERS,
        "brawl stars": BRAWL_STARS,
    }
    
    # Select appropriate prompt template
    prompt_template = GAME_PROMPTS.get(game_name)
    if prompt_template:
        logger.info(f"Using {game_name} specific prompt template")
    else:
        logger.warning(f"No specific prompt for '{game_name}', using default Subway Surfers template")
        prompt_template = SUBWAY_SURFERS
    
    # Load upload results
    logger.info(f"Loading upload results from: {upload_results_file}")
    with open(upload_results_file, "r") as f:
        upload_data = json.load(f)
    logger.info(f"Loaded {len(upload_data['results'])} upload results")
    
    # Load chunk metadata
    metadata_file = os.path.join(chunks_dir, "metadata.json")
    logger.info(f"Loading chunk metadata from: {metadata_file}")
    with open(metadata_file, "r") as f:
        chunk_metadata = json.load(f)
    logger.info(f"Video duration: {chunk_metadata['total_duration']:.2f}s, {chunk_metadata['num_chunks']} chunks")
    
    # Filter results to only include chunks from the selected directory
    chunks_dir_norm = os.path.normpath(chunks_dir)
    filtered_results = []
    for path, result in upload_data["results"].items():
        if result["status"] == "success":
            path_dir = os.path.normpath(os.path.dirname(path))
            if path_dir == chunks_dir_norm:
                filtered_results.append((path, result))
    
    # Sort results by display_name
    sorted_results = sorted(
        filtered_results,
        key=lambda x: int(x[1]["display_name"]),
    )
    
    logger.info(f"Found {len(sorted_results)} successfully uploaded chunks from selected directory")
    
    if not sorted_results:
        logger.error("No matching chunks found in upload results!")
        return 1
    
    # Initialize GenAI client
    logger.info("Initializing GenAI client")
    client = genai.Client()
    
    # Configure generation with file tools
    config = types.GenerateContentConfig(
        temperature=1.5,
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=65536,
        top_p=0.95,
        tools=[read_file, write_file, edit_file],
    )
    
    # Create game-specific unity output directory
    game_clean, session_name = get_game_and_session(chunks_dir)
    output_dir = Path("unity") / game_clean / session_name
    output_dir.mkdir(parents=True, exist_ok=True)
    unity_file_path = output_dir / "output.json"
    
    logger.info(f"Created output directory: {output_dir}")
    logger.info(f"Unity file will be saved to: {unity_file_path}")
    
    # Process tracking
    successful_chunks = 0
    failed_chunks = []
    total_function_calls = {}  # Track all function calls across chunks
    retry_statistics = {}  # Track retry attempts per chunk
    
    logger.info("=" * 80)
    logger.info("Starting spatial reasoning processing")
    logger.info(f"Processing {len(sorted_results)} chunks sequentially")
    logger.info("=" * 80)
    
    # Process each chunk in order
    for i, (path, result) in enumerate(sorted_results):
        chunk_id = int(result["display_name"])
        
        # Get timestamps for this chunk
        start_time, end_time = chunk_metadata["chunk_times"][chunk_id]
        chunk_duration = end_time - start_time
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing chunk {chunk_id}/{len(sorted_results) - 1}")
        logger.info(f"Time range: {start_time:.2f}s - {end_time:.2f}s (duration: {chunk_duration:.2f}s)")
        logger.info(f"File: {result['file_name']}")
        
        try:
            # Track retry attempts for this chunk
            retry_count = 0
            
            # Get file from GenAI with retry
            logger.debug(f"Retrieving file from GenAI: {result['file_name']}")
            try:
                chunk_file = retry_file_get(client, result["file_name"])
            except Exception as e:
                # Count retries from exception context if available
                if hasattr(e, '__context__') and hasattr(e.__context__, 'last_attempt'):
                    retry_count = e.__context__.last_attempt.attempt_number - 1
                raise
            
            # Format prompt with actual timestamps and info
            prompt = prompt_template.format(
                chunk_id=chunk_id,
                chunk_seconds=chunk_duration,
                start_timestamp=start_time,
                end_timestamp=end_time,
                unity_file=str(unity_file_path),
            )
            
            # Log the prompt being sent
            logger.debug(f"Prompt preview (first 200 chars): {prompt[:200]}...")
            
            # Process with agent with retry
            logger.info("Sending to spatial reasoning agent...")
            try:
                response = retry_generate_content(
                    client,
                    model="gemini-2.5-pro", 
                    config=config, 
                    contents=[chunk_file, prompt]
                )
            except Exception as e:
                # Count retries from exception context if available
                if hasattr(e, '__context__') and hasattr(e.__context__, 'last_attempt'):
                    retry_count = max(retry_count, e.__context__.last_attempt.attempt_number - 1)
                raise
            
            # Log function calls made during processing
            chunk_function_calls = log_function_calls(response, chunk_id)
            
            # Update total function call counts
            for func_name, count in chunk_function_calls.items():
                total_function_calls[func_name] = total_function_calls.get(func_name, 0) + count
            
            # Check response
            if response and response.text:
                logger.info(f"✓ Chunk {chunk_id} processed successfully")
                logger.info(
                    f"Agent response preview: {response.text[:200]}..."
                    if len(response.text) > 200
                    else f"Agent response: {response.text}"
                )
                
                # Log function call summary for this chunk
                if chunk_function_calls:
                    logger.info(f"Function calls for chunk {chunk_id}: {chunk_function_calls}")
                else:
                    logger.info(f"No function calls detected for chunk {chunk_id}")
                
                # Track retry statistics
                if retry_count > 0:
                    retry_statistics[chunk_id] = retry_count
                    logger.info(f"Chunk {chunk_id} required {retry_count} retry attempt(s)")
                
                successful_chunks += 1
            else:
                logger.warning(f"✗ Chunk {chunk_id} - No response from agent")
                failed_chunks.append(chunk_id)
                
        except Exception as e:
            logger.error(f"✗ Error processing chunk {chunk_id}: {e}", exc_info=True)
            failed_chunks.append(chunk_id)
    
    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Processing Complete!")
    logger.info(f"Successful chunks: {successful_chunks}/{len(sorted_results)}")
    if failed_chunks:
        logger.warning(f"Failed chunks: {failed_chunks}")
    
    # Log function call statistics
    logger.info("\nFunction Call Statistics:")
    if total_function_calls:
        for func_name, count in sorted(total_function_calls.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {func_name}: {count} calls")
    else:
        logger.info("  No function calls were made")
    
    # Log retry statistics
    if retry_statistics:
        logger.info("\nRetry Statistics:")
        total_retries = sum(retry_statistics.values())
        logger.info(f"  Total retry attempts: {total_retries}")
        logger.info(f"  Chunks that required retries: {len(retry_statistics)}")
        for chunk_id, retries in sorted(retry_statistics.items()):
            logger.info(f"    Chunk {chunk_id}: {retries} retry attempt(s)")
    else:
        logger.info("\nNo retries were needed - all API calls succeeded on first attempt!")
    
    # Log summary
    logger.info(f"\nLog file saved to: {log_file_path}")
    logger.info(f"Unity output saved to: {unity_file_path}")
    logger.info("=" * 80)
    
    return 0 if not failed_chunks else 1


if __name__ == "__main__":
    sys.exit(main())