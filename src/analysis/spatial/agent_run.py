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
from src.analysis.spatial.prompts import SYSTEM_PROMPT


# Base prompt that's always included
BASE_PROMPT = """Everything should be as high quality and detailed as possible. Don't attend to UI and mobile elements.
You have a chunk {chunk_id} of a video. It is {chunk_seconds} long.
The video is from timestamp: {start_timestamp} to {end_timestamp}
Think deeply and understand the video.
Always read the entire file before you go and edit or write it. Note that you are not supposed to overwrite it in any way; you are only supposed to add more information to it. Never ever overwrite anything; instead, append and add information to it. You know that you know that we have time stamps that are being tracked, and so as in when you look at the scene and it's being updated, always append information to it. Only edit it when you want to, or only overwrite it when you want to add more information to it. Information should not be deleted in any way.
There is no need to depict the player character.
Then, go in, read the Unity 3D JSON file at {unity_file} and rewrite OR edit it."""


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
        logging.StreamHandler(),  # Console output
    ]

    # Configure root logger
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)

    # Create logger
    logger = logging.getLogger("spatial_reasoning")
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger, log_file


# Initialize logging
logger, log_file_path = setup_logging()


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
    Interactive selection of chunks directory and instructions file.

    Returns:
        Tuple of (chunks_dir, instructions_file) or (None, None) if cancelled.
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

    if selection.lower() == "q" or not selection:
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

    # Check if upload results exist in the selected chunks directory
    upload_results_path = Path(selected_chunks) / "upload_results.json"
    if not upload_results_path.exists():
        logger.error(f"No upload results found in {selected_chunks}")
        logger.error("Please run the chunk uploader first to upload the video chunks.")
        return None, None

    # Show selected chunks info
    print(f"\n✓ Selected chunks directory: {selected_chunks}")
    print(f"✓ Upload results found: {upload_results_path}")

    # Ask for instructions file
    print("\n" + "=" * 60)
    print("Additional Instructions Configuration")
    print("=" * 60)
    print("\nProvide the path to a text file containing additional instructions.")
    print("This will be appended to the base prompt that includes:")
    print("  - Chunk information (ID, duration, timestamps)")
    print("  - Unity file path")
    print("  - Basic video analysis instructions")
    print("\nYour file should contain game-specific or task-specific details.")
    print("-" * 60)

    instructions_file = input(
        "\nEnter instructions file path (or press Enter to cancel): "
    ).strip()

    if not instructions_file:
        logger.info("No instructions file specified, cancelling...")
        return None, None

    # Validate instructions file exists
    if not os.path.exists(instructions_file):
        logger.error(f"Instructions file not found: {instructions_file}")
        return None, None

    return selected_chunks, instructions_file


# Retry configuration for API calls
retry_config = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=32),
    retry=retry_if_exception_type(Exception),  # Retry on all exceptions
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)


@retry_config
def retry_file_get(client: genai.Client, file_name: str):
    """Get file from GenAI with retry logic."""
    logger.debug(f"Attempting to get file: {file_name}")
    file = client.files.get(name=file_name)
    file.video_metadata = types.VideoMetadata(fps=24.0).model_dump()
    logger.info(f"File metadata: {file.video_metadata}")
    return file


@retry_config
def retry_generate_content(
    client: genai.Client,
    model: str,
    config: types.GenerateContentConfig,
    contents: List[Any],
):
    """Generate content with retry logic."""
    logger.debug(f"Attempting to generate content with model: {model}")
    return client.models.generate_content(model=model, config=config, contents=contents)


def log_response_content(response, chunk_id: int) -> Dict[str, int]:
    """
    Log all response content including function calls and text responses.

    Returns a dictionary with function call counts.
    """
    function_counts = {}

    if not response or not response.candidates:
        logger.info(f"[Chunk {chunk_id}] No response candidates")
        return function_counts

    # Try to access function calls
    candidate = response.candidates[0]
    logger.info(f"[Chunk {chunk_id}] Response content:")

    # Check if the response has function calls
    if hasattr(candidate, "function_calls") and candidate.function_calls:
        for fc in candidate.function_calls:
            if fc is not None:  # Add null check
                func_name = fc.name
                func_args = fc.args
                # Log the function call
                logger.info(
                    f"  → Function: {func_name}({', '.join(f'{k}={v}' for k, v in func_args.items())})"
                )
                # Count function calls
                function_counts[func_name] = function_counts.get(func_name, 0) + 1

    # Also check content parts for mixed responses
    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
        for i, part in enumerate(candidate.content.parts):
            # Check for function calls in parts
            if hasattr(part, "function_call") and part.function_call is not None:
                fc = part.function_call
                func_name = fc.name
                func_args = fc.args
                logger.info(
                    f"  → Function: {func_name}({', '.join(f'{k}={v}' for k, v in func_args.items())})"
                )
                function_counts[func_name] = function_counts.get(func_name, 0) + 1

            # Check for text content
            elif hasattr(part, "text") and part.text:
                text_preview = part.text[:100].replace("\n", " ")
                if len(part.text) > 100:
                    text_preview += "..."
                logger.info(f"  → Text: {text_preview}")

            # Log other content types
            else:
                logger.info(f"  → Other content type in part {i}: {type(part)}")

    # If no function calls were found, ensure we've logged something
    if not function_counts and hasattr(candidate, "content"):
        if hasattr(candidate.content, "text") and candidate.content.text:
            text_preview = candidate.content.text[:100].replace("\n", " ")
            if len(candidate.content.text) > 100:
                text_preview += "..."
            logger.info(f"  → Text response: {text_preview}")

    return function_counts


def main():
    """Main entry point for the spatial reasoning processor."""
    parser = argparse.ArgumentParser(
        description="Process uploaded video chunks through spatial reasoning agent"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        help="Directory containing video chunks and upload results to process",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode to select chunks directory",
    )
    parser.add_argument(
        "--instructions-file",
        type=str,
        help="Path to text file containing additional instructions/context for the analysis",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set")
        return 1

    # Determine chunks directory and instructions file
    chunks_dir = None
    instructions_file = None

    if args.chunks_dir:
        # Directory specified explicitly
        chunks_dir = args.chunks_dir
        instructions_file = args.instructions_file

        # Validate chunks directory
        if not Path(chunks_dir).exists():
            logger.error(f"Chunks directory not found: {chunks_dir}")
            return 1

        # Check for upload results
        upload_results_path = Path(chunks_dir) / "upload_results.json"
        if not upload_results_path.exists():
            logger.error(f"No upload results found in {chunks_dir}")
            logger.error(
                "Please run the chunk uploader first to upload the video chunks."
            )
            return 1

    elif args.interactive or not args.chunks_dir:
        # Interactive mode
        chunks_dir, instructions_file = select_chunks_interactive()
        if not chunks_dir:
            logger.info("Selection cancelled")
            return 0

    logger.info(f"Using chunks directory: {chunks_dir}")

    # Check if instructions file is specified
    if not instructions_file:
        logger.error(
            "No instructions file specified! Use --instructions-file or interactive mode."
        )
        return 1

    # Load additional instructions from file
    logger.info(f"Loading instructions from: {instructions_file}")
    try:
        with open(instructions_file, "r") as f:
            additional_instructions = f.read()
        logger.info(f"Loaded instructions ({len(additional_instructions)} characters)")
    except Exception as e:
        logger.error(f"Failed to load instructions file: {e}")
        return 1

    # Combine base prompt with additional instructions
    prompt_template = BASE_PROMPT + "\n\n" + additional_instructions
    logger.info("Combined base prompt with additional instructions")

    # Load upload results from chunks directory
    upload_results_file = os.path.join(chunks_dir, "upload_results.json")
    logger.info(f"Loading upload results from: {upload_results_file}")
    with open(upload_results_file, "r") as f:
        upload_data = json.load(f)
    logger.info(f"Loaded {len(upload_data['results'])} upload results")

    # Load chunk metadata
    metadata_file = os.path.join(chunks_dir, "metadata.json")
    logger.info(f"Loading chunk metadata from: {metadata_file}")
    with open(metadata_file, "r") as f:
        chunk_metadata = json.load(f)
    logger.info(
        f"Video duration: {chunk_metadata['total_duration']:.2f}s, {chunk_metadata['num_chunks']} chunks"
    )

    # Get successful results and sort by display_name
    sorted_results = []
    for path, result in upload_data["results"].items():
        if result["status"] == "success":
            sorted_results.append((path, result))

    # Sort results by display_name
    sorted_results = sorted(
        sorted_results,
        key=lambda x: int(x[1]["display_name"]),
    )

    logger.info(f"Found {len(sorted_results)} successfully uploaded chunks")

    if not sorted_results:
        logger.error("No successfully uploaded chunks found in upload results!")
        return 1

    # Initialize GenAI client
    logger.info("Initializing GenAI client")
    client = genai.Client()

    # Configure generation with file tools
    config = types.GenerateContentConfig(
        temperature=1.6,
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=65536,
        top_p=0.95,
        tools=[read_file, write_file, edit_file],
    )

    # Create unity output directory based on chunks path
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
        logger.info(
            f"Time range: {start_time:.2f}s - {end_time:.2f}s (duration: {chunk_duration:.2f}s)"
        )
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
                if hasattr(e, "__context__") and hasattr(e.__context__, "last_attempt"):
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
                    contents=[chunk_file, prompt],
                )
            except Exception as e:
                # Count retries from exception context if available
                if hasattr(e, "__context__") and hasattr(e.__context__, "last_attempt"):
                    retry_count = max(
                        retry_count, e.__context__.last_attempt.attempt_number - 1
                    )
                raise

            # Log response content and function calls
            chunk_function_calls = log_response_content(response, chunk_id)

            # Update total function call counts
            for func_name, count in chunk_function_calls.items():
                total_function_calls[func_name] = (
                    total_function_calls.get(func_name, 0) + count
                )

            # Check response
            if response and (response.text or chunk_function_calls):
                logger.info(f"✓ Chunk {chunk_id} processed successfully")

                # Log function call summary for this chunk
                if chunk_function_calls:
                    logger.info(
                        f"Summary - Function calls made: {chunk_function_calls}"
                    )
                else:
                    logger.info("Summary - No function calls made (text-only response)")

                # Track retry statistics
                if retry_count > 0:
                    retry_statistics[chunk_id] = retry_count
                    logger.info(
                        f"Chunk {chunk_id} required {retry_count} retry attempt(s)"
                    )

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
        for func_name, count in sorted(
            total_function_calls.items(), key=lambda x: x[1], reverse=True
        ):
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
        logger.info(
            "\nNo retries were needed - all API calls succeeded on first attempt!"
        )

    # Log summary
    logger.info(f"\nLog file saved to: {log_file_path}")
    logger.info(f"Unity output saved to: {unity_file_path}")
    logger.info("=" * 80)

    return 0 if not failed_chunks else 1


if __name__ == "__main__":
    sys.exit(main())
