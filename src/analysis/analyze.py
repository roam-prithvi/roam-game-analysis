# ---------------------------------------------------------------------------
#  CLI / ENTRY-POINT
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import instructor
from google import generativeai as genai  # type: ignore
from PIL import Image as PILImage
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from src import GEMINI_API_KEY
from src.analysis.base_models import ActionAnalysis, SceneAnalysis
from src.analysis.prompts import get_analyze_action_prompt, get_find_assets_prompt
from src.util import list_sessions, sanitize_path_component
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

PROD: bool = True  # Set to True for production, False for development/testing
LLM_NAME: str = (
    "models/gemini-2.5-pro-preview-06-05"
    if PROD
    else "models/gemini-2.5-flash-preview-05-20"
)
MAX_CONCURRENT_REQUESTS: int = 100 if PROD else 500

# ---------------------------------------------------------------------------
# 🛠️  Type stubs / aliases
# ---------------------------------------------------------------------------

# `FrameCombinedAnalysis` isn't yet implemented in the codebase, but it's only
# referenced for type hints.  We alias it to `Any` to keep the module runnable
# without pulling in extra dependencies.
FrameCombinedAnalysis = Any  # type: ignore

# ---------------------------------------------------------------------------
#  Action Analysis Data Structures
# ---------------------------------------------------------------------------


class TouchInteraction:
    """Represents a single touch interaction with its associated frame files."""

    def __init__(self, touch_frame: Path, timestamp: float) -> None:
        self.touch_frame = touch_frame
        self.timestamp = timestamp
        self.pre_frames: List[Path] = []
        self.post_frames: List[Path] = []

    def add_pre_frame(self, frame: Path) -> None:
        """Add a pre-touch frame and keep the list sorted by timestamp."""
        self.pre_frames.append(frame)
        self.pre_frames.sort(key=lambda p: float(p.stem.split("_")[0]))

    def add_post_frame(self, frame: Path) -> None:
        """Add a post-touch frame and keep the list sorted by timestamp."""
        self.post_frames.append(frame)
        self.post_frames.sort(key=lambda p: float(p.stem.split("_")[0]))

    def get_all_frames(self) -> List[Path]:
        """Return all frames in chronological order: pre -> touch -> post."""
        return [*self.pre_frames, self.touch_frame, *self.post_frames]

    def __repr__(self) -> str:
        return f"TouchInteraction(timestamp={self.timestamp}, pre={len(self.pre_frames)}, post={len(self.post_frames)})"


class TouchActionContext:
    """Correlates a touch interaction with its corresponding scene analysis."""

    def __init__(
        self, interaction: TouchInteraction, scene_analysis: Dict[str, Any]
    ) -> None:
        self.interaction = interaction
        self.scene_analysis = scene_analysis

    def get_assets(self) -> List[SceneAnalysis.Asset]:
        """Extract assets from the scene analysis and convert to Pydantic objects."""
        assets_data = self.scene_analysis.get("assets", [])
        return [
            SceneAnalysis.Asset.model_validate(asset_dict) for asset_dict in assets_data
        ]

    def get_ui_elements(self) -> List[SceneAnalysis.UIElement]:
        """Extract UI elements from the scene analysis and convert to Pydantic objects."""
        ui_elements_data = self.scene_analysis.get("ui_elements", [])
        return [
            SceneAnalysis.UIElement.model_validate(ui_dict)
            for ui_dict in ui_elements_data
        ]

    def __repr__(self) -> str:
        return f"TouchActionContext(timestamp={self.interaction.timestamp})"


# ---------------------------------------------------------------------------
#  Action Analysis Helper Functions
# ---------------------------------------------------------------------------


def _load_all_touch_interactions(frames_dir: Path) -> List[TouchInteraction]:
    """Load all touch interactions from the frames directory."""

    # Constants from frame_cutter.py
    N_BEFORE = 3
    N_AFTER = 3
    PRE_INTERVAL = 0.3
    POST_INTERVAL = 0.3

    # Find all touch frames first - be specific about the pattern
    all_png_files = list(frames_dir.glob("*_touch.png"))
    touch_frames = [
        f
        for f in all_png_files
        if f.name.endswith("_touch.png")
        and not f.name.endswith("_pre_touch.png")
        and not f.name.endswith("_post_touch.png")
    ]
    touch_frames = sorted(touch_frames)

    if not touch_frames:
        return []

    interactions: List[TouchInteraction] = []

    # Create a lookup dict for all frames by timestamp
    all_frames_by_timestamp: Dict[float, Path] = {}
    for frame_file in frames_dir.glob("*.png"):
        try:
            timestamp = float(frame_file.stem.split("_")[0])
            all_frames_by_timestamp[timestamp] = frame_file
        except ValueError:
            continue  # Skip malformed filenames

    for touch_frame in touch_frames:
        # Extract timestamp from filename (format: "timestamp_touch.png")
        timestamp_str = touch_frame.stem.split("_")[0]
        try:
            touch_timestamp = float(timestamp_str)
        except ValueError:
            continue  # Skip malformed filenames

        interaction = TouchInteraction(touch_frame, touch_timestamp)

        # Find pre-touch frames (going backwards in time)
        for k in range(1, N_BEFORE + 1):
            pre_timestamp = touch_timestamp - k * PRE_INTERVAL
            # Look for frame with this exact timestamp (with small tolerance for floating point)
            for candidate_timestamp, candidate_frame in all_frames_by_timestamp.items():
                if abs(
                    candidate_timestamp - pre_timestamp
                ) < 0.001 and candidate_frame.name.endswith("_pre_touch.png"):
                    interaction.add_pre_frame(candidate_frame)
                    break

        # Find post-touch frames (going forwards in time)
        for k in range(1, N_AFTER + 1):
            post_timestamp = touch_timestamp + k * POST_INTERVAL
            # Look for frame with this exact timestamp (with small tolerance for floating point)
            for candidate_timestamp, candidate_frame in all_frames_by_timestamp.items():
                if abs(
                    candidate_timestamp - post_timestamp
                ) < 0.001 and candidate_frame.name.endswith("_post_touch.png"):
                    interaction.add_post_frame(candidate_frame)
                    break

        interactions.append(interaction)

    return sorted(interactions, key=lambda x: x.timestamp)


def _correlate_touch_with_scene_analysis(
    touch_interactions: List[TouchInteraction], scene_analyses: List[Dict[str, Any]]
) -> List[TouchActionContext]:
    """Correlate touch interactions with their corresponding scene analyses."""

    contexts: List[TouchActionContext] = []

    # Create a mapping of frame names to scene analyses for quick lookup
    scene_by_frame: Dict[str, Dict[str, Any]] = {}
    for analysis in scene_analyses:
        frame_name = analysis.get("frame", "")
        if frame_name:
            scene_by_frame[frame_name] = analysis

    for interaction in touch_interactions:
        # Find the closest _time.png analysis by looking for nearby timestamps
        touch_timestamp = interaction.timestamp
        best_match: Dict[str, Any] | None = None
        best_diff = float("inf")

        for analysis in scene_analyses:
            frame_name = analysis.get("frame", "")
            if frame_name.endswith("_time.png"):
                try:
                    # Extract timestamp from frame name
                    frame_timestamp = float(frame_name.split("_")[0])
                    diff = abs(touch_timestamp - frame_timestamp)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = analysis
                except ValueError:
                    continue

        if best_match:
            contexts.append(TouchActionContext(interaction, best_match))

    return contexts


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def _analyze_action(context: TouchActionContext) -> ActionAnalysis:
    """Call Gemini via Instructor to extract an `ActionAnalysis` for a touch context."""

    try:
        # Configure the GenAI SDK (idempotent)
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[arg-type]

        # Get the prompt with assets and UI elements from scene analysis
        assets = context.get_assets()
        ui_elements = context.get_ui_elements()
        prompt: str = get_analyze_action_prompt(assets, ui_elements)

        # Open all images for this interaction (pre -> touch -> post)
        all_frames = context.interaction.get_all_frames()
        image_blobs = [PILImage.open(p) for p in all_frames]

        # Build the prompt content: textual instruction followed by images
        prompt_parts: List[object] = [prompt, *image_blobs]

        # Patch the Gemini client for structured outputs
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=LLM_NAME),
            use_async=True,
        )

        # The patched client returns a fully-validated `ActionAnalysis` instance
        result = await client.chat.completions.create(  # type: ignore[return-value]
            messages=[{"role": "user", "content": prompt_parts}],
            response_model=ActionAnalysis,
        )

        return result

    except Exception as e:
        # Note: We don't use console here as it might interfere with progress bar
        # The error will be handled and displayed properly in the calling function
        raise


async def _analyze_session_actions(
    session_path: Path, scene_analyses: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Analyze all touch interactions in a session concurrently with progress tracking."""

    frames_dir = session_path / "frames"
    console = Console()

    try:
        # Debug: Count actual touch files
        touch_files = list(frames_dir.glob("*_touch.png"))
        console.print(
            f"🔍  Found {len(touch_files)} _touch.png files in frames directory"
        )

        # Load touch interactions
        interactions = _load_all_touch_interactions(frames_dir)
        console.print(f"📝  Loaded {len(interactions)} touch interactions")

        if not interactions:
            console.print("ℹ️  No touch interactions found in frames directory")
            return []

        # Debug: Show first few interactions
        if interactions:
            console.print("📋  Sample interactions:")
            for i, interaction in enumerate(interactions[:3]):
                console.print(
                    f"    • {i+1}: {interaction.timestamp:.3f}s - {len(interaction.pre_frames)} pre, {len(interaction.post_frames)} post"
                )

        # Correlate with scene analyses
        contexts = _correlate_touch_with_scene_analysis(interactions, scene_analyses)
        console.print(
            f"🔗  Correlated to {len(contexts)} touch interactions to analyze"
        )

        if not contexts:
            console.print(
                "⚠️  No touch interactions could be correlated with scene analyses"
            )
            return []

        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing actions..."),
            MofNCompleteColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:

            # Add main task
            main_task = progress.add_task("Action Analysis", total=len(contexts))

            # Create semaphore to limit concurrent requests (avoid overwhelming the API)
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

            async def analyze_action_with_progress(
                context_idx: int, context: TouchActionContext
            ) -> Tuple[int, Dict[str, Any] | Exception]:
                """Analyze a single action with semaphore limiting and progress updates."""
                async with semaphore:
                    try:
                        result = await _analyze_action(context)
                        # Add metadata to the result
                        analysis_with_metadata = result.model_dump(mode="json")
                        analysis_with_metadata["timestamp"] = (
                            context.interaction.timestamp
                        )
                        analysis_with_metadata["touch_frame"] = (
                            context.interaction.touch_frame.name
                        )
                        analysis_with_metadata["num_pre_frames"] = len(
                            context.interaction.pre_frames
                        )
                        analysis_with_metadata["num_post_frames"] = len(
                            context.interaction.post_frames
                        )
                        analysis_with_metadata["total_frames"] = len(
                            context.interaction.get_all_frames()
                        )
                        progress.update(main_task, advance=1)
                        return context_idx, analysis_with_metadata
                    except Exception as e:
                        progress.update(main_task, advance=1)
                        return context_idx, e

            # Create tasks for concurrent processing
            tasks = []
            for i, context in enumerate(contexts):
                task = analyze_action_with_progress(i, context)
                tasks.append(task)

            # Run all analysis tasks concurrently
            results = await asyncio.gather(*tasks)

            # Process results and handle exceptions
            successful_analyses: List[Dict[str, Any]] = []
            failed_analyses: List[Tuple[int, Exception]] = []

            for context_idx, result in results:
                if isinstance(result, Exception):
                    failed_analyses.append((context_idx, result))
                else:
                    successful_analyses.append(result)

        # Print summary
        console.print(f"✅  Action analysis complete:")
        console.print(f"    • {len(successful_analyses)} successful analyses")
        if failed_analyses:
            console.print(f"    • {len(failed_analyses)} failed analyses")
            console.print("❌  Failed actions:")
            for context_idx, exception in failed_analyses:
                context = contexts[context_idx]
                console.print(
                    f"    • Action {context_idx + 1}: {context.interaction.timestamp:.3f}s ({type(exception).__name__}: {exception})"
                )

        return successful_analyses

    except Exception as e:
        console.print(f"❌  Error during action analysis: {e}")
        raise


def _write_action_output(session_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    """Write action analysis results to JSON file."""
    console = Console()
    out_dir: Path = session_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file: Path = out_dir / "action_analysis.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    console.print(f"💾  Wrote {len(results):,} action analyses ➜ {out_file}")


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------


def _load_first_two_time_images(frames_dir: Path) -> List[Path]:
    """Return the first two *_time.png images in *frames_dir* (sorted)."""

    images: List[Path] = sorted(p for p in frames_dir.glob("*_time.png") if p.is_file())
    if len(images) < 2:
        raise RuntimeError(f"Expected at least two *_time.png images in {frames_dir}")
    return images[50:52]


def _load_all_time_image_pairs(frames_dir: Path) -> List[tuple[Path, Path]]:
    """Return all consecutive pairs of *_time.png images in *frames_dir* (sorted)."""

    images: List[Path] = sorted(p for p in frames_dir.glob("*_time.png") if p.is_file())
    if len(images) < 2:
        raise RuntimeError(f"Expected at least two *_time.png images in {frames_dir}")

    # Create pairs: (image[i], image[i+1]) for each consecutive pair
    pairs: List[tuple[Path, Path]] = []
    for i in range(len(images) - 1):
        pairs.append((images[i], images[i + 1]))

    return pairs


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def _find_assets(images: List[Path]) -> SceneAnalysis:
    """Call Gemini via Instructor to extract a `SceneAnalysis` for *images*."""

    try:
        # Configure the GenAI SDK (idempotent)
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[arg-type]

        # Open images using Pillow, which the Gemini SDK accepts natively.
        image_blobs = [PILImage.open(p) for p in images]

        prompt: str = get_find_assets_prompt()
        # Build the prompt content: textual instruction followed by images.
        prompt_parts: List[object] = [prompt, *image_blobs]

        # Patch the Gemini client for structured outputs.
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=LLM_NAME),
            use_async=True,
        )

        # The patched client returns a fully-validated `SceneAnalysis` instance.
        result = await client.chat.completions.create(  # type: ignore[return-value]
            messages=[{"role": "user", "content": prompt_parts}],
            response_model=SceneAnalysis,
        )

        return result

    except Exception as e:
        # Note: We don't use console here as it might interfere with progress bar
        # The error will be handled and displayed properly in the calling function
        raise


async def _analyze_session(session_path: Path) -> List[Dict[str, Any]]:
    """Analyze all frame pairs in a session concurrently with progress tracking."""

    frames_dir = session_path / "frames"
    console = Console()

    if not PROD:
        # Return mock data for development/testing
        console.print("🧪  Development mode: Using mock scene analysis data")

        # Create mock data that matches the SceneAnalysis structure
        mock_scene_analysis = {
            "assets": [
                {
                    "name": "Player Character",
                    "description": "A character in a white hoodie running on a skateboard",
                    "primary_color": "white",
                    "secondary_color": "red",
                    "style": "toon",
                    "shader": "unlit",
                    "texture": "flat-color",
                },
                {
                    "name": "Gold Coin",
                    "description": "A floating, rotating, shiny gold coin collectible",
                    "primary_color": "gold",
                    "secondary_color": "yellow",
                    "style": "toon",
                    "shader": "unlit",
                    "texture": "flat-color",
                },
            ],
            "ui_elements": [
                {
                    "name": "Score Counter",
                    "description": "Numeric display showing current score in top-right",
                    "primary_color": "white",
                    "secondary_color": "yellow",
                    "style": "toon",
                    "shader": "unlit",
                    "texture": "flat-color",
                    "font": "cartoon",
                },
                {
                    "name": "Pause Button",
                    "description": "Blue rectangular pause button in top-left corner",
                    "primary_color": "blue",
                    "secondary_color": "white",
                    "style": "toon",
                    "shader": "unlit",
                    "texture": "flat-color",
                    "font": None,
                },
            ],
            "background": {
                "description": "Subway tunnel with brown stone blocks and warm lighting"
            },
        }

        # Create multiple mock analyses for different time frames
        mock_analyses = []
        time_frames = sorted(frames_dir.glob("*_time.png"))[
            :3
        ]  # Just first 3 for testing

        for i, time_frame in enumerate(time_frames):
            mock_analysis = mock_scene_analysis.copy()
            mock_analysis["frame"] = time_frame.name
            mock_analyses.append(mock_analysis)

        console.print(f"📊  Generated {len(mock_analyses)} mock scene analyses")
        return mock_analyses

    try:
        pairs = _load_all_time_image_pairs(frames_dir)
        console.print(f"🔍  Found {len(pairs)} frame pairs to analyze")

        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing frames..."),
            MofNCompleteColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:

            # Add main task
            main_task = progress.add_task("Frame Analysis", total=len(pairs))

            # Create semaphore to limit concurrent requests (avoid overwhelming the API)
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

            async def analyze_pair_with_progress(
                pair_idx: int, prev_frame: Path, curr_frame: Path
            ) -> Tuple[int, Dict[str, Any] | Exception]:
                """Analyze a single pair with semaphore limiting and progress updates."""
                async with semaphore:
                    try:
                        result = await _find_assets([prev_frame, curr_frame])
                        # Add frame information to the result
                        analysis_with_frame = result.model_dump(mode="json")
                        analysis_with_frame["frame"] = (
                            curr_frame.name
                        )  # Use current frame as the reference
                        progress.update(main_task, advance=1)
                        return pair_idx, analysis_with_frame
                    except Exception as e:
                        progress.update(main_task, advance=1)
                        return pair_idx, e

            # Create tasks for concurrent processing
            tasks = []
            for i, (prev_frame, curr_frame) in enumerate(pairs):
                task = analyze_pair_with_progress(i, prev_frame, curr_frame)
                tasks.append(task)

            # Run all analysis tasks concurrently
            results = await asyncio.gather(*tasks)

            # Process results and handle exceptions
            successful_analyses: List[Dict[str, Any]] = []
            failed_analyses: List[Tuple[int, Exception]] = []

            for pair_idx, result in results:
                if isinstance(result, Exception):
                    failed_analyses.append((pair_idx, result))
                else:
                    successful_analyses.append(result)

        # Print summary
        console.print(f"✅  Analysis complete:")
        console.print(f"    • {len(successful_analyses)} successful analyses")
        if failed_analyses:
            console.print(f"    • {len(failed_analyses)} failed analyses")
            console.print("❌  Failed pairs:")
            for pair_idx, exception in failed_analyses:
                prev_name = pairs[pair_idx][0].name
                curr_name = pairs[pair_idx][1].name
                console.print(
                    f"    • Pair {pair_idx + 1}: {prev_name} → {curr_name} ({type(exception).__name__}: {exception})"
                )

        return successful_analyses

    except Exception as e:
        console.print(f"❌  Error during session analysis: {e}")
        raise


# ---------------------------------------------------------------
#  CLI FUNCTIONS
# ---------------------------------------------------------------


def _write_output(session_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    console = Console()
    out_dir: Path = session_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file: Path = out_dir / "frame_analysis.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    console.print(f"💾  Wrote {len(results):,} frame analyses ➜ {out_file}")


def _choose_game_and_session(base_data_dir: Path) -> Path:
    """Interactive helper that mirrors visualise.py's UX (game → session)."""

    from datetime import datetime

    # TODO
    game_name: str = "subway surfers"
    # game_name: str = input("🎮  Enter the game name: ").strip()
    # while not game_name:
    #     game_name = input("Please enter a non-empty game name: ").strip()

    game_dir: Path = base_data_dir / sanitize_path_component(game_name)
    if not game_dir.exists():
        raise FileNotFoundError(f"Game directory not found: {game_dir}")

    sessions = list_sessions(game_dir)
    if not sessions:
        raise FileNotFoundError(f"No recording sessions in {game_dir}")

    print("\n📂  Available recording sessions:")
    for idx, sess in enumerate(sessions, start=1):
        ts: str = datetime.fromtimestamp(sess.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"  {idx:>2}. {sess.name} ({ts})")

    choice_str: str = input(
        f"\nSelect a session [1-{len(sessions)}] (default 1): "
    ).strip()
    if not choice_str:
        idx = 1
    else:
        try:
            idx = int(choice_str)
            if not (1 <= idx <= len(sessions)):
                raise ValueError
        except ValueError:
            raise RuntimeError("Invalid selection – exiting.")

    return sessions[idx - 1]


async def _async_main(session_path: Path) -> None:
    console = Console()
    console.print(f"▶️  Starting analysis for: {session_path}\n")

    # Phase 1: Scene Analysis (existing functionality)
    console.print("📊  Phase 1: Scene Analysis")
    analyses = await _analyze_session(session_path)
    _write_output(session_path, analyses)

    # Phase 2: Action Analysis (new functionality)
    console.print("\n🎯  Phase 2: Action Analysis")
    action_analyses = await _analyze_session_actions(session_path, analyses)
    if action_analyses:
        _write_action_output(session_path, action_analyses)

    console.print("\n✅  All analysis complete!")


def main(argv: Sequence[str] | None = None) -> None:  # noqa: D401
    console = Console()
    parser = argparse.ArgumentParser(description="Run LLM-based frame analysis.")
    parser.add_argument(
        "session_dir",
        nargs="?",
        help="Path to a <game>/<session> directory (interactive if omitted)",
    )
    args = parser.parse_args(argv)

    if args.session_dir:
        session_path = Path(args.session_dir)
        if not session_path.exists():
            console.print(f"❌  Path does not exist: {session_path}", style="red")
            sys.exit(1)
    else:
        session_path = _choose_game_and_session(Path("data"))

    try:
        asyncio.run(_async_main(session_path))
    except KeyboardInterrupt:
        console.print("\n⏹️  Interrupted by user.")


if __name__ == "__main__":
    # Quick smoke-test: run *one* find_assets call and print the JSON.
    # This avoids the full multi-frame, multi-request workflow so we can
    # verify credentials + connectivity first.

    console = Console()
    parser = argparse.ArgumentParser(
        description="Analyze frame pairs or run smoke-test."
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        help="Path to a <game>/<session> directory (interactive if omitted)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test on first frame pair only",
    )
    cli_args = parser.parse_args()

    if cli_args.session_dir:
        session_path = Path(cli_args.session_dir)
    else:
        session_path = _choose_game_and_session(Path("data"))

    if cli_args.smoke_test:

        async def _smoke_test() -> None:
            """Run the Instructor-powered Gemini call on the first two *_time.png frames."""

            frames_dir = session_path / "frames"
            try:
                images = _load_first_two_time_images(frames_dir)
            except RuntimeError as exc:
                console.print(f"❌  {exc}", style="red")
                return

            # Instructor/Gemini call with retry logic
            console.print("🔄  Running smoke test analysis...")
            scene: SceneAnalysis = await _find_assets(images)

            console.print("\n🔎  SceneAnalysis JSON:\n")
            console.print_json(json.dumps(scene.model_dump(mode="json"), indent=2))

        asyncio.run(_smoke_test())
    else:
        # Run full analysis workflow
        try:
            asyncio.run(_async_main(session_path))
        except KeyboardInterrupt:
            console.print("\n⏹️  Interrupted by user.")
