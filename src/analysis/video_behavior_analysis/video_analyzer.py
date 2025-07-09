"""
Video-based gameplay analysis using Gemini 2.5 Pro's native video understanding.
Analyzes game recordings to extract detailed behavior patterns, mechanics, and relationships.
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import google.genai as genai
from pydantic import BaseModel, Field
import re

from src import GEMINI_API_KEY
from src.streaming.android_streamer import sanitize_path_component
from src.util import list_sessions

# Import our universal object extractor
from .universal_object_extractor import UniversalObjectExtractor


class TouchEvent(BaseModel):
    """Represents a touch interaction from the log."""

    timestamp: float = Field(description="Time when touch occurred (seconds)")
    action: str = Field(description="Type of touch action (down, move, up)")
    x: int = Field(description="X coordinate of touch")
    y: int = Field(description="Y coordinate of touch")
    pressure: float = Field(description="Touch pressure if available", default=1.0)


class GameplayPattern(BaseModel):
    """Describes a specific gameplay pattern observed."""

    pattern_name: str = Field(description="Name of the gameplay pattern")
    trigger_conditions: List[str] = Field(
        description="What conditions trigger this pattern"
    )
    player_actions: List[str] = Field(description="What actions the player takes")
    game_responses: List[str] = Field(
        description="How the game responds to player actions"
    )
    strategic_value: str = Field(
        description="Why this pattern is strategically important"
    )
    difficulty_factors: List[str] = Field(
        description="What makes this pattern challenging"
    )


class ObjectBehavior(BaseModel):
    """Describes how game objects behave and interact."""

    object_name: str = Field(description="Name/type of the game object")
    movement_patterns: List[str] = Field(description="How this object moves or behaves")
    interaction_with_player: str = Field(
        description="How this object interacts with the player"
    )
    strategic_importance: str = Field(
        description="Role of this object in gameplay strategy"
    )
    behavioral_rules: List[str] = Field(
        description="Rules governing this object's behavior"
    )
    timing_characteristics: str = Field(
        description="Timing aspects of this object's behavior"
    )


class GameMechanic(BaseModel):
    """Describes a core game mechanic."""

    mechanic_name: str = Field(description="Name of the game mechanic")
    description: str = Field(description="Detailed description of how it works")
    player_control: str = Field(description="How players interact with this mechanic")
    strategic_depth: str = Field(
        description="Strategic considerations around this mechanic"
    )
    success_indicators: List[str] = Field(description="What indicates successful use")
    failure_consequences: List[str] = Field(
        description="What happens when this mechanic fails"
    )


class PlayerBehaviorAnalysis(BaseModel):
    """Analysis of player decision-making and reactions."""

    decision_triggers: List[str] = Field(
        description="What causes the player to make decisions"
    )
    reaction_patterns: List[str] = Field(
        description="How player reacts to different situations"
    )
    skill_progression: str = Field(
        description="How player skill is demonstrated or improves"
    )
    error_patterns: List[str] = Field(description="Common mistakes or suboptimal plays")
    adaptation_strategies: List[str] = Field(
        description="How player adapts to challenges"
    )


class GameplayAnalysis(BaseModel):
    """Comprehensive analysis of gameplay session."""

    game_type_classification: str = Field(
        description="What type of game this appears to be"
    )
    core_gameplay_loop: str = Field(
        description="The fundamental repeating gameplay cycle"
    )

    # Detailed breakdowns
    gameplay_patterns: List[GameplayPattern] = Field(
        description="Specific gameplay patterns observed"
    )
    object_behaviors: List[ObjectBehavior] = Field(
        description="How different objects behave"
    )
    game_mechanics: List[GameMechanic] = Field(description="Core mechanics identified")
    player_behavior: PlayerBehaviorAnalysis = Field(
        description="Analysis of player decision-making"
    )

    # Relationships and dynamics
    object_relationships: List[str] = Field(
        description="How different objects interact with each other"
    )
    difficulty_progression: str = Field(
        description="How difficulty changes throughout the session"
    )
    pacing_analysis: str = Field(description="Analysis of game pacing and rhythm")

    # Strategic insights
    optimal_strategies: List[str] = Field(
        description="What appear to be optimal player strategies"
    )
    challenge_design: List[str] = Field(
        description="How challenges are designed and presented"
    )
    engagement_factors: List[str] = Field(
        description="What keeps the gameplay engaging"
    )

    # Meta-analysis for game design
    good_design_elements: List[str] = Field(
        description="Elements that make for good game design"
    )
    potential_improvements: List[str] = Field(
        description="How this game design could be improved"
    )
    behavior_tree_insights: List[str] = Field(
        description="Insights for creating behavior trees for AI"
    )


class VideoGameplayAnalyzer:
    """
    Enhanced gameplay analyzer that extracts objects for universal detection.
    """

    def __init__(self, session_paths: List[Path]):
        """Initialize the analyzer for multiple session paths."""
        self.session_paths = (
            session_paths if isinstance(session_paths, list) else [session_paths]
        )
        self.sessions_data = []

        # Configure Gemini client
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Validate and prepare session data
        for session_path in self.session_paths:
            # Check for trimmed video first, fallback to original
            trimmed_video = session_path / "trimmed_screen_recording.mp4"
            if trimmed_video.exists():
                video_file = trimmed_video
            else:
                video_file = session_path / "screen_recording.mp4"
            touch_log_file = session_path / "touch_events.log"

            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_file}")
            if not touch_log_file.exists():
                raise FileNotFoundError(f"Touch events log not found: {touch_log_file}")

            self.sessions_data.append(
                {
                    "session_path": session_path,
                    "video_file": video_file,
                    "touch_log_file": touch_log_file,
                    "session_name": session_path.name,
                }
            )

        # Set up output directory (use first session for output location)
        first_session = self.session_paths[0]
        if len(self.session_paths) > 1:
            # Multi-session analysis - put results in parent game directory
            self.analysis_dir = first_session.parent / "analysis"
            self.video_analysis_dir = (
                self.analysis_dir / "video_behavior_analysis" / "multi_session"
            )
        else:
            # Single session analysis
            self.analysis_dir = first_session / "analysis"
            self.video_analysis_dir = self.analysis_dir / "video_behavior_analysis"

        # Ensure output directory exists
        self.video_analysis_dir.mkdir(parents=True, exist_ok=True)

        # Add universal object extraction capability
        self.universal_extractor = None
        self.extracted_objects: Set[str] = set()

        # Initialize video upload cache
        self.cache_file = self.video_analysis_dir / "gemini_upload_cache.json"
        self.upload_cache = self._load_upload_cache()

    def load_touch_events(self) -> Dict[str, List[TouchEvent]]:
        """Load and parse touch events from all session log files."""
        all_touch_events = {}

        for session_data in self.sessions_data:
            session_name = session_data["session_name"]
            touch_log_file = session_data["touch_log_file"]
            touch_events = []

            try:
                current_timestamp = None
                current_x = None
                current_y = None
                current_pressure = 1.0
                tracking_active = False

                with open(touch_log_file, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Parse Linux kernel event log format
                            # Format: [timestamp] EVENT_TYPE EVENT_CODE value
                            if line.startswith("[") and "]" in line:
                                # Extract timestamp from [timestamp] format
                                timestamp_end = line.find("]")
                                timestamp_str = line[1:timestamp_end].strip()
                                current_timestamp = float(timestamp_str)

                                # Parse the event info
                                event_part = line[timestamp_end + 1 :].strip()
                                parts = event_part.split()

                                if len(parts) >= 3:
                                    event_type = parts[0]
                                    event_code = parts[1]

                                    if event_type == "EV_ABS":
                                        if event_code == "ABS_MT_TRACKING_ID":
                                            # Touch tracking started/ended
                                            value = parts[2]
                                            if value == "ffffffff":
                                                # Touch ended
                                                if (
                                                    tracking_active
                                                    and current_x is not None
                                                    and current_y is not None
                                                ):
                                                    touch_events.append(
                                                        TouchEvent(
                                                            timestamp=current_timestamp,
                                                            action="up",
                                                            x=current_x,
                                                            y=current_y,
                                                            pressure=current_pressure,
                                                        )
                                                    )
                                                tracking_active = False
                                            else:
                                                # Touch started
                                                tracking_active = True

                                        elif (
                                            event_code == "ABS_MT_POSITION_X"
                                            and tracking_active
                                        ):
                                            current_x = int(parts[2])

                                        elif (
                                            event_code == "ABS_MT_POSITION_Y"
                                            and tracking_active
                                        ):
                                            current_y = int(parts[2])

                                        elif (
                                            event_code == "ABS_MT_TOUCH_MAJOR"
                                            and tracking_active
                                        ):
                                            current_pressure = (
                                                float(parts[2]) / 100.0
                                            )  # Normalize

                                    elif (
                                        event_type == "EV_KEY"
                                        and event_code == "BTN_TOUCH"
                                    ):
                                        if (
                                            parts[2] == "DOWN"
                                            and current_x is not None
                                            and current_y is not None
                                        ):
                                            touch_events.append(
                                                TouchEvent(
                                                    timestamp=current_timestamp,
                                                    action="down",
                                                    x=current_x,
                                                    y=current_y,
                                                    pressure=current_pressure,
                                                )
                                            )

                        except (ValueError, IndexError) as e:
                            # Skip unparseable lines (common in kernel logs)
                            continue

            except Exception as e:
                print(f"Error loading touch events from {session_name}: {e}")
                touch_events = []

            # Sort and store events for this session
            sorted_events = sorted(touch_events, key=lambda x: x.timestamp)
            all_touch_events[session_name] = sorted_events
            print(f"✅ Loaded {len(sorted_events)} touch events from {session_name}")

        return all_touch_events

    def create_touch_events_summary(
        self, all_touch_events: Dict[str, List[TouchEvent]]
    ) -> str:
        """Create a formatted summary of touch events from all sessions for the prompt."""
        if not all_touch_events:
            return "No touch events recorded."

        summary_lines = []
        total_events_across_sessions = sum(
            len(events) for events in all_touch_events.values()
        )
        summary_lines.append(
            f"=== MULTI-SESSION TOUCH EVENTS SUMMARY ({total_events_across_sessions} total events across {len(all_touch_events)} sessions) ==="
        )

        for session_name, touch_events in all_touch_events.items():
            if not touch_events:
                continue

            summary_lines.append(
                f"\n--- SESSION: {session_name} ({len(touch_events)} events) ---"
            )

            # Sample key events throughout this session
            total_events = len(touch_events)
            if total_events > 20:  # Reduced sample size for multiple sessions
                sample_indices = [int(i * total_events / 20) for i in range(20)]
                sampled_events = [touch_events[i] for i in sample_indices]
                summary_lines.append("(Showing sampled events due to high volume)")
            else:
                sampled_events = touch_events

            for event in sampled_events:
                time_str = f"{event.timestamp:.2f}s"
                location_str = f"({event.x}, {event.y})"
                summary_lines.append(f"  {time_str}: {event.action} at {location_str}")

            # Add session-specific statistics
            action_counts = {}
            for event in touch_events:
                action_counts[event.action] = action_counts.get(event.action, 0) + 1

            pattern_parts = []
            for action, count in sorted(action_counts.items()):
                pattern_parts.append(f"{action}: {count}")
            summary_lines.append(f"  Patterns: {', '.join(pattern_parts)}")

            # Add timing analysis for this session
            if len(touch_events) > 1:
                session_duration = (
                    touch_events[-1].timestamp - touch_events[0].timestamp
                )
                actions_per_second = (
                    len(touch_events) / session_duration if session_duration > 0 else 0
                )
                summary_lines.append(
                    f"  Duration: {session_duration:.1f}s, Rate: {actions_per_second:.1f} actions/sec"
                )

        # Add cross-session insights
        summary_lines.append(f"\n=== CROSS-SESSION PATTERNS ===")
        all_events_flat = []
        for events in all_touch_events.values():
            all_events_flat.extend(events)

        if all_events_flat:
            total_action_counts = {}
            for event in all_events_flat:
                total_action_counts[event.action] = (
                    total_action_counts.get(event.action, 0) + 1
                )

            for action, count in sorted(total_action_counts.items()):
                summary_lines.append(f"{action}: {count} times across all sessions")

        return "\n".join(summary_lines)

    def create_gameplay_analysis_prompt(
        self, all_touch_events: Dict[str, List[TouchEvent]]
    ) -> str:
        """Create a constrained prompt that forces visual-only analysis across multiple videos."""

        touch_summary = self.create_touch_events_summary(all_touch_events)
        num_videos = len(self.sessions_data)

        prompt = f"""You are analyzing {num_videos} gameplay video{"s" if num_videos > 1 else ""} from the same game. You must ONLY describe what you can actually see in the videos. Do NOT make assumptions or use external knowledge about games.

CRITICAL RULES:
1. ONLY describe visual elements you can actually observe in the video
2. Do NOT assume what game this is or use external knowledge
3. Do NOT make up specific names for objects unless clearly visible as text
4. Include specific timestamps when describing events
5. If you can't clearly see something, say "unclear" or "not visible"

📱 PLAYER INPUT DATA:
{touch_summary}

🔍 ANALYSIS TASKS:

**STEP 1: VISUAL INVENTORY**
- What type of game interface do you see? (grid, lanes, open world, etc.)
- What visual elements are present on screen? (buttons, meters, objects)
- What colors, shapes, and UI elements are visible?
- Are there any text labels or numbers you can read?

**STEP 2: OBJECT/ENTITY CATALOG**
- What distinct objects/entities do you see moving or static?
- Describe their visual appearance (color, shape, size, animations)
- Where are they positioned on screen?
- Do NOT assume names - describe what you see

**CRITICAL OBJECT GRANULARITY RULE:**
If the same visual entity appears with multiple distinct behavioral modes, treat EACH mode as a separate object type. For example, if you see the same type of vehicle that can be both moving and stationary, create separate entries for "Moving [Vehicle]" and "Static [Vehicle]" because they require completely different AI behavior trees.

**STEP 3: COMPLETE BEHAVIORAL ANALYSIS**
Focus on creating COMPLETE behavioral specifications for AI implementation:

For EACH distinct object/entity (including behavioral variants), describe:
- **Movement Behavior**: Where does it move? (toward what? in what direction? speed? constraints?)
- **Targeting Behavior**: What does it target? (nearest? specific type? random? criteria for selection?)
- **Attack/Action Behavior**: When and how does it act? (conditions, timing, effects, range)
- **Spatial Constraints**: Where can it go? (lanes, areas, boundaries, obstacles)
- **State Changes**: What causes it to change behavior? (health, proximity, time, triggers)
- **Interaction Rules**: How does it respond to other objects? (collision, overlap, proximity)

CRITICAL: Include ALL behavioral details, even "obvious" ones. Don't assume anything.

**STEP 4: OBJECT INTERACTION MATRIX**
For each pair of object types that interact, describe:
- **Collision Effects**: What happens when A touches B? (damage, destruction, state change)
- **Proximity Effects**: What happens when A gets near B? (detection range, behavioral change)
- **Targeting Priority**: If A can target multiple B's, which does it choose? (nearest, weakest, random)
- **Action Triggers**: What causes A to take action against B? (range, line of sight, timing)

**STEP 5: ENVIRONMENTAL RULES**
- **Spatial Layout**: How is the play area organized? (grid, lanes, zones, boundaries)
- **Movement Constraints**: What limits where objects can move? (walls, lanes, obstacles)
- **Spawning Behavior**: Where and when do new objects appear? (locations, timing, conditions)
- **Victory/Defeat Conditions**: What ends the game or changes state? (objectives, failure states)

**STEP 6: TIMING AND PACING**
- How fast do events occur?
- Are there periods of high/low activity?
- What creates urgency or pressure?
- How does difficulty appear to change over time?

**STEP 7: VISUAL STYLE ANALYSIS**
Analyze the visual style comprehensively for asset generation and art direction:

**Art Style Classification**:
- Is this pixel art, vector graphics, 3D rendered, hand-drawn, or hybrid?
- What is the resolution/fidelity level? (8-bit, 16-bit, HD, stylized, realistic)
- Are there visible outlines, cel-shading, or other distinctive rendering techniques?

**Color Palette**:
- What are the dominant colors in the game?
- How many colors are typically on screen at once?
- Is the palette vibrant, muted, pastel, neon, dark, or monochromatic?
- Are there color themes for different elements (e.g., enemies red, collectibles yellow)?

**Visual Effects & Particles**:
- Describe any particle effects (explosions, sparkles, smoke, dust)
- What happens visually when objects interact or are destroyed?
- Are there screen shakes, flashes, or other camera effects?
- How are impacts and collisions visualized?

**Animation Style**:
- How smooth are the animations? (frame count, fluidity)
- Are movements exaggerated/cartoony or realistic?
- Do objects squash and stretch, or maintain rigid forms?
- What easing/timing is used (snappy, smooth, bouncy)?

**UI/HUD Design**:
- What is the visual style of buttons, meters, and interface elements?
- Are UI elements flat, skeuomorphic, minimalist, or ornate?
- What fonts or text styles are visible?
- How is information hierarchy established visually?

**Lighting & Atmosphere**:
- Is there dynamic lighting, shadows, or purely flat lighting?
- What is the overall mood (bright, dark, cheerful, ominous)?
- Are there environmental effects (fog, rain, particles in air)?
- Time of day or lighting conditions?

**Texture & Surface Details**:
- Are textures detailed or simplified?
- What materials are suggested (metal, wood, fabric, organic)?
- Is there visible texture work or are surfaces flat-colored?
- Any repeating patterns or tileable elements?

**Camera & Perspective**:
- What is the camera angle (top-down, side-view, isometric, 3D)?
- Is the camera static or does it move/follow action?
- What is the field of view and scene framing?
- Any depth of field or focus effects?

**Overall Aesthetic Theme**:
- What genre/theme does the visual style suggest?
- Does it feel retro, modern, futuristic, fantasy, realistic?
- What emotional tone do the visuals convey?
- What similar games or art styles does this resemble?

**STEP 8: CROSS-SESSION ANALYSIS (if multiple videos)**
- What behaviors are consistent across all sessions?
- What varies between different sessions?
- Do you see learning/adaptation patterns between sessions?
- Are there different strategies employed in different videos?

**FORMAT YOUR RESPONSE:**

**SECTION 1: COMPLETE OBJECT BEHAVIORAL SPECIFICATIONS**

For EACH distinct object/entity type, provide:

**Object Name**: [Visual description since names may not be visible]
**Complete Behavioral Specification**:
- **Movement**: [Direction, speed, targeting, path constraints]
- **Targeting**: [What it targets, selection criteria, detection range]
- **Actions**: [What it does, when, conditions, effects, cooldowns]
- **Spatial Rules**: [Where it can go, boundaries, lane restrictions]
- **Interaction Rules**: [How it responds to each other object type]
- **State Changes**: [What causes behavior changes, health states, modes]
- **Timing**: [Action frequency, movement speed, reaction times]

**SECTION 2: ENVIRONMENTAL SYSTEMS**
- Game layout and spatial organization
- Spawning rules and patterns
- Win/loss conditions
- Resource systems

**SECTION 3: CROSS-SESSION PATTERNS** (if multiple videos)
- Behavioral consistency across sessions
- Variations in object behavior
- Pattern differences between sessions

**SECTION 4: VISUAL STYLE ANALYSIS**
Provide a comprehensive visual style breakdown covering:
- Art style classification and rendering techniques
- Complete color palette description
- Visual effects and particle systems
- Animation style and timing
- UI/HUD design language
- Lighting and atmospheric effects
- Texture and material details
- Camera system and perspective
- Overall aesthetic theme and mood

CRITICAL REQUIREMENTS:
- Focus on OBJECT behavior, not player behavior
- Include ALL behavioral details, even "obvious" ones
- Provide complete specifications suitable for AI behavior tree generation
- Base everything on visual observation only"""

        return prompt

    def _load_upload_cache(self) -> Dict[str, Any]:
        """Load the video upload cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                print(f"📋 Loaded upload cache with {len(cache)} entries")
                return cache
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}")
                return {}
        return {}

    def _save_upload_cache(self):
        """Save the video upload cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.upload_cache, f, indent=2)
            print(f"💾 Saved upload cache with {len(self.upload_cache)} entries")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for a file based on path, size, and modification time."""
        stat = file_path.stat()
        # Combine file path, size, and modification time for a unique identifier
        hash_string = f"{file_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _check_gemini_file_active(self, file_uri: str) -> bool:
        """Check if a Gemini file URI is still active and accessible."""
        try:
            # Extract file name from URI (format: https://generativelanguage.googleapis.com/v1beta/files/...)
            file_name = file_uri.split("/")[-1]
            file_info = self.client.files.get(name=f"files/{file_name}")
            return file_info.state == "ACTIVE"
        except Exception as e:
            print(f"⚠️ Failed to check file status: {e}")
            return False

    def upload_videos_to_gemini(self):
        """Upload all video files to Gemini and return list of file objects."""
        uploaded_files = []
        cache_updated = False

        for session_data in self.sessions_data:
            video_file = session_data["video_file"]
            session_name = session_data["session_name"]

            # Generate file hash for cache key
            file_hash = self._get_file_hash(video_file)
            cache_key = f"{video_file.absolute()}:{file_hash}"

            # Check if file is in cache and still active
            if cache_key in self.upload_cache:
                cached_entry = self.upload_cache[cache_key]
                cached_uri = cached_entry.get("uri")

                print(f"🔍 Found cached upload for: {session_name}")

                # Verify the cached file is still active
                if self._check_gemini_file_active(cached_uri):
                    print(
                        f"✅ Using cached video. Session: {session_name}, URI: {cached_uri}"
                    )
                    # Recreate file object from cached URI
                    uploaded_file = type(
                        "obj",
                        (object,),
                        {
                            "uri": cached_uri,
                            "name": f"files/{cached_uri.split('/')[-1]}",
                        },
                    )
                    uploaded_files.append(
                        {"file": uploaded_file, "session_name": session_name}
                    )
                    continue
                else:
                    print(f"⚠️ Cached file expired, re-uploading...")
                    del self.upload_cache[cache_key]
                    cache_updated = True

            # Upload video file (not in cache or cache expired)
            print(f"📤 Uploading video to Gemini: {session_name}")

            try:
                # Upload video file
                uploaded_file = self.client.files.upload(file=str(video_file))
                print(
                    f"✅ Video uploaded successfully. Session: {session_name}, URI: {uploaded_file.uri}"
                )

                # Add to uploaded files list
                uploaded_files.append(
                    {"file": uploaded_file, "session_name": session_name}
                )

                # Update cache
                self.upload_cache[cache_key] = {
                    "uri": uploaded_file.uri,
                    "session_name": session_name,
                    "uploaded_at": datetime.now().isoformat(),
                    "file_size": video_file.stat().st_size,
                    "file_mtime": video_file.stat().st_mtime,
                }
                cache_updated = True

            except Exception as e:
                print(f"❌ Error uploading video from {session_name}: {e}")
                raise

        # Save cache if updated
        if cache_updated:
            self._save_upload_cache()

        # Wait for all newly uploaded files to be processed
        newly_uploaded = [
            data for data in uploaded_files if hasattr(data["file"], "state")
        ]
        if newly_uploaded:
            print(
                f"⏳ Waiting for {len(newly_uploaded)} newly uploaded video(s) to be processed..."
            )
            max_wait = 300  # 5 minutes max
            wait_time = 0

            while wait_time < max_wait:
                all_active = True
                for uploaded_data in newly_uploaded:
                    uploaded_file = uploaded_data["file"]
                    session_name = uploaded_data["session_name"]

                    # Check file status
                    file_info = self.client.files.get(name=uploaded_file.name)
                    if file_info.state == "FAILED":
                        raise Exception(
                            f"Video processing failed for {session_name}: {file_info}"
                        )
                    elif file_info.state != "ACTIVE":
                        all_active = False

                if all_active:
                    print("✅ All videos processing complete!")
                    break

                # Wait and check again
                time.sleep(5)
                wait_time += 5
                if wait_time % 30 == 0:  # Update every 30 seconds
                    print(f"⏳ Still processing... ({wait_time}s elapsed)")

            if wait_time >= max_wait:
                raise Exception(f"Video processing timed out after {max_wait} seconds")

        return [data["file"] for data in uploaded_files]

    def run_analysis(self) -> str:
        """Main entry point to run the complete analysis."""
        try:
            # Load touch events from all sessions
            all_touch_events = self.load_touch_events()

            # Upload all videos
            uploaded_files = self.upload_videos_to_gemini()

            # Create analysis prompt
            prompt = self.create_gameplay_analysis_prompt(all_touch_events)

            print(
                f"🤖 Analyzing {len(uploaded_files)} gameplay video{'s' if len(uploaded_files) > 1 else ''} with Gemini 2.5 Flash..."
            )

            # Prepare content list with all videos and the prompt
            contents = uploaded_files + [prompt]

            # Perform analysis using correct API format
            response = self.client.models.generate_content(
                model="gemini-2.5-pro", contents=contents
            )

            raw_response = response.text
            print("✅ Multi-video analysis complete!")

            # Extract objects from Gemini analysis for universal detection
            extracted_objects = self.extract_objects_from_analysis(raw_response)

            # Save raw response as text
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if len(self.session_paths) > 1:
                analysis_type = f"multi_session_{len(self.session_paths)}videos"
            else:
                analysis_type = "single_session"
            text_file = (
                self.video_analysis_dir
                / f"detailed_analysis_{analysis_type}_{timestamp}.txt"
            )

            # Calculate total touch events
            total_touch_events = sum(
                len(events) for events in all_touch_events.values()
            )

            with open(text_file, "w") as f:
                f.write(f"=== MULTI-SESSION VIDEO GAMEPLAY ANALYSIS ===\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Sessions analyzed: {len(self.session_paths)}\n")
                for session_path in self.session_paths:
                    f.write(f"  - {session_path.name}\n")
                f.write(f"Total touch events: {total_touch_events}\n\n")
                f.write("=== EXTRACTED OBJECTS FOR DETECTION ===\n")
                if extracted_objects:
                    for obj in sorted(extracted_objects):
                        f.write(f"  - {obj}\n")
                    f.write(
                        f"\nUniversal Grounding DINO Prompt: {self.create_universal_grounded_sam_prompt()}\n"
                    )
                else:
                    f.write(
                        "No specific objects extracted, will use universal prompts.\n"
                    )
                f.write("\n=== TOUCH EVENTS SUMMARY ===\n")
                f.write(self.create_touch_events_summary(all_touch_events))
                f.write("\n\n=== DETAILED GAMEPLAY ANALYSIS ===\n")
                f.write(raw_response)

            print(f"💾 Detailed analysis saved to: {text_file}")

            # Save extracted objects for use by other components
            if extracted_objects:
                objects_file = (
                    self.video_analysis_dir / f"extracted_objects_{timestamp}.json"
                )
                objects_data = {
                    "extracted_objects": list(extracted_objects),
                    "universal_prompt": self.create_universal_grounded_sam_prompt(),
                    "session_paths": [str(p) for p in self.session_paths],
                    "timestamp": timestamp,
                }
                with open(objects_file, "w") as f:
                    json.dump(objects_data, f, indent=2)
                print(f"🎯 Extracted objects saved to: {objects_file}")

            print("\n📊 Analysis Summary:")
            print(f"✓ Sessions analyzed: {len(self.session_paths)}")
            print(f"✓ Total touch events processed: {total_touch_events}")
            print(f"✓ Videos analyzed with Gemini 2.5 Pro")
            print(f"✓ Objects extracted for detection: {len(extracted_objects)}")
            print(f"✓ Results saved to: {text_file}")

            # Extract and save visual style information
            visual_style_prompt = self.extract_and_generate_visual_style_prompt(
                raw_response
            )
            if visual_style_prompt:
                style_file = (
                    self.video_analysis_dir / f"visual_style_prompt_{timestamp}.txt"
                )
                with open(style_file, "w") as f:
                    f.write(visual_style_prompt)
                print(f"🎨 Visual style prompt saved to: {style_file}")

                # Also save as JSON for easier parsing
                style_json_file = (
                    self.video_analysis_dir / f"visual_style_data_{timestamp}.json"
                )
                style_data = {
                    "visual_style_prompt": visual_style_prompt,
                    "extracted_elements": self._parse_visual_style_content(
                        self._extract_style_section(raw_response)
                    ),
                    "session_paths": [str(p) for p in self.session_paths],
                    "timestamp": timestamp,
                }
                with open(style_json_file, "w") as f:
                    json.dump(style_data, f, indent=2)

            return raw_response

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

    def extract_and_generate_visual_style_prompt(self, analysis_text: str) -> str:
        """
        Extract visual style information from the analysis and generate a detailed prompt
        suitable for asset generation, texture creation, and animation design.

        Args:
            analysis_text: Raw analysis text from Gemini containing visual style section

        Returns:
            A detailed prompt describing the visual style of the game
        """
        # Look for the visual style analysis section
        style_section_pattern = r"\*\*SECTION 4: VISUAL STYLE ANALYSIS\*\*(.+?)(?=\*\*CRITICAL REQUIREMENTS|\Z)"
        style_match = re.search(
            style_section_pattern, analysis_text, re.DOTALL | re.IGNORECASE
        )

        if not style_match:
            # Try alternative patterns
            alt_patterns = [
                r"VISUAL STYLE ANALYSIS(.+?)(?=CRITICAL REQUIREMENTS|\Z)",
                r"Art [Ss]tyle [Cc]lassification(.+?)(?=CRITICAL REQUIREMENTS|\Z)",
            ]
            for pattern in alt_patterns:
                style_match = re.search(
                    pattern, analysis_text, re.DOTALL | re.IGNORECASE
                )
                if style_match:
                    break

        if not style_match:
            print("⚠️ Visual style section not found in analysis")
            return ""

        style_content = (
            style_match.group(1) if style_match.lastindex else style_match.group(0)
        )

        # Parse the visual style content
        visual_elements = self._parse_visual_style_content(style_content)

        # Generate the comprehensive style prompt
        style_prompt = self._generate_comprehensive_style_prompt(visual_elements)

        return style_prompt

    def _parse_visual_style_content(self, style_content: str) -> Dict[str, str]:
        """Parse the visual style content into structured elements."""
        elements = {}

        # Define the style categories we're looking for
        categories = [
            "art style classification",
            "color palette",
            "visual effects",
            "animation style",
            "ui/hud design",
            "lighting",
            "texture",
            "camera",
            "overall aesthetic",
        ]

        # Extract information for each category
        for category in categories:
            # Create pattern to find this category's content
            pattern = rf"{category}[:\s]*(.+?)(?=(?:" + "|".join(categories) + r")|\Z)"
            match = re.search(pattern, style_content, re.IGNORECASE | re.DOTALL)

            if match:
                content = match.group(1).strip()
                # Clean up the content
                content = re.sub(r"[-•]\s*", "", content)  # Remove bullet points
                content = re.sub(r"\n+", " ", content)  # Replace newlines with spaces
                content = re.sub(r"\s+", " ", content)  # Normalize spaces
                elements[category] = content.strip()

        return elements

    def _extract_style_section(self, analysis_text: str) -> str:
        """Extract just the visual style section from the full analysis."""
        style_section_pattern = r"\*\*SECTION 4: VISUAL STYLE ANALYSIS\*\*(.+?)(?=\*\*CRITICAL REQUIREMENTS|\Z)"
        style_match = re.search(
            style_section_pattern, analysis_text, re.DOTALL | re.IGNORECASE
        )

        if not style_match:
            # Try alternative patterns
            alt_patterns = [
                r"VISUAL STYLE ANALYSIS(.+?)(?=CRITICAL REQUIREMENTS|\Z)",
                r"Art [Ss]tyle [Cc]lassification(.+?)(?=CRITICAL REQUIREMENTS|\Z)",
            ]
            for pattern in alt_patterns:
                style_match = re.search(
                    pattern, analysis_text, re.DOTALL | re.IGNORECASE
                )
                if style_match:
                    break

        if style_match:
            return (
                style_match.group(1) if style_match.lastindex else style_match.group(0)
            )
        return ""

    def _generate_comprehensive_style_prompt(
        self, visual_elements: Dict[str, str]
    ) -> str:
        """Generate a comprehensive style prompt for asset generation."""
        prompt_lines = [
            "=== COMPREHENSIVE VISUAL STYLE PROMPT FOR ASSET GENERATION ===",
            "",
            "This game should be created with the following visual style specifications:",
            "",
        ]

        # Art Style Foundation
        if "art style classification" in visual_elements:
            prompt_lines.extend(
                [
                    "🎨 ART STYLE FOUNDATION:",
                    f"Create all assets in this style: {visual_elements['art style classification']}",
                    "This style should be consistently applied across all game elements including:",
                    "- Character designs and animations",
                    "- Environment and background assets",
                    "- UI elements and menus",
                    "- Visual effects and particles",
                    "",
                ]
            )

        # Color Palette Specification
        if "color palette" in visual_elements:
            prompt_lines.extend(
                [
                    "🎨 COLOR PALETTE SPECIFICATION:",
                    f"Use this color scheme: {visual_elements['color palette']}",
                    "Apply these colors to:",
                    "- Primary gameplay elements (player, enemies, collectibles)",
                    "- Environmental elements (backgrounds, platforms, obstacles)",
                    "- UI elements (buttons, meters, text)",
                    "- Visual feedback (damage, success, failure indicators)",
                    "",
                ]
            )

        # Animation Guidelines
        if "animation style" in visual_elements:
            prompt_lines.extend(
                [
                    "🎬 ANIMATION GUIDELINES:",
                    f"Animate with these characteristics: {visual_elements['animation style']}",
                    "Apply to:",
                    "- Character movement cycles (walk, run, jump, idle)",
                    "- Attack and action animations",
                    "- Environmental animations (water, wind, particles)",
                    "- UI transitions and feedback",
                    "",
                ]
            )

        # Visual Effects Specification
        if "visual effects" in visual_elements:
            prompt_lines.extend(
                [
                    "✨ VISUAL EFFECTS SPECIFICATION:",
                    f"Create effects with these properties: {visual_elements['visual effects']}",
                    "Include effects for:",
                    "- Collision impacts and destruction",
                    "- Power-ups and collectible acquisition",
                    "- Environmental interactions",
                    "- State changes and transformations",
                    "",
                ]
            )

        # UI/HUD Design Language
        if "ui/hud design" in visual_elements:
            prompt_lines.extend(
                [
                    "🖼️ UI/HUD DESIGN LANGUAGE:",
                    f"Design interface elements with: {visual_elements['ui/hud design']}",
                    "Apply to:",
                    "- Health bars, score displays, and meters",
                    "- Menu screens and navigation",
                    "- Button designs and interactive elements",
                    "- Text styling and information display",
                    "",
                ]
            )

        # Lighting and Atmosphere
        if "lighting" in visual_elements:
            prompt_lines.extend(
                [
                    "💡 LIGHTING AND ATMOSPHERE:",
                    f"Implement lighting as: {visual_elements['lighting']}",
                    "Consider:",
                    "- Overall scene illumination",
                    "- Shadow implementation (if applicable)",
                    "- Atmospheric effects and mood",
                    "- Special lighting for important elements",
                    "",
                ]
            )

        # Texture and Material Guidelines
        if "texture" in visual_elements:
            prompt_lines.extend(
                [
                    "🎨 TEXTURE AND MATERIAL GUIDELINES:",
                    f"Create textures with: {visual_elements['texture']}",
                    "Apply textures to:",
                    "- Character surfaces and clothing",
                    "- Environmental surfaces (ground, walls, objects)",
                    "- Special materials (metal, glass, organic)",
                    "- Background elements",
                    "",
                ]
            )

        # Camera and Perspective
        if "camera" in visual_elements:
            prompt_lines.extend(
                [
                    "📷 CAMERA AND PERSPECTIVE:",
                    f"Set up camera system as: {visual_elements['camera']}",
                    "This affects:",
                    "- Asset creation perspective",
                    "- Depth and layering of elements",
                    "- Movement and parallax effects",
                    "- Scene composition",
                    "",
                ]
            )

        # Overall Aesthetic Direction
        if "overall aesthetic" in visual_elements:
            prompt_lines.extend(
                [
                    "🎯 OVERALL AESTHETIC DIRECTION:",
                    f"The complete visual experience should feel: {visual_elements['overall aesthetic']}",
                    "This theme should permeate:",
                    "- All visual assets and designs",
                    "- Animation timing and feel",
                    "- Color choices and combinations",
                    "- Overall game atmosphere",
                    "",
                ]
            )

        # Technical Specifications
        prompt_lines.extend(
            [
                "📋 TECHNICAL SPECIFICATIONS FOR ASSET CREATION:",
                "- Maintain consistent style across all assets",
                "- Ensure visual hierarchy guides player attention",
                "- Optimize for mobile display if applicable",
                "- Create modular assets for efficient game development",
                "- Consider performance implications of visual choices",
                "",
                "🎮 ASSET CATEGORIES TO CREATE:",
                "1. Characters (player, enemies, NPCs)",
                "2. Environment (backgrounds, platforms, obstacles)",
                "3. UI Elements (buttons, meters, menus)",
                "4. Visual Effects (particles, explosions, transitions)",
                "5. Collectibles and Power-ups",
                "6. Decorative Elements",
                "",
                "Use this prompt when generating any visual assets for the game to ensure",
                "consistency and adherence to the observed visual style.",
            ]
        )

        return "\n".join(prompt_lines)

    def extract_objects_from_analysis(self, analysis_text: str) -> Set[str]:
        """
        Extract object names from Gemini's behavior analysis for use in universal detection.

        Args:
            analysis_text: Raw analysis text from Gemini

        Returns:
            Set of object names that can be used for detection
        """
        objects = set()

        # Pattern 1: Look for "Object Name:" or similar headers
        object_name_patterns = [
            r"\*\*Object Name\*\*:\s*([^\n\r]+)",
            r"Object Name:\s*([^\n\r]+)",
            r"Entity:\s*([^\n\r]+)",
            r"Character:\s*([^\n\r]+)",
        ]

        for pattern in object_name_patterns:
            matches = re.finditer(pattern, analysis_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                obj_name = match.group(1).strip()
                cleaned_objects = self._clean_and_split_objects(obj_name)
                objects.update(cleaned_objects)

        # Pattern 2: Look for behavioral descriptions that start with object names
        behavior_patterns = [
            r"([A-Z][a-zA-Z\s]+?)\s+(?:moves|attacks|shoots|collects|spawns|appears|behaves)",
            r"([A-Z][a-zA-Z\s]+?)\s+(?:is|are)\s+(?:moving|attacking|collecting|spawning)",
            r"The\s+([a-zA-Z][a-zA-Z\s]+?)\s+(?:moves|attacks|shoots|collects)",
        ]

        for pattern in behavior_patterns:
            matches = re.finditer(pattern, analysis_text, re.MULTILINE)
            for match in matches:
                obj_name = match.group(1).strip()
                if len(obj_name) > 2 and len(obj_name) < 50:
                    objects.add(obj_name)

        # Pattern 3: Look for quoted object names (likely specific game entities)
        quoted_pattern = r'"([^"]{3,30})"'
        quoted_matches = re.finditer(quoted_pattern, analysis_text)
        for match in quoted_matches:
            obj_name = match.group(1).strip()
            if self._is_likely_object_name(obj_name):
                objects.add(obj_name)

        # Pattern 4: Look for section headers that describe objects
        section_patterns = [
            r"\*\*([^*\n\r]{5,40})\*\*\s*\n\s*(?:Complete Behavioral Specification|Movement|Targeting)",
            r"(?:SECTION|OBJECT|ENTITY)\s+\d+:\s*([^\n\r]+)",
        ]

        for pattern in section_patterns:
            matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
            for match in matches:
                obj_text = match.group(1).strip()
                cleaned_objects = self._clean_and_split_objects(obj_text)
                objects.update(cleaned_objects)

        # Filter and clean the results
        filtered_objects = set()
        for obj in objects:
            cleaned = self._clean_object_name(obj)
            if cleaned and self._is_valid_object_name(cleaned):
                filtered_objects.add(cleaned)

        self.extracted_objects.update(filtered_objects)
        print(f"🎯 Extracted {len(filtered_objects)} object names from Gemini analysis")

        return filtered_objects

    def _clean_and_split_objects(self, text: str) -> List[str]:
        """Clean and split object names from text."""
        # Remove common prefixes and suffixes
        text = re.sub(
            r"(?:Visual description since names may not be visible)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?:Complete Behavioral Specification)", "", text, flags=re.IGNORECASE
        )

        # Split on common separators
        objects = []
        separators = [" and ", " or ", ",", ";", "/", "|", "\n"]

        parts = [text]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts

        for part in parts:
            cleaned = part.strip(" \t\n\r\"'()[]{}")
            if len(cleaned) > 2:
                objects.append(cleaned)

        return objects

    def _clean_object_name(self, name: str) -> str:
        """Clean a single object name."""
        # Remove brackets and quotes
        name = re.sub(r"[\[\](){}]", "", name)
        name = name.strip(" \t\n\r\"'")

        # Remove common descriptive prefixes
        prefixes_to_remove = [
            "moving ",
            "static ",
            "animated ",
            "visible ",
            "large ",
            "small ",
            "the ",
            "a ",
            "an ",
            "some ",
            "various ",
            "multiple ",
        ]

        name_lower = name.lower()
        for prefix in prefixes_to_remove:
            if name_lower.startswith(prefix):
                name = name[len(prefix) :]
                break

        # Capitalize first letter
        if name:
            name = name[0].upper() + name[1:]

        return name.strip()

    def _is_likely_object_name(self, text: str) -> bool:
        """Check if text is likely to be an object name."""
        text_lower = text.lower()

        # Skip generic terms
        generic_terms = {
            "object",
            "entity",
            "item",
            "element",
            "component",
            "thing",
            "behavior",
            "movement",
            "action",
            "interaction",
            "response",
            "complete",
            "behavioral",
            "specification",
            "analysis",
        }

        if text_lower in generic_terms:
            return False

        # Skip very long descriptions
        if len(text) > 40:
            return False

        # Skip sentences (contain multiple spaces or end with punctuation)
        if text.count(" ") > 3 or text.endswith("."):
            return False

        # Must contain at least one letter
        if not any(c.isalpha() for c in text):
            return False

        return True

    def _is_valid_object_name(self, name: str) -> bool:
        """Check if a name is valid for object detection."""
        if len(name) < 3 or len(name) > 30:
            return False

        # Must start with a letter
        if not name[0].isalpha():
            return False

        # Skip common analysis terms
        analysis_terms = {
            "movement",
            "targeting",
            "behavior",
            "specification",
            "analysis",
            "interaction",
            "response",
            "section",
            "complete",
            "visual",
        }

        if name.lower() in analysis_terms:
            return False

        return True

    def get_extracted_objects_for_detection(self) -> List[str]:
        """Get the list of extracted objects suitable for detection prompts."""
        return list(self.extracted_objects)

    def create_universal_grounded_sam_prompt(self) -> str:
        """Create a Grounding DINO prompt using extracted objects."""
        if not self.extracted_objects:
            # Fallback to universal prompts
            universal_prompts = [
                "character",
                "player",
                "enemy",
                "obstacle",
                "collectible",
                "coin",
                "item",
                "button",
                "platform",
                "vehicle",
            ]
            return " . ".join(universal_prompts) + " ."

        # Use extracted objects
        object_list = list(self.extracted_objects)[
            :25
        ]  # Limit to avoid too long prompts
        return " . ".join(object_list) + " ."


def main():
    """CLI entry point for video analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze gameplay videos for behavior extraction"
    )
    parser.add_argument(
        "target_path",
        nargs="?",
        help="Path to session directory OR game directory to analyze all sessions",
    )
    parser.add_argument(
        "--all-sessions",
        "-a",
        action="store_true",
        help="Analyze all sessions in the specified game directory",
    )

    args = parser.parse_args()

    if args.target_path:
        target_path = Path(args.target_path)

        # Check if it's a game directory (contains multiple session folders) or single session
        if target_path.is_dir():
            # List potential sessions
            potential_sessions = list_sessions(target_path)

            if potential_sessions and (
                args.all_sessions or len(potential_sessions) > 1
            ):
                # It's a game directory
                if args.all_sessions:
                    session_paths = potential_sessions
                else:
                    # Show options for individual or all sessions
                    print(
                        f"\n📂 Found {len(potential_sessions)} session(s) in {target_path.name}:"
                    )
                    for i, session in enumerate(potential_sessions, 1):
                        print(f"  [{i}] {session.name}")
                    print(
                        f"  [A] Analyze ALL {len(potential_sessions)} sessions together"
                    )

                    while True:
                        choice = (
                            input(
                                f"\n🔢 Select option [1-{len(potential_sessions)}, A]: "
                            )
                            .strip()
                            .upper()
                        )
                        if choice == "A":
                            session_paths = potential_sessions
                            break
                        else:
                            try:
                                session_idx = int(choice) - 1
                                if 0 <= session_idx < len(potential_sessions):
                                    session_paths = [potential_sessions[session_idx]]
                                    break
                                else:
                                    print("❌ Invalid selection. Please try again.")
                            except ValueError:
                                print("❌ Please enter a valid number or 'A'.")
            else:
                # It's a single session directory
                session_paths = [target_path]
        else:
            print(f"❌ Path not found: {target_path}")
            sys.exit(1)
    else:
        # Interactive selection
        base_data_dir = Path("data")

        # Ask for game name
        game_name = input("🎮 Enter the game name: ").strip()
        while not game_name:
            game_name = input("Please enter a non-empty game name: ").strip()

        game_dir = base_data_dir / sanitize_path_component(game_name)
        if not game_dir.exists():
            print(f"❌ Game directory not found: {game_dir}")
            sys.exit(1)

        # List sessions
        sessions = list_sessions(game_dir)
        if not sessions:
            print(f"❌ No sessions found for game '{game_name}'")
            sys.exit(1)

        print(f"\n📂 Found {len(sessions)} session(s):")
        for i, session in enumerate(sessions, 1):
            print(f"  [{i}] {session.name}")
        if len(sessions) > 1:
            print(f"  [A] Analyze ALL {len(sessions)} sessions together")

        # Select session(s)
        while True:
            if len(sessions) > 1:
                choice = (
                    input(f"\n🔢 Select option [1-{len(sessions)}, A]: ")
                    .strip()
                    .upper()
                )
                if choice == "A":
                    session_paths = sessions
                    break
            else:
                choice = input(f"\n🔢 Select session [1]: ").strip()

            try:
                session_idx = int(choice) - 1
                if 0 <= session_idx < len(sessions):
                    session_paths = [sessions[session_idx]]
                    break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                if len(sessions) > 1 and choice != "A":
                    print("❌ Please enter a valid number or 'A'.")
                else:
                    print("❌ Please enter a valid number.")

    # Run analysis
    try:
        analyzer = VideoGameplayAnalyzer(session_paths)
        analysis_text = analyzer.run_analysis()

        if len(session_paths) > 1:
            print(f"\n🎉 Multi-session video analysis completed successfully!")
            print(f"📊 Analyzed {len(session_paths)} sessions together")
        else:
            print(f"\n🎉 Video analysis completed successfully!")
        print(f"📁 Results saved in: {analyzer.video_analysis_dir}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
