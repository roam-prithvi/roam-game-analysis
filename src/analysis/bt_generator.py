from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
import argparse

import instructor
from google.generativeai import GenerativeModel
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import GEMINI_API_KEY
# Corrected imports to be absolute
from src.analysis.base_models import SceneAnalysis, ActionAnalysis
from src.streaming.android_streamer import sanitize_path_component
from src.util import list_sessions
import sys
from datetime import datetime
from pathlib import Path


# Pydantic models for structured outputs
class CausalRule(BaseModel):
    if_condition: str = Field(description="The condition that triggers this rule (IF)")
    then_action: str = Field(description="The action to take when condition is met (THEN)")
    confidence: float = Field(description="Confidence level of this rule (0-1)", ge=0, le=1)
    examples: List[str] = Field(description="Specific examples from the data supporting this rule")


class CausalRuleSet(BaseModel):
    rules: List[CausalRule] = Field(description="List of causal rules derived from the gameplay")
    game_summary: str = Field(description="Brief summary of the game mechanics observed")


class NodePlan(BaseModel):
    node_id: str = Field(description="Unique identifier for this node")
    node_type: str = Field(description="Type of behavior tree node (Sequence, Selector, Action, Condition, etc.)")
    description: str = Field(description="What this node does")
    children: List[str] = Field(description="List of child node IDs", default_factory=list)
    node_parameters: Dict[str, Any] = Field(
        description="Parameters specific to this node type (e.g., condition values, action parameters)",
        default_factory=dict
    )


class BehaviorTreePlan(BaseModel):
    nodes: List[NodePlan] = Field(description="All nodes in the behavior tree")
    root_node_id: str = Field(description="ID of the root node")
    description: str = Field(description="High-level description of the behavior tree")


class BehaviorDescription(BaseModel):
    """Natural language description of AI behavior"""
    primary_goal: str = Field(description="The main objective or purpose of this AI behavior")
    behavior_rules: List[str] = Field(description="Ordered list of behavioral rules, from highest to lowest priority")
    detailed_description: str = Field(description="A comprehensive description of how the AI should behave")


class BehaviorTreeGenerator:
    def __init__(self, session_path: str, output_format: str = "json"):
        self.session_path = session_path
        self.analysis_dir = os.path.join(session_path, "analysis")
        self.frame_data: List[SceneAnalysis] = []
        self.action_data: List[ActionAnalysis] = []
        self.unified_timeline: List[Dict[str, Any]] = []
        self.causal_rules: Optional[CausalRuleSet] = None
        self.tree_plan: Optional[BehaviorTreePlan] = None
        self.behavior_description: Optional[BehaviorDescription] = None
        self.output_format = output_format
        
        # Configure the GenAI SDK (idempotent)
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize the instructor client with the sync interface
        self.client = instructor.from_gemini(
            client=GenerativeModel(model_name="gemini-1.5-flash-latest"),
        )

    def load_and_process_data(self) -> None:
        """Load frame and action analysis data and merge into unified timeline."""
        # Load frame analysis
        frame_path = os.path.join(self.analysis_dir, "frame_analysis.json")
        with open(frame_path, "r") as f:
            frame_data = json.load(f)

        # Load action analysis
        action_path = os.path.join(self.analysis_dir, "action_analysis.json")
        with open(action_path, "r") as f:
            action_data = json.load(f)

        # Create a lookup for actions by timestamp for efficient merging
        actions_by_timestamp: Dict[float, ActionAnalysis] = {}
        for action_entry in action_data:
            try:
                # Timestamps in the JSON are now a field, not the key
                ts = float(action_entry["timestamp"])
                actions_by_timestamp[ts] = ActionAnalysis(**action_entry)
            except (ValueError, ValidationError, IndexError, KeyError) as e:
                print(f"Skipping malformed action entry: {action_entry}, error: {e}")
                continue

        sorted_action_timestamps = sorted(actions_by_timestamp.keys())

        for frame_entry in frame_data:
            try:
                # Timestamps in the JSON are now a field, not the key
                ts_str = Path(frame_entry["frame"]).stem.split("_")[0]
                ts = float(ts_str)
                scene = SceneAnalysis(**frame_entry)

                # Find the corresponding action (if any)
                action = None
                for action_ts in sorted_action_timestamps:
                    if abs(ts - action_ts) < 1.0:  # Within 1 second
                        action = actions_by_timestamp[action_ts]
                        break

                # Create unified entry
                unified_entry = {
                    "timestamp": ts,
                    "scene": scene,
                    "action": action,
                }
                self.unified_timeline.append(unified_entry)

            except (ValueError, ValidationError, IndexError, KeyError) as e:
                print(f"Skipping malformed frame entry: {frame_entry}, error: {e}")
                continue

        # Sort timeline by timestamp
        self.unified_timeline.sort(key=lambda x: x["timestamp"])
        print(f"✅ Loaded {len(self.unified_timeline)} timeline entries")

    def prepare_timeline_summary(self) -> str:
        """Create a detailed summary of the game session for the AI."""
        summary_parts = []
        
        # First, add a header that helps identify the game type
        game_identifiers = set()
        asset_names = set()
        ui_elements = set()
        
        # Scan first few entries to understand the game
        for entry in self.unified_timeline[:20]:
            scene = entry["scene"]
            if scene.assets:
                for asset in scene.assets:
                    asset_names.add(asset.name.lower())
            if scene.ui_elements:
                for ui in scene.ui_elements:
                    ui_elements.add(ui.name.lower())
        
        # Try to identify game type from assets and UI
        if any(term in str(asset_names) for term in ["plant", "zombie", "sun", "peashooter", "wall-nut"]):
            summary_parts.append("=== Game Type: Tower Defense (Plants vs Zombies style) ===")
        elif any(term in str(asset_names) for term in ["train", "coin", "jetpack", "skateboard"]):
            summary_parts.append("=== Game Type: Endless Runner ===")
        else:
            summary_parts.append("=== Game Type: Analyzing... ===")
        
        summary_parts.append(f"Common assets: {', '.join(list(asset_names)[:10])}")
        summary_parts.append("")
        
        # Use more entries for better context
        max_entries = min(len(self.unified_timeline), 100)
        
        for i, entry in enumerate(self.unified_timeline[:max_entries]):
            ts = entry["timestamp"]
            scene = entry["scene"]
            action = entry["action"]
            
            scene_desc = f"[{i+1}] Time {ts:.1f}s:"
            
            # Focus on action first if present (more important for understanding gameplay)
            if action:
                scene_desc += f" ACTION: {action.player_action_description}"
                
                # Include significant changes
                significant_changes = []
                if action.assets:
                    for action_asset in action.assets[:3]:
                        if action_asset.changes_due_to_action and "remained unchanged" not in action_asset.changes_due_to_action.lower():
                            significant_changes.append(f"{action_asset.name}: {action_asset.changes_due_to_action}")
                
                if action.ui_elements:
                    for ui_elem in action.ui_elements[:2]:
                        if ui_elem.changes_due_to_action and "remained unchanged" not in ui_elem.changes_due_to_action.lower():
                            significant_changes.append(f"{ui_elem.name}: {ui_elem.changes_due_to_action}")
                
                if significant_changes:
                    scene_desc += f" | Results: {'; '.join(significant_changes[:3])}"
            
            # Add scene context briefly
            if scene.assets:
                key_assets = [asset.name for asset in scene.assets[:3]]
                scene_desc += f" | Scene contains: {', '.join(key_assets)}"
            
            summary_parts.append(scene_desc)
        
        # Add a summary of key game patterns
        summary_parts.append("\n--- Observed Gameplay Patterns ---")
        
        # Count action types more specifically
        action_patterns = {}
        tap_targets = []
        
        for entry in self.unified_timeline:
            if entry["action"]:
                action_desc = entry["action"].player_action_description.lower()
                
                # Categorize actions more specifically
                if "tap" in action_desc:
                    if "plant" in action_desc:
                        action_patterns["plant_placement"] = action_patterns.get("plant_placement", 0) + 1
                    elif "grid" in action_desc or "square" in action_desc:
                        action_patterns["grid_interaction"] = action_patterns.get("grid_interaction", 0) + 1
                    else:
                        action_patterns["tap_other"] = action_patterns.get("tap_other", 0) + 1
                elif "swipe" in action_desc:
                    direction = "unknown"
                    if "left" in action_desc: direction = "left"
                    elif "right" in action_desc: direction = "right"
                    elif "up" in action_desc: direction = "up"
                    elif "down" in action_desc: direction = "down"
                    action_patterns[f"swipe_{direction}"] = action_patterns.get(f"swipe_{direction}", 0) + 1
        
        if action_patterns:
            summary_parts.append("Player actions: " + ", ".join([f"{k}: {v}" for k, v in sorted(action_patterns.items())]))
        
        return "\n".join(summary_parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate_causal_rules(self) -> None:
        """Use AI to generate causal rules from the unified timeline."""
        timeline_summary = self.prepare_timeline_summary()
        
        # Analyze the actual game type from the data
        game_type_hints = []
        for entry in self.unified_timeline[:10]:
            if entry["action"]:
                action_desc = entry["action"].player_action_description.lower()
                if "plant" in action_desc or "zombie" in action_desc or "sun" in action_desc:
                    game_type_hints.append("tower defense")
                elif "swipe" in action_desc and ("lane" in action_desc or "jump" in action_desc):
                    game_type_hints.append("endless runner")
                elif "match" in action_desc or "swap" in action_desc:
                    game_type_hints.append("match-3")
        
        prompt = f"""Analyze this game session data and extract causal rules (IF-THEN patterns) that describe the game's mechanics and optimal player behavior.

IMPORTANT: Base your analysis ONLY on the actual data provided. Do not make assumptions about game mechanics that aren't evident in the timeline.

Game Session Timeline:
{timeline_summary}

Based ONLY on what you observe in this specific data, identify:
1. Core game mechanics visible in the actions and their results
2. Player strategies that appear to be successful
3. Cause-and-effect patterns between actions and outcomes
4. Resource management patterns (if any)
5. Defensive or offensive patterns (if any)
6. Timing-based patterns (if any)
7. Failure conditions or negative outcomes (if any)

For each rule you identify:
- It must be directly supported by events in the timeline
- Include the specific timestamp or example from the data
- Don't assume mechanics that aren't shown in the data
- Focus on what actually happened, not what might happen

Remember: Different games have different mechanics. This could be:
- A tower defense game (placing units, managing resources)
- An endless runner (swiping to avoid obstacles)
- A puzzle game (matching, solving)
- Or something else entirely

Let the data tell you what kind of game this is and what rules apply."""

        try:
            print("Generating causal rules from game events...")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=CausalRuleSet,
                max_retries=3,
            )
            self.causal_rules = response
            print(f"✅ Generated {len(self.causal_rules.rules)} causal rules")
        except Exception as e:
            print(f"Error calling LLM for causal rule generation: {e}")
            # Fallback rules
            self.causal_rules = CausalRuleSet(
                rules=[
                    CausalRule(
                        if_condition="obstacle detected ahead",
                        then_action="swipe up to jump or swipe left/right to dodge",
                        confidence=0.9,
                        examples=["Timestamp 5.2s: Player swiped up when train appeared"]
                    )
                ],
                game_summary="Unable to generate rules from AI, using defaults"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate_tree_plan(self) -> None:
        """Use AI to generate a behavior tree plan based on causal rules."""
        if not self.causal_rules:
            raise ValueError("Causal rules must be generated first")
        
        rules_text = "\n".join([
            f"- IF {rule.if_condition} THEN {rule.then_action} (confidence: {rule.confidence})"
            for rule in self.causal_rules.rules
        ])
        
        prompt = f"""Based on the following game analysis and causal rules, create a natural language description of how an AI should play this game.

Game Summary: {self.causal_rules.game_summary}

Observed Patterns (Causal Rules):
{rules_text}

Create a behavioral description that:
1. States the primary goal clearly based on what was observed
2. Lists behavioral rules in priority order (most important first)
3. Uses natural language without technical terms
4. Describes what to do in different game situations
5. Includes specific details when they were observed in the data

Your description should be structured as:
- A clear primary goal (what is the AI trying to achieve?)
- A prioritized list of behaviors (what should it do and when?)
- A detailed paragraph explaining the overall strategy

Focus on describing:
- What conditions trigger each behavior (based on the causal rules)
- What specific actions to take
- Any parameters that were observed (timing, distances, quantities)
- Priority order when multiple conditions might apply
- What to do when no special conditions apply

Base everything on the causal rules provided. Don't invent new mechanics or behaviors that weren't observed.

Do NOT use technical terms like: selector, sequence, condition node, action node, behavior tree, etc.
Just describe what the AI should DO in plain language."""

        try:
            print("Generating behavior description...")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=BehaviorDescription,
                max_retries=3,
            )
            self.behavior_description = response
            print(f"✅ Generated behavior description with {len(self.behavior_description.behavior_rules)} rules")
        except Exception as e:
            print(f"Error calling LLM for behavior generation: {e}")
            # Fallback description
            self.behavior_description = BehaviorDescription(
                primary_goal="Navigate the game safely while maximizing score",
                behavior_rules=[
                    "When obstacles are detected ahead, swipe to avoid them",
                    "Collect coins and power-ups when it's safe to do so",
                    "Maintain optimal lane position for future moves"
                ],
                detailed_description="Unable to generate description from AI, using defaults"
            )

    def convert_to_json_format(self) -> Dict[str, Any]:
        """Convert the behavior description to JSON format."""
        if not self.behavior_description or not self.causal_rules:
            raise ValueError("Behavior description must be generated first")
        
        return {
            "behavior_description": {
                "primary_goal": self.behavior_description.primary_goal,
                "behavior_rules": self.behavior_description.behavior_rules,
                "detailed_description": self.behavior_description.detailed_description
            },
            "metadata": {
                "game_summary": self.causal_rules.game_summary,
                "causal_rules": [
                    {
                        "if": rule.if_condition,
                        "then": rule.then_action,
                        "confidence": rule.confidence
                    }
                    for rule in self.causal_rules.rules
                ],
                "generated_at": datetime.now().isoformat(),
                "session_path": self.session_path
            }
        }

    def convert_to_text_format(self) -> str:
        """Convert the behavior description to plain text format."""
        if not self.behavior_description or not self.causal_rules:
            raise ValueError("Behavior description and causal rules must be generated first")
        
        lines = []
        lines.append("=== AI BEHAVIOR DESCRIPTION ===")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Game: {self.causal_rules.game_summary}")
        lines.append("")
        
        lines.append(f"PRIMARY GOAL: {self.behavior_description.primary_goal}")
        lines.append("")
        
        lines.append("BEHAVIOR RULES (in priority order):")
        for i, rule in enumerate(self.behavior_description.behavior_rules, 1):
            lines.append(f"{i}. {rule}")
        lines.append("")
        
        lines.append("DETAILED DESCRIPTION:")
        # Word wrap the detailed description for readability
        import textwrap
        wrapped_text = textwrap.fill(self.behavior_description.detailed_description, width=80)
        lines.append(wrapped_text)
        lines.append("")
        
        lines.append("=== ANALYSIS DATA ===")
        lines.append(f"Based on {len(self.causal_rules.rules)} observed patterns:")
        for i, rule in enumerate(self.causal_rules.rules, 1):
            lines.append(f"{i}. When {rule.if_condition}, then {rule.then_action} (confidence: {rule.confidence:.2f})")
        
        return "\n".join(lines)

    def run(self) -> None:
        """Main execution flow."""
        print(f"\n🤖 Starting Behavior Description Generation for: {self.session_path}")
        
        # Load and process data
        self.load_and_process_data()
        
        # Generate causal rules
        self.generate_causal_rules()
        
        # Generate behavior description
        self.generate_tree_plan()
        
        if self.output_format == "text":
            # Convert to text format
            behavior_text = self.convert_to_text_format()
            
            # Save to file
            output_path = os.path.join(self.analysis_dir, "ai_behavior.txt")
            with open(output_path, "w") as f:
                f.write(behavior_text)
            
            print(f"\n✅ Behavior description saved to: {output_path}")
            print(f"📋 Based on {len(self.causal_rules.rules)} causal rules")
            
            # Also print to console
            print("\n" + "="*50)
            print(behavior_text)
        else:
            # Convert to JSON format (default)
            behavior_json = self.convert_to_json_format()
            
            # Save to file
            output_path = os.path.join(self.analysis_dir, "ai_behavior.json")
            with open(output_path, "w") as f:
                json.dump(behavior_json, f, indent=2)
            
            print(f"\n✅ Behavior description saved to: {output_path}")
            print(f"📋 Based on {len(self.causal_rules.rules)} causal rules")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate behavior trees from game analysis data")
    parser.add_argument(
        "--format", 
        choices=["json", "text"], 
        default="json",
        help="Output format for the behavior tree (default: json)"
    )
    parser.add_argument(
        "--game",
        type=str,
        help="Game name (if provided, skips interactive prompt)"
    )
    parser.add_argument(
        "--session",
        type=int,
        help="Session number to use (1-based index, if provided with --game, skips interactive prompt)"
    )
    
    args = parser.parse_args()
    
    base_data_dir: Path = Path("data")

    # Handle game name
    if args.game:
        game_name = args.game.strip()
    else:
        # 1) Ask for the game name
        game_name: str = input("🎮  Enter the game name: ").strip()
        while not game_name:
            game_name = input("Please enter a non-empty game name: ").strip()
    
    game_dir: Path = base_data_dir / sanitize_path_component(game_name)
    if not game_dir.exists():
        print(f"❌  Game directory not found: {game_dir}", file=sys.stderr)
        sys.exit(1)

    # 2) Enumerate sessions
    sessions: List[Path] = list_sessions(game_dir)
    if not sessions:
        print(f"❌  No sessions found for game '{game_name}'.", file=sys.stderr)
        sys.exit(1)

    # Handle session selection
    if args.game and args.session:
        # Non-interactive mode
        session_idx = args.session - 1
        if 0 <= session_idx < len(sessions):
            selected_session = sessions[session_idx]
        else:
            print(f"❌  Invalid session number. Available sessions: 1-{len(sessions)}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive mode
        print(f"\n📂  Found {len(sessions)} session(s) for '{game_name}':")
        for i, session in enumerate(sessions, 1):
            print(f"  [{i}] {session.name}")

        # 3) Select a session
        while True:
            choice: str = input("\n🔢  Select a session by number: ").strip()
            try:
                session_idx: int = int(choice) - 1
                if 0 <= session_idx < len(sessions):
                    selected_session: Path = sessions[session_idx]
                    break
                else:
                    print("❌  Invalid selection. Please try again.")
            except ValueError:
                print("❌  Please enter a valid number.")

    # 4) Build the path and check if the analysis exists
    session_path: Path = selected_session
    analysis_dir: Path = session_path / "analysis"
    frame_analysis_json: Path = analysis_dir / "frame_analysis.json"
    action_analysis_json: Path = analysis_dir / "action_analysis.json"

    if not frame_analysis_json.exists() or not action_analysis_json.exists():
        print(f"❌  Required analysis files not found in: {analysis_dir}")
        print("Please run the analysis script first.")
        sys.exit(1)

    # Create and run the generator
    generator = BehaviorTreeGenerator(str(session_path), output_format=args.format)
    generator.run() 