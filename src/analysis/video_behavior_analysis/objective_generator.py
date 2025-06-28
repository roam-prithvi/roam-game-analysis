"""
Generate natural language objective descriptions from video analysis outputs.
Processes detailed video analysis files to create prompts for objective-builder-v2-roam.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.genai as genai
from pydantic import BaseModel, Field

from src import GEMINI_API_KEY


class ObjectiveDescription(BaseModel):
    """Represents a game objective extracted from video analysis."""
    objective_type: str = Field(description="Type of objective (Score, Time, Survival, etc.)")
    description: str = Field(description="Natural language description of the objective")
    resources_involved: List[str] = Field(description="Game resources involved in this objective")
    conditions: List[str] = Field(description="Conditions or requirements for the objective")
    sequence_notes: Optional[str] = Field(description="Notes about sequencing with other objectives")


class GameObjectives(BaseModel):
    """Complete set of objectives for a game."""
    game_name: str = Field(description="Name of the game")
    core_gameplay_loop: str = Field(description="Brief description of the core gameplay")
    primary_objectives: List[ObjectiveDescription] = Field(description="Main objectives")
    secondary_objectives: List[ObjectiveDescription] = Field(description="Optional/bonus objectives")
    resources: List[Dict[str, str]] = Field(description="Resources that need to be tracked")
    natural_language_prompt: str = Field(description="Complete prompt for objective-builder-v2-roam")


class VideoAnalysisObjectiveGenerator:
    """Generates objective descriptions from video analysis outputs."""
    
    def __init__(self, analysis_file_path: Path):
        """
        Initialize with path to a video analysis output file.
        
        Args:
            analysis_file_path: Path to detailed_analysis_*.txt file
        """
        self.analysis_file_path = Path(analysis_file_path)
        if not self.analysis_file_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {analysis_file_path}")
        
        # Set up output directory
        self.output_dir = self.analysis_file_path.parent / "objectives"
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure Gemini client
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def load_analysis(self) -> str:
        """Load the video analysis text."""
        with open(self.analysis_file_path, 'r') as f:
            return f.read()
    
    def extract_game_info(self, analysis_text: str) -> Tuple[str, str]:
        """Extract game name and session info from analysis."""
        # Try to extract from file path first
        game_name = "Unknown Game"
        session_info = ""
        
        # Look for game name in path
        path_parts = self.analysis_file_path.parts
        if "data" in path_parts:
            data_idx = path_parts.index("data")
            if data_idx + 1 < len(path_parts):
                game_name = path_parts[data_idx + 1]
        
        # Extract session info from analysis header
        lines = analysis_text.split('\n')
        for line in lines[:20]:  # Check first 20 lines
            if "Sessions analyzed:" in line:
                session_info = line.strip()
            elif line.strip().startswith("- ") and "at" in line:
                session_info += f"\n{line.strip()}"
        
        return game_name, session_info
    
    def create_objective_generation_prompt(self, analysis_text: str) -> str:
        """Create prompt for generating objectives from video analysis."""
        
        prompt = f"""You are analyzing a detailed video game analysis to generate objectives for an objective-based game design system.

The analysis below describes gameplay patterns, object behaviors, mechanics, and player interactions observed in actual gameplay footage.

Your task is to:
1. Identify the core objectives that define this game's progression
2. Extract resources that need to be tracked (collectibles, enemies, power-ups, etc.)
3. Create natural language descriptions suitable for the objective-builder-v2-roam tool

IMPORTANT GUIDELINES:
- Focus on OBSERVABLE objectives from the gameplay (collect X items, survive Y time, defeat Z enemies)
- Identify both primary objectives (core progression) and secondary objectives (bonuses, achievements)
- Extract all trackable resources (coins, power-ups, enemies, obstacles, etc.)
- Consider time-based objectives if the game has survival elements
- Look for sequential objectives (do A, then B, then C)
- Identify failure conditions that end the game

VIDEO ANALYSIS:
{analysis_text}

Based on this analysis, generate:

1. **CORE GAMEPLAY LOOP**: A brief description of what the player repeatedly does

2. **PRIMARY OBJECTIVES**: The main goals that drive gameplay
   - For each objective, specify:
     - Type (Score, Time, Survival, Collection, Destruction)
     - Clear description
     - Resources involved
     - Success/failure conditions
     - Any sequencing requirements

3. **SECONDARY OBJECTIVES**: Optional or bonus goals
   - Same format as primary objectives

4. **RESOURCES TO TRACK**: All game elements that need tracking
   - Resource name
   - Category (collectible, enemy, power-up, obstacle, etc.)
   - How it's acquired/lost
   - What it affects

5. **NATURAL LANGUAGE PROMPT**: A complete, detailed prompt that describes all objectives in natural language suitable for objective-builder-v2-roam. This should be structured and clear, like:
   "The player must [objective 1], then [objective 2] within [time limit]. Additionally, [secondary objectives]..."

FORMAT YOUR RESPONSE AS:

## CORE GAMEPLAY LOOP
[Description]

## PRIMARY OBJECTIVES
### Objective 1: [Name]
- Type: [Score/Time/Survival/etc.]
- Description: [What the player must do]
- Resources: [What's being tracked]
- Conditions: [Success/failure criteria]
- Sequence: [If part of a sequence]

### Objective 2: [Name]
[Same format...]

## SECONDARY OBJECTIVES
[Same format as primary...]

## RESOURCES TO TRACK
### Resource 1: [Name]
- Category: [Type]
- Acquisition: [How obtained]
- Purpose: [What it's for]

### Resource 2: [Name]
[Same format...]

## NATURAL LANGUAGE PROMPT FOR OBJECTIVE BUILDER
[Complete, structured prompt that captures all objectives in a clear, sequential manner]"""

        return prompt
    
    def generate_objectives(self) -> Dict:
        """Generate objectives from the video analysis."""
        try:
            # Load analysis
            analysis_text = self.load_analysis()
            game_name, session_info = self.extract_game_info(analysis_text)
            
            print(f"🎮 Generating objectives for: {game_name}")
            print(f"📄 Analysis file: {self.analysis_file_path.name}")
            
            # Create prompt
            prompt = self.create_objective_generation_prompt(analysis_text)
            
            print("🤖 Analyzing with Gemini 2.5 Flash...")
            
            # Generate objectives
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt]
            )
            
            objectives_text = response.text
            print("✅ Objective generation complete!")
            
            # Save raw response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"objectives_{game_name.lower().replace(' ', '_')}_{timestamp}.md"
            
            with open(output_file, 'w') as f:
                f.write(f"# Game Objectives: {game_name}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Source: {self.analysis_file_path.name}\n")
                f.write(f"{session_info}\n\n")
                f.write("---\n\n")
                f.write(objectives_text)
            
            print(f"💾 Objectives saved to: {output_file}")
            
            # Extract the natural language prompt section
            natural_language_prompt = self.extract_natural_language_prompt(objectives_text)
            
            if natural_language_prompt:
                # Save just the prompt separately for easy copying
                prompt_file = self.output_dir / f"objective_prompt_{game_name.lower().replace(' ', '_')}_{timestamp}.txt"
                with open(prompt_file, 'w') as f:
                    f.write(natural_language_prompt)
                print(f"📝 Natural language prompt saved to: {prompt_file}")
                
                # Display the prompt
                print("\n" + "="*60)
                print("📋 NATURAL LANGUAGE PROMPT FOR OBJECTIVE BUILDER:")
                print("="*60)
                print(natural_language_prompt)
                print("="*60)
            
            return {
                "game_name": game_name,
                "objectives_text": objectives_text,
                "natural_language_prompt": natural_language_prompt,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            print(f"❌ Objective generation failed: {e}")
            raise
    
    def extract_natural_language_prompt(self, objectives_text: str) -> Optional[str]:
        """Extract the natural language prompt section from the generated text."""
        # Look for the natural language prompt section
        patterns = [
            r"## NATURAL LANGUAGE PROMPT.*?\n(.*?)(?=\n##|\Z)",
            r"NATURAL LANGUAGE PROMPT.*?:\s*\n(.*?)(?=\n##|\Z)",
            r"### Natural Language Prompt.*?\n(.*?)(?=\n##|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, objectives_text, re.DOTALL | re.IGNORECASE)
            if match:
                prompt_text = match.group(1).strip()
                # Clean up any markdown formatting
                prompt_text = prompt_text.replace("```", "").strip()
                return prompt_text
        
        # If no specific section found, look for any paragraph that looks like a prompt
        lines = objectives_text.split('\n')
        for i, line in enumerate(lines):
            if "player must" in line.lower() or "the player" in line.lower():
                # Collect this and subsequent lines until a blank line or header
                prompt_lines = []
                for j in range(i, len(lines)):
                    if lines[j].strip() == "" or lines[j].startswith("#"):
                        break
                    prompt_lines.append(lines[j])
                if prompt_lines:
                    return '\n'.join(prompt_lines).strip()
        
        return None


def main():
    """CLI entry point for objective generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate game objectives from video analysis outputs"
    )
    parser.add_argument(
        "analysis_file",
        type=str,
        help="Path to the video analysis output file (detailed_analysis_*.txt)"
    )
    parser.add_argument(
        "--list-analyses",
        action="store_true",
        help="List all available analysis files in the data directory"
    )
    
    args = parser.parse_args()
    
    if args.list_analyses:
        # List all analysis files
        data_dir = Path("data")
        analysis_files = list(data_dir.glob("*/*/analysis/video_behavior_analysis/detailed_analysis_*.txt"))
        
        if not analysis_files:
            print("❌ No analysis files found in data directory")
            return
        
        print(f"\n📄 Found {len(analysis_files)} analysis file(s):")
        for i, file_path in enumerate(analysis_files, 1):
            game_name = file_path.parts[-5] if len(file_path.parts) > 5 else "Unknown"
            session_name = file_path.parts[-4] if len(file_path.parts) > 4 else "Unknown"
            print(f"  [{i}] {game_name} / {session_name} / {file_path.name}")
        
        # Allow selection
        while True:
            try:
                choice = input(f"\n🔢 Select file to generate objectives [1-{len(analysis_files)}]: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(analysis_files):
                    selected_file = analysis_files[idx]
                    break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Generate objectives for selected file
        generator = VideoAnalysisObjectiveGenerator(selected_file)
        generator.generate_objectives()
    
    else:
        # Generate objectives for specified file
        analysis_path = Path(args.analysis_file)
        if not analysis_path.exists():
            print(f"❌ Analysis file not found: {analysis_path}")
            return
        
        generator = VideoAnalysisObjectiveGenerator(analysis_path)
        generator.generate_objectives()


if __name__ == "__main__":
    main() 