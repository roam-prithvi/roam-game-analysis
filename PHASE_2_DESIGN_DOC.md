# Phase 2 Design Document: `bt_generator.py`

This document outlines the purpose, architecture, and implementation plan for `bt_generator.py`, which constitutes Phase 2 of the Game Design Extractor project.

## 1. High-Level Goal

The primary objective of this script is to bridge the gap between the raw, observational data produced by Phase 1 and a natural language behavior description that can be processed by your existing system.

In essence, it answers the question: **"Given what we've seen, what was the AI *trying* to do, and how can we describe that behavior in plain language?"**

It transforms a timeline of game events into a natural language description file (`ai_behavior.txt` or `ai_behavior.json`) by inferring causal relationships and describing them as behavioral rules.

## 2. The Three-Step Pipeline

The script orchestrates a three-step pipeline, executed by the `run()` method in the `BehaviorTreeGenerator` class.

**Step 1: `load_and_process_data()` - Creating a Unified Timeline**

*   **What it does:** This function reads the two key outputs from Phase 1: `frame_analysis.json` (containing per-frame scene graphs) and `action_analysis.json` (containing LLM-identified player actions).
*   **How it works:** It creates a single, time-sorted list of events. Each event links a specific game state to the player action that happened closest to it in time. A 1-second threshold is used to prevent associating unrelated actions and states.
*   **Why it's important:** This unified timeline provides the necessary context for the subsequent causal inference step, allowing the model to see "when the game state was X, the player did Y."

**Step 2: `generate_causal_rules()` - Inferring Intent**

*   **What it does:** This is the first "reasoning" step. Its goal is to identify patterns and generate "IF-THEN" rules that explain the observed behavior.
*   **How it works:**
    1.  The unified timeline is passed to an LLM (Gemini 1.5 Flash).
    2.  The LLM identifies recurring patterns and formulates them as `CausalRule` objects with conditions, actions, and confidence levels.
    3.  It also generates a high-level game summary.
*   **Current Implementation:** Uses the instructor library with Gemini to get structured outputs.

**Step 3: `generate_tree_plan()` - Creating Natural Language Description**

*   **What it does:** This step translates the causal rules into a natural language behavior description without technical jargon.
*   **How it works:** The causal rules are sent to an LLM with examples of good behavioral descriptions. The prompt explicitly instructs to avoid technical terms and describe behaviors in plain language.
*   **Output:** A `BehaviorDescription` object containing:
    - `primary_goal`: The main objective
    - `behavior_rules`: Prioritized list of behavioral rules
    - `detailed_description`: Comprehensive description of the behavior

## 3. Natural Language Output Format

The script now generates plain language descriptions instead of technical behavior trees. Examples:

**Simple Example:**
```
When danger is nearby, evade immediately. If safe, collect power-ups and coins. Otherwise, position for future moves.
```

**Detailed Example:**
```
The goal is to navigate safely while maximizing score collection.

The behavior has 3 distinct parts, prioritized in order:
A. Avoid Obstacles:
   - When obstacles appear within 2 lanes, immediately swipe away
   - Jump over low obstacles, slide under high ones
   
B. Collect Rewards:
   - When safe, move towards visible coins and power-ups
   - Prioritize power-ups over regular coins
   
C. Optimal Positioning:
   - Stay in center lane when no immediate threats
   - Prepare for upcoming obstacles by pre-positioning
```

## 4. Prerequisites and Input Files

Before running `bt_generator.py`, you must have completed Phase 1, which produces the following files:

### Required Input Files

1. **`frame_analysis.json`** - Contains scene analysis data for each frame
   - Located at: `{session_path}/analysis/frame_analysis.json`
   - Format: Array of objects with frame data and timestamps

2. **`action_analysis.json`** - Contains detected player actions
   - Located at: `{session_path}/analysis/action_analysis.json`
   - Format: Array of objects with action data and timestamps

### Directory Structure
```
data/
└── {game_name}/
    └── {session_id}/
        └── analysis/
            ├── frame_analysis.json
            └── action_analysis.json
```

## 5. How to Run the Script

### Basic Usage

1. **Ensure Phase 1 has been completed** and the required JSON files exist.

2. **Run with command line arguments:**
   ```bash
   # Interactive mode (will prompt for game and session)
   python src/analysis/bt_generator.py --format text
   
   # Non-interactive mode
   python src/analysis/bt_generator.py --game "Subway Surfers" --session 1 --format text
   ```

3. **Or import and use programmatically:**
   ```python
   from src.analysis.bt_generator import BehaviorTreeGenerator
   
   generator = BehaviorTreeGenerator("data/subway_surfers/SESSION_123", output_format="text")
   generator.run()
   ```

### Output Formats

- `--format text`: Generates `ai_behavior.txt` with formatted plain text description
- `--format json`: Generates `ai_behavior.json` with structured data

## 6. Expected Output

### Text Format Output (`ai_behavior.txt`)
```
=== AI BEHAVIOR DESCRIPTION ===
Generated: 2024-01-15 10:30:00
Game: Subway Surfers - endless runner with obstacle avoidance

PRIMARY GOAL: Navigate through the subway while avoiding obstacles and maximizing score

BEHAVIOR RULES (in priority order):
1. When trains or barriers appear ahead, immediately swipe to avoid collision
2. When coins are visible and reachable, adjust path to collect them
3. When power-ups appear, prioritize collecting them over regular coins
4. When no immediate threats, maintain center lane for flexibility

DETAILED DESCRIPTION:
The AI should play as a cautious but opportunistic runner. Safety is the top
priority - always avoid obstacles first. When safe, the AI should actively 
collect rewards, prioritizing rare power-ups over common coins. The AI should
maintain good positioning by staying in the center lane when possible, giving
maximum options for future moves.

=== ANALYSIS DATA ===
Based on 8 observed patterns:
1. When train appears in current lane, then swipe to adjacent lane (confidence: 0.95)
2. When coins form a line, then follow the coin path (confidence: 0.85)
...
```

### JSON Format Output (`ai_behavior.json`)
```json
{
  "behavior_description": {
    "primary_goal": "Navigate through the subway while avoiding obstacles and maximizing score",
    "behavior_rules": [
      "When trains or barriers appear ahead, immediately swipe to avoid collision",
      "When coins are visible and reachable, adjust path to collect them"
    ],
    "detailed_description": "The AI should play as a cautious but opportunistic runner..."
  },
  "metadata": {
    "game_summary": "Subway Surfers - endless runner with obstacle avoidance",
    "causal_rules": [...],
    "generated_at": "2024-01-15T10:30:00",
    "session_path": "data/subway_surfers/SESSION_123"
  }
}
```

## 7. Key Differences from Original Design

1. **No Technical Terms**: The output uses only natural language, avoiding terms like "selector", "sequence", "behavior tree", etc.

2. **Priority-Based Rules**: Behaviors are described as prioritized rules rather than tree structures.

3. **Human-Readable**: The descriptions can be understood by non-technical users and processed by your existing system.

4. **Flexible Output**: Supports both structured JSON and formatted text output.

## 8. Next Steps for Implementation

1. **Fine-tune prompts** for better behavior descriptions based on your specific games.

2. **Add game-specific templates** for common behavior patterns.

3. **Implement confidence thresholds** to filter out low-confidence rules.

4. **Add validation** to ensure generated descriptions match your system's expectations.

5. **Consider caching** LLM responses to reduce API costs during development. 