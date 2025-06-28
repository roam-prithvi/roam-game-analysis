# PvZ2 Data Issue - RESOLVED

## Problem
The behavior tree generator was producing incorrect output for PvZ2, mentioning trains and barriers which are from Subway Surfers.

## Root Cause
The `frame_analysis.json` files in both PvZ2 sessions contained Subway Surfers data instead of PvZ2 data. This happened because:
1. `analyze.py` had `PROD = False` which used mock data
2. The mock data (`mock_scene_analysis.py`) was hardcoded with Subway Surfers assets

## Solution Implemented

### 1. Fixed analyze.py
- Set `PROD = True` to use real Gemini API analysis instead of mock data
- Added game detection and validation warnings
- Created game-specific mock data generators for development mode

### 2. Improved bt_generator.py
- Made prompts data-driven and game-agnostic
- Added game type detection from actual data
- Removed hardcoded examples that could mislead the LLM

### 3. Added game_specific_mocks.py
- Created separate mock generators for different games
- PvZ mock includes plants, zombies, sun counter
- Subway Surfers mock includes trains, coins, skateboard
- Falls back to generic mock for unknown games

## How to Re-analyze PvZ2 Data

Run the provided script:
```bash
./reanalyze_pvz2.sh
```

Or manually:
```bash
python -m src.analysis.analyze data/PvZ2/25-06-25_at_00.48.15
python -m src.analysis.analyze data/PvZ2/25-06-25_at_00.42.21
```

Then generate behavior descriptions:
```bash
python -m src.analysis.bt_generator --game PvZ2 --session 1 --format text
```

## Prevention
- Always use `PROD = True` for real analysis
- Mock data is now game-aware in development mode
- Validation warnings detect game/data mismatches 