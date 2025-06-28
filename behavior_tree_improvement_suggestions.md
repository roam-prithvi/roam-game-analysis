# Behavior Tree Generation Improvements

## Summary of Changes Made

I've successfully updated `bt_generator.py` to add a command-line flag for plain text output:

### Usage Examples:
```bash
# Generate JSON format (default)
python -m src.analysis.bt_generator

# Generate plain text format
python -m src.analysis.bt_generator --format text

# Non-interactive mode with arguments
python -m src.analysis.bt_generator --format text --game "subway surfers" --session 1
```

### Text Format Features:
- Clean hierarchical structure with indentation showing parent-child relationships
- Separate sections for metadata, causal rules, and tree structure
- Node types shown in brackets: `[Selector]`, `[Sequence]`, `[Action]`, `[Condition]`
- Node parameters shown in parentheses
- Examples included for each causal rule

## Suggestions for More Detailed Behavior Trees

The current behavior tree is limited by the short recording (only 3 timeline entries). Here are suggestions to make it more detailed:

### 1. **Collect Longer Gameplay Sessions**
- Record at least 5-10 minutes of gameplay
- Include multiple runs (deaths and restarts)
- Capture various game scenarios:
  - Different power-up collections
  - Various obstacle patterns
  - Special events (e.g., bonus rounds)
  - Different character abilities

### 2. **Enhance Timeline Processing**
I've already improved the `prepare_timeline_summary()` method to:
- Process up to 100 entries (increased from 50)
- Include more detailed asset information
- Track asset changes from actions
- Add pattern analysis for action frequencies

### 3. **Improve Causal Rule Generation**
The updated prompt now requests more detailed rules:
- Combo patterns
- Power-up effects
- Score mechanics
- Environmental hazards
- Lane switching mechanics
- Specific obstacle patterns
- Speed and difficulty progression

### 4. **Enhanced Behavior Tree Structure**
The tree generation prompt now includes:
- Priority-based sequences
- Multiple condition types
- Various action nodes
- Decorator nodes for timing/repetition
- Risk vs reward decision handling

### 5. **Additional Data Collection Suggestions**

#### a. **Frame Analysis Enhancement**
- Run frame analysis with shorter intervals (e.g., every 0.5 seconds instead of 1 second)
- This captures more granular game state changes

#### b. **Action Analysis Enhancement**
- Ensure touch events are captured with higher precision
- Include gesture duration and velocity data

#### c. **Game-Specific Metadata**
Consider adding a game configuration file that includes:
```json
{
  "game_name": "Subway Surfers",
  "lanes": 3,
  "actions": ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "double_tap"],
  "power_ups": ["jetpack", "magnet", "2x_multiplier", "sneakers", "hoverboard"],
  "obstacles": ["train", "barrier", "moving_train", "tunnel"],
  "collectibles": ["coin", "key", "mystery_box", "weekly_hunt_token"]
}
```

### 6. **Multi-Session Analysis**
Create a script to analyze multiple sessions together:
```python
# Pseudo-code for multi-session analysis
def analyze_multiple_sessions(game_dir: Path):
    all_timelines = []
    for session in list_sessions(game_dir):
        generator = BehaviorTreeGenerator(str(session))
        generator.load_and_process_data()
        all_timelines.extend(generator.unified_timeline)
    
    # Generate rules from combined data
    combined_generator = BehaviorTreeGenerator(str(game_dir))
    combined_generator.unified_timeline = all_timelines
    combined_generator.generate_causal_rules()
    combined_generator.generate_tree_plan()
```

### 7. **State Machine Integration**
Consider adding state tracking:
- Player state (running, jumping, rolling, surfing)
- Game state (normal, bonus round, near miss, power-up active)
- Environmental state (speed level, obstacle density)

### 8. **Performance Metrics**
Track and use performance data:
- Average reaction time to obstacles
- Success rate for different maneuvers
- Optimal lane positioning statistics
- Power-up effectiveness metrics

### 9. **Adaptive Difficulty Response**
Add nodes that respond to game difficulty:
```
[Sequence] Adapt to increasing speed
  [Condition] Check game speed (speed_level > 5)
  [Selector] Choose conservative strategy
    [Action] Maintain center lane
    [Action] Prioritize obstacle avoidance over coins
```

### 10. **Learning from Failures**
Analyze death scenarios to improve the tree:
- What caused each game over?
- What actions could have prevented it?
- Add specific failure-prevention sequences

## Implementation Priority

1. **Immediate**: Collect longer gameplay sessions (5-10 minutes)
2. **Short-term**: Add game-specific configuration
3. **Medium-term**: Implement multi-session analysis
4. **Long-term**: Add state machine and learning components

The current implementation provides a solid foundation. With more data and these enhancements, the behavior trees will become significantly more sophisticated and effective for AI gameplay. 