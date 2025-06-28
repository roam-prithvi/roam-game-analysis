"""
Game-specific prompt dictionaries for object detection.

This module contains carefully crafted text prompts for different mobile games
to improve detection accuracy with open-vocabulary models like Grounding DINO.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GamePromptConfig:
    """Configuration for game-specific prompts."""
    primary_prompts: List[str]  # Main objects to detect
    secondary_prompts: List[str]  # Less important objects
    ui_prompts: List[str]  # UI elements
    background_prompts: List[str]  # Background/environment elements
    confidence_thresholds: Dict[str, float]  # Per-prompt confidence thresholds
    size_filters: Dict[str, Dict[str, int]]  # Min/max sizes per prompt


# Game-specific configurations
GAME_CONFIGS = {
    "subway_surfers": GamePromptConfig(
        primary_prompts=[
            "character running", "person running", "subway surfer character",
            "train", "subway train", "moving train",
            "police officer", "security guard", "chasing guard",
            "coin", "gold coin", "collectible coin",
            "power up", "jetpack", "magnet", "sneakers", "multiplier"
        ],
        secondary_prompts=[
            "barrier", "obstacle", "train car", "subway car",
            "ramp", "jumping ramp", "tunnel", "subway tunnel",
            "tracks", "railway tracks", "train tracks"
        ],
        ui_prompts=[
            "score", "number", "coin counter", "distance meter",
            "pause button", "menu button", "power up icon",
            "multiplier text", "mission text"
        ],
        background_prompts=[
            "subway platform", "underground station", "city background",
            "buildings", "urban landscape", "graffiti", "walls"
        ],
        confidence_thresholds={
            "character running": 0.3,
            "train": 0.4,
            "coin": 0.2,
            "police officer": 0.3,
            "power up": 0.25
        },
        size_filters={
            "coin": {"min_size": 8, "max_size": 50},
            "character running": {"min_size": 30, "max_size": 200},
            "train": {"min_size": 100, "max_size": 800}
        }
    ),
    
    "temple_run": GamePromptConfig(
        primary_prompts=[
            "running character", "person running", "temple runner",
            "demon monkey", "evil monkey", "chasing monkey",
            "coin", "gem", "gold coin", "collectible",
            "tree", "fallen tree", "log obstacle",
            "cliff", "gap", "chasm", "pit"
        ],
        secondary_prompts=[
            "temple wall", "stone wall", "ancient wall",
            "rope", "zip line", "vine",
            "torch", "fire", "flame",
            "stairs", "stone steps", "temple steps"
        ],
        ui_prompts=[
            "score", "distance", "coin counter",
            "objective text", "mission text",
            "pause button", "menu icon"
        ],
        background_prompts=[
            "temple", "ancient temple", "stone temple",
            "forest", "jungle", "trees",
            "mountain", "cliff face", "stone path"
        ],
        confidence_thresholds={
            "running character": 0.3,
            "demon monkey": 0.4,
            "coin": 0.2,
            "tree": 0.3
        },
        size_filters={
            "coin": {"min_size": 6, "max_size": 40},
            "running character": {"min_size": 40, "max_size": 250}
        }
    ),
    
    "crossy_road": GamePromptConfig(
        primary_prompts=[
            "chicken", "character", "animal character", "player character",
            "car", "vehicle", "truck", "bus", "moving car",
            "log", "floating log", "tree trunk",
            "eagle", "bird", "flying eagle",
            "coin", "collectible"
        ],
        secondary_prompts=[
            "road", "street", "lane", "highway",
            "river", "water", "stream",
            "grass", "safe zone", "starting area",
            "tree", "obstacle tree", "stationary tree"
        ],
        ui_prompts=[
            "score", "high score", "best score",
            "character selection", "unlock text",
            "retry button", "menu button"
        ],
        background_prompts=[
            "landscape", "terrain", "game world",
            "horizon", "sky", "background scenery"
        ],
        confidence_thresholds={
            "chicken": 0.3,
            "car": 0.4,
            "log": 0.3,
            "eagle": 0.4
        },
        size_filters={
            "chicken": {"min_size": 20, "max_size": 80},
            "car": {"min_size": 30, "max_size": 150}
        }
    ),
    
    "plants_vs_zombies": GamePromptConfig(
        primary_prompts=[
            "plant", "sunflower", "peashooter", "chomper", "walnut",
            "zombie", "walking zombie", "cone zombie", "bucket zombie",
            "sun", "sunlight", "sun token",
            "pea", "projectile", "pea shot",
            "lawn mower", "mower"
        ],
        secondary_prompts=[
            "plant pot", "flower pot", "planting spot",
            "grave", "tombstone", "zombie grave",
            "pool", "water", "lily pad",
            "fog", "mist", "night fog"
        ],
        ui_prompts=[
            "seed packet", "plant icon", "plant selection",
            "sun counter", "sun meter",
            "progress bar", "level progress",
            "shovel", "tool icon"
        ],
        background_prompts=[
            "lawn", "grass", "yard", "backyard",
            "house", "home", "building",
            "fence", "yard fence"
        ],
        confidence_thresholds={
            "plant": 0.3,
            "zombie": 0.3,
            "sun": 0.2,
            "pea": 0.25
        },
        size_filters={
            "sun": {"min_size": 10, "max_size": 60},
            "plant": {"min_size": 25, "max_size": 100},
            "zombie": {"min_size": 30, "max_size": 120}
        }
    ),
    
    "plants_vs_zombies_2": GamePromptConfig(
        primary_prompts=[
            "plant", "sunflower", "peashooter", "chomper", "walnut",
            "zombie", "walking zombie", "cone zombie", "bucket zombie",
            "sun", "sunlight", "sun token",
            "pea", "projectile", "pea shot",
            "lawn mower", "mower",
            "power mint", "mint plant", "special plant",
            "plant food", "green sparkle", "plant boost",
            "premium plant", "gem plant",
            "pirate zombie", "cowboy zombie", "pharaoh zombie"
        ],
        secondary_prompts=[
            "plant pot", "flower pot", "planting spot",
            "grave", "tombstone", "zombie grave",
            "world map", "level node", "world selection",
            "star", "level star", "achievement star",
            "gem", "premium gem", "diamond"
        ],
        ui_prompts=[
            "seed packet", "plant icon", "plant selection",
            "sun counter", "sun meter",
            "gem counter", "premium currency",
            "world progress", "mastery points",
            "league icon", "arena icon"
        ],
        background_prompts=[
            "lawn", "grass", "yard", "backyard",
            "house", "home", "building",
            "world theme", "desert", "pirate ship", "wild west"
        ],
        confidence_thresholds={
            "plant": 0.3,
            "zombie": 0.3,
            "sun": 0.2,
            "plant food": 0.3
        },
        size_filters={
            "sun": {"min_size": 10, "max_size": 60},
            "plant": {"min_size": 25, "max_size": 100}
        }
    )
}

# Generic fallback prompts
GENERIC_PROMPTS = [
    "character", "player", "obstacle", "collectible", "coin",
    "enemy", "power up", "platform", "barrier", "item",
    "button", "menu", "score", "health", "weapon"
]


def get_game_prompts(game_name: str) -> List[str]:
    """Get all prompts for a specific game."""
    game_key = game_name.lower().replace(" ", "_").replace("-", "_")
    
    if game_key in GAME_CONFIGS:
        config = GAME_CONFIGS[game_key]
        all_prompts = []
        all_prompts.extend(config.primary_prompts)
        all_prompts.extend(config.secondary_prompts) 
        all_prompts.extend(config.ui_prompts)
        all_prompts.extend(config.background_prompts)
        return list(set(all_prompts))  # Remove duplicates
    
    return GENERIC_PROMPTS


def get_primary_game_prompts(game_name: str) -> List[str]:
    """Get only primary prompts for a game."""
    game_key = game_name.lower().replace(" ", "_").replace("-", "_")
    
    if game_key in GAME_CONFIGS:
        return GAME_CONFIGS[game_key].primary_prompts
    
    return ["character", "obstacle", "collectible", "enemy"]


def get_game_config(game_name: str) -> Optional[GamePromptConfig]:
    """Get full configuration for a game."""
    game_key = game_name.lower().replace(" ", "_").replace("-", "_")
    return GAME_CONFIGS.get(game_key)


def get_supported_games() -> List[str]:
    """Get list of all supported game names."""
    return list(GAME_CONFIGS.keys())


def create_grounding_prompt(prompts: List[str]) -> str:
    """
    Create a Grounding DINO compatible prompt string.
    
    Args:
        prompts: List of object names to detect
        
    Returns:
        Formatted prompt string using <and> separators
    """
    if not prompts:
        return ""
    
    # Filter out very generic terms that might cause false positives
    filtered_prompts = []
    generic_terms = {"background", "scenery", "landscape", "sky", "ground"}
    
    for prompt in prompts:
        if prompt.lower() not in generic_terms:
            filtered_prompts.append(prompt)
    
    # Join with <and> separator as required by Grounding DINO
    return " <and> ".join(filtered_prompts[:20])  # Limit to 20 prompts to avoid too long queries


def get_grounding_prompt_for_game(game_name: str, primary_only: bool = False) -> str:
    """
    Get a Grounding DINO compatible prompt string for a game.
    
    Args:
        game_name: Name of the game
        primary_only: If True, only use primary prompts
        
    Returns:
        Formatted prompt string
    """
    if primary_only:
        prompts = get_primary_game_prompts(game_name)
    else:
        prompts = get_game_prompts(game_name)
    
    return create_grounding_prompt(prompts) 