"""Game-specific mock data generators for development/testing."""

def get_pvz_mock():
    """Return mock scene analysis data for Plants vs Zombies."""
    return {
        "assets": [
            {
                "name": "Peashooter",
                "description": "A green plant that shoots peas at zombies",
                "primary_color": "green",
                "secondary_color": "brown",
                "style": "toon",
                "shader": "unlit",
                "texture": "flat-color",
            },
            {
                "name": "Zombie",
                "description": "A basic zombie enemy walking towards the house",
                "primary_color": "brown",
                "secondary_color": "gray",
                "style": "toon",
                "shader": "unlit",
                "texture": "flat-color",
            },
            {
                "name": "Sunflower",
                "description": "A yellow plant that produces sun currency",
                "primary_color": "yellow",
                "secondary_color": "green",
                "style": "toon",
                "shader": "unlit",
                "texture": "flat-color",
            },
        ],
        "ui_elements": [
            {
                "name": "Sun Counter",
                "description": "Displays current sun currency in top-left",
                "primary_color": "yellow",
                "secondary_color": "orange",
                "style": "toon",
                "shader": "unlit",
                "texture": "flat-color",
                "font": "cartoon",
            },
            {
                "name": "Plant Selection Bar",
                "description": "Horizontal bar showing available plants to place",
                "primary_color": "brown",
                "secondary_color": "green",
                "style": "toon",
                "shader": "unlit",
                "texture": "flat-color",
                "font": None,
            },
        ],
        "background": {
            "description": "A suburban lawn with a house on the left, divided into a grid for plant placement. Bright daylight setting."
        },
    }


def get_subway_surfers_mock():
    """Return mock scene analysis data for Subway Surfers."""
    return {
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
            {
                "name": "Train",
                "description": "A moving train obstacle on the tracks",
                "primary_color": "blue",
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
            "description": "Subway tunnel with brown stone blocks and warm lighting, tracks visible in three lanes"
        },
    }


def get_mock_for_game(game_name: str) -> dict:
    """Return appropriate mock data based on game name."""
    game_lower = game_name.lower()
    
    if "pvz" in game_lower or "plant" in game_lower or "zombie" in game_lower:
        return get_pvz_mock()
    elif "subway" in game_lower or "surf" in game_lower:
        return get_subway_surfers_mock()
    else:
        # Default generic mock
        return {
            "assets": [
                {
                    "name": "Game Object",
                    "description": "A generic game object",
                    "primary_color": "blue",
                    "secondary_color": "white",
                    "style": "toon",
                    "shader": "unlit",
                    "texture": "flat-color",
                }
            ],
            "ui_elements": [
                {
                    "name": "Score Display",
                    "description": "Shows the current score",
                    "primary_color": "white",
                    "secondary_color": "black",
                    "style": "toon",
                    "shader": "unlit",
                    "texture": "flat-color",
                    "font": "default",
                }
            ],
            "background": {
                "description": "A generic game environment"
            },
        } 