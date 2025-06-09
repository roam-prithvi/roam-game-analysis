from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

from .base_models import SceneAnalysis


def _load_str(name: str) -> str:
    return Path("prompts", f"{name}.txt").read_text()


def get_find_assets_prompt() -> str:
    """Get the prompt for finding assets in game screenshots."""
    return _load_str("find_assets")


def get_analyze_action_prompt(assets: list, ui_elements: list) -> str:
    """Get the prompt for analyzing player actions in game screenshots.

    Args:
        assets: List of Asset objects from SceneAnalysis
        ui_elements: List of UIElement objects from SceneAnalysis

    Returns:
        The formatted prompt with assets and UI elements as XML
    """
    # Convert assets to XML format
    assets_xml = "<assets>\n"
    for asset in assets:
        assets_xml += f"  <{asset.name}>{asset.description}</{asset.name}>\n"
    assets_xml += "</assets>"

    # Convert UI elements to XML format
    ui_elements_xml = "\n<ui_elements>\n"
    for ui_element in ui_elements:
        ui_elements_xml += (
            f"  <{ui_element.name}>{ui_element.description}</{ui_element.name}>\n"
        )
    ui_elements_xml += "</ui_elements>"

    # Load the template and substitute the XML content
    template = _load_str("analyze_action")
    return template.replace("{{ assets_xml }}", assets_xml).replace(
        "{{ ui_elements_xml }}", ui_elements_xml
    )


# Disregard the following code, just testing ------------------------------------------------------------

json_str = """{
  "assets": [
    {
      "name": "Player Character",
      "description": "A stylized, cartoonish human character wearing a white hoodie, a red and yellow cap, and blue pants, riding a red hoverboard.",
      "style": "toon",
      "shader": "blinn-phong",
      "texture": "hand-painted"
    },
    {
      "name": "Gold Coins",
      "description": "Floating, rotating gold coins with a star insignia that the player can collect.",
      "style": "toon",
      "shader": "unlit",
      "texture": "hand-painted"
    },
    {
      "name": "Train Cars (Blue)",
      "description": "Long, blue train cars with a rounded top that act as moving obstacles or platforms.",
      "style": "toon",
      "shader": "blinn-phong",
      "texture": "hand-painted"
    },
    {
      "name": "Train Car (Orange Front)",
      "description": "An orange and red-fronted train car that appears to be an obstacle coming towards the player.",
      "style": "toon",
      "shader": "blinn-phong",
      "texture": "hand-painted"
    },
    {
      "name": "Power-up/Mystery Box",
      "description": "A floating, golden-glowing box with a question mark, which gives the player a random temporary advantage.",
      "style": "toon",
      "shader": "unlit",
      "texture": "hand-painted"
    },
    {
      "name": "Barrier",
      "description": "A wooden barrier with yellow and black striped legs that the player must avoid.",
      "style": "toon",
      "shader": "blinn-phong",
      "texture": "hand-painted"
    },
    {
      "name": "Wall Light",
      "description": "A glowing yellow light fixture mounted on the tunnel walls.",
      "style": "toon",
      "shader": "unlit",
      "texture": "procedural"
    },
    {
      "name": "End-of-Track Bumper",
      "description": "A red and yellow striped bumper at the end of a railway track.",
      "style": "toon",
      "shader": "blinn-phong",
      "texture": "hand-painted"
    }
  ],
  "ui_elements": [
    {
      "name": "Pause Button",
      "description": "A button in the top-left corner with two vertical white bars to pause the game.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": null
    },
    {
      "name": "Score Multiplier",
      "description": "Text in the top center displaying 'x1', indicating the current score multiplier.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Score Counter",
      "description": "A numerical display next to the multiplier showing the player's current score.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Coin Counter",
      "description": "A display below the score showing a coin icon and the number of coins collected during the run.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Top Run Score",
      "description": "A box on the right side of the screen showing the player's highest score achieved.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Mission Complete Alert",
      "description": "A pop-up box in the upper center of the screen with a checkmark, indicating a mission has been completed.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Power-up Meter (Jetpack)",
      "description": "An icon and a horizontal bar in the bottom-left corner that shows the remaining duration of the jetpack power-up.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": null
    },
    {
      "name": "Power-up Icon (Hoverboard)",
      "description": "An icon in the bottom center with a number '10' and a depleting bar, indicating the active hoverboard and its duration.",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    }
  ],
  "background": {
    "description": "The background is a toon-styled, three-lane subway tunnel made of light brown, chiseled stone walls. The lighting is warm, with a distinct glow emanating from sconce-like lights mounted along the arched walls, creating a soft and ambient feel. The overall environment is clean and stylized rather than gritty or realistic. The camera is positioned behind and slightly above the player character, following them from a third-person perspective as they move forward through the endless tunnel."
  }
}
"""


if __name__ == "__main__":

    def _extract_scene_components(
        json_payload: str,
    ) -> Tuple[List[SceneAnalysis.Asset], List[SceneAnalysis.UIElement]]:
        """
        Ingest a JSON string, validate it against `SceneAnalysis`, and return
        the assets and UI elements as two separate lists.

        Args:
            json_payload: The raw JSON string describing the scene.

        Returns:
            A tuple where the first element is the list of `Asset` instances and
            the second element is the list of `UIElement` instances.
        """
        data: dict[str, Any] = json.loads(json_payload)
        scene: SceneAnalysis = SceneAnalysis.model_validate(data)
        return scene.assets, scene.ui_elements

    assets, ui_elements = _extract_scene_components(json_str)

    # # Demo output – replace with downstream logic as needed.
    # print("Assets:")
    # for asset in assets:
    #     print(f"  • {asset.name}: {asset.description}")

    # print("\nUI Elements:")
    # for ui in ui_elements:
    #     print(f"  • {ui.name}: {ui.description}")

    print(get_analyze_action_prompt(assets, ui_elements))
