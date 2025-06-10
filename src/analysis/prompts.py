from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple, Type

from pydantic import BaseModel

from .base_models import ActionAnalysis, SceneAnalysis


def _load_str(name: str) -> str:
    return Path("prompts", f"{name}.txt").read_text()


def _json_conform_str(model: Type[BaseModel]) -> str:
    """Return an instruction string forcing the LLM to output the given model.

    The returned string starts with the sentence:
    "Respond only in JSON conforming to the following format:"
    and then appends the JSON schema of the provided *model* (pretty-printed).

    Args:
        model: Any subclass of ``pydantic.BaseModel`` that describes the
            desired output schema.

    Returns:
        A string suitable for inclusion in a prompt sent to an LLM.
    """

    schema_str: str = json.dumps(model.model_json_schema(), indent=2)
    return "Respond only in JSON conforming to the format. "  # actually not using schema_str, instructor provides it


def get_find_assets_prompt() -> str:
    """Get the prompt for finding assets in game screenshots."""
    template: str = _load_str("find_assets")
    # Encourage the model to structure its response as valid JSON.
    return f"{template}\n\n{_json_conform_str(SceneAnalysis)}"


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
    template: str = _load_str("analyze_action")
    prompt_body: str = template.replace("{{ assets_xml }}", assets_xml).replace(
        "{{ ui_elements_xml }}", ui_elements_xml
    )
    # Append structured-response instructions for the ActionAnalysis schema.
    return f"{prompt_body}\n\n{_json_conform_str(ActionAnalysis)}"


# Disregard the following code, just testing ------------------------------------------------------------

json_str = """
{
  "assets": [
    {
      "name": "Player Character",
      "description": "A character in a white hoodie and red Santa hat riding a red and green skateboard.",
      "primary_color": "white",
      "secondary_color": "red",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color"
    },
    {
      "name": "Gold Coin",
      "description": "A floating, rotating, shiny gold coin that serves as a collectible item.",
      "primary_color": "gold",
      "secondary_color": "yellow",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color"
    },
    {
      "name": "Train Car",
      "description": "A blue passenger train car that acts as a moving obstacle or a platform to run on.",
      "primary_color": "blue",
      "secondary_color": "yellow",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color"
    },
    {
      "name": "Mine Cart",
      "description": "A blue mine cart filled with a golden substance that serves as a ramp-like obstacle.",
      "primary_color": "blue",
      "secondary_color": "gold",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color"
    }
  ],
  "ui_elements": [
    {
      "name": "Pause Button",
      "description": "A blue rectangular button in the top-left corner with two white vertical bars for pausing the game.",
      "primary_color": "blue",
      "secondary_color": "white",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": null
    },
    {
      "name": "Score Counter",
      "description": "A numeric display in the top-right corner that shows the player's current score in white text.",
      "primary_color": "white",
      "secondary_color": "yellow",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Coin Counter",
      "description": "A display below the score showing the total number of coins collected, accompanied by a coin icon.",
      "primary_color": "yellow",
      "secondary_color": "white",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Mission Complete Banner",
      "description": "A white banner with a green checkmark that appears in the center to announce a completed mission.",
      "primary_color": "white",
      "secondary_color": "green",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": "cartoon"
    },
    {
      "name": "Magnet Power-up Timer",
      "description": "An icon of a magnet with a yellow progress bar in the bottom-left, indicating the power-up's remaining time.",
      "primary_color": "blue",
      "secondary_color": "yellow",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": null
    },
    {
      "name": "Power-up Timer",
      "description": "An icon of a shoe with wings with an orange progress bar in the bottom-right, indicating a power-up's remaining time.",
      "primary_color": "blue",
      "secondary_color": "orange",
      "style": "toon",
      "shader": "unlit",
      "texture": "flat-color",
      "font": null
    }
  ],
  "background": {
    "description": "The setting is a large, arched subway tunnel made of brown stone blocks, rendered in a clean, cartoon style. The scene is illuminated by warm light from lanterns mounted on the tunnel walls, which casts a bright glow on the central tracks. The camera maintains a third-person perspective, positioned directly behind and slightly above the player character to follow the action."
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
