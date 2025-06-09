from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# TODO refine the BaseModels if needed.
# Caution: the field and class names will have direct impact on the quality of the model output. bc they're fed directly into the prompt.
# I recommend not changing them much.


Style = Literal["toon", "pixel-art", "cel-shaded", "realistic"]
Shader = Literal["unlit", "blinn-phong", "pbr-metallic", "pbr-specular"]
Texture = Literal["flat-color", "hand-painted", "photographic", "procedural"]
Font = Literal["sans-serif", "serif", "monospace", "handwritten", "cartoon"]


class SceneAnalysis(BaseModel):
    class Asset(BaseModel):
        name: str
        description: str  # = Field(..., max_length=120)
        style: Style | None = None
        shader: Shader | None = None
        texture: Texture | None = None

    class UIElement(BaseModel):
        name: str
        description: str
        style: Style | None = None
        shader: Shader | None = None
        texture: Texture | None = None
        font: Font | None = None

    class Background(BaseModel):
        description: str

    assets: List[Asset]
    ui_elements: List[UIElement]
    background: Background


class ActionAnalysis(BaseModel):
    class Asset(BaseModel):
        name: str
        changes_due_to_action: str

    class UIElement(BaseModel):
        name: str
        changes_due_to_action: str

    class Background(BaseModel):
        changes_due_to_action: str

    assets: List[Asset]
    ui_elements: List[UIElement]
    background: Background
