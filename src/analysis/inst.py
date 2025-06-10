import asyncio
from pathlib import Path
from typing import List

import google.generativeai as genai
import instructor
from PIL import Image as PILImage
from pydantic import BaseModel


class User(BaseModel):
    """Simple pydantic model used for the demo call."""

    character_description: str
    character_color: str
    character_style: str
    character_shader: str
    character_texture: str


def _find_first_two_time_images(base_dir: Path = Path("data")) -> List[Path]:
    """Return the first two ``*_time.png`` images under *base_dir*.

    The function walks ``base_dir/**/frames`` sub-directories and returns the
    first two matches in lexicographic order.  This mirrors the convention used
    elsewhere in the codebase for exported frame images.
    """

    images: List[Path] = sorted(base_dir.glob("**/frames/*_time.png"))
    if len(images) < 2:
        raise FileNotFoundError(
            f"Expected at least two *_time.png images under {base_dir.absolute()}"
        )
    return images[50:52]


async def extract_user() -> User:  # type: ignore[return-value]
    """Example Gemini call with images attached to the prompt."""

    # Discover frame images to send to the model.
    image_paths: List[Path] = _find_first_two_time_images()

    # Open images using Pillow, which the Gemini SDK accepts natively.
    image_blobs = [PILImage.open(p) for p in image_paths]

    # Build the prompt content: textual instruction followed by images.
    prompt_parts: List[object] = [
        (
            "Identify the main playable character depicted in these two frames. "
            "Return a JSON object with the following keys: "
            "character_description, character_color, character_style, "
            "character_shader, character_texture."
        ),
        *image_blobs,
    ]

    # Initialise the Gemini client patched by Instructor.
    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="models/gemini-2.5-pro-preview-06-05",
        ),
        use_async=True,
    )

    # Send the request – the response is automatically parsed into ``User``.
    return await client.chat.completions.create(  # type: ignore[return-value]
        messages=[{"role": "user", "content": prompt_parts}],
        response_model=User,
    )


if __name__ == "__main__":
    # Run async function and print the structured result.
    user: User = asyncio.run(extract_user())
    print(user.model_dump_json(indent=2))  # noqa: T201 – simple demo output
