import asyncio
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import sieve
from PIL import Image

# Initialize Sieve with API key
sieve.api_key = os.getenv("SIEVE_API_KEY")


async def get_object_bbox(image: sieve.File, object_name: str) -> List[int]:
    """Get bounding box for an object using YOLOv8."""
    yolo = sieve.function.get("sieve/yolov8")

    # Use push for async - returns SieveFuture object
    job = yolo.push(
        file=image,
        classes=object_name,
        models="yolov8l-world",
    )

    # .result() is synchronous according to Sieve docs
    response = job.result()

    if not response.get("boxes"):
        raise ValueError(f"No {object_name} detected in the image")

    box = response["boxes"][0]  # most confident bounding box
    bounding_box = [box["x1"], box["y1"], box["x2"], box["y2"]]

    return bounding_box


async def segment_image(
    file: sieve.File, object_name: str
) -> Tuple[sieve.File, Dict[str, sieve.File]]:
    """Segment an object from an image using text prompt."""
    sam = sieve.function.get("sieve/sam2")

    print(f"Fetching bounding box for '{object_name}'...")
    box = await get_object_bbox(file, object_name)
    print(f"Found bounding box: {box}")

    sam_prompt = {
        "object_id": 1,
        "frame_index": 0,  # for images, this is always 0
        "box": box,
    }

    print("Running SAM 2 segmentation...")
    # Use push for async - returns SieveFuture object
    sam_job = sam.push(
        file=file,
        prompts=[sam_prompt],
        model_type="large",  # Changed from "tiny" - Options: "tiny", "small", "base", "large" - larger models are more accurate but slower
        # Other available parameters to control sensitivity/quality:
        # preview_mode=False,  # If True, returns lower quality but faster results
        # normalize_coords=True,  # Whether to normalize coordinates (default: True)
        # max_edge_length=1024,  # Maximum edge length for processing (affects quality vs speed)
        # return_confidence=True,  # Whether to return confidence scores (available in some versions)
    )

    # .result() is synchronous according to Sieve docs
    sam_out = sam_job.result()
    return sam_out


async def cut_object(image_path: str, text_prompt: str, output_path: str) -> None:
    """
    Cut out an object from an image using segmentation and save as transparent PNG.

    Args:
        image_path: Path to the input image
        text_prompt: Text description of the object to segment
        output_path: Path where to save the cutout image with transparency
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Create Sieve File object
        image = sieve.File(path=image_path)

        # Run segmentation to get masks
        print(f"Segmenting '{text_prompt}' from {image_path}...")
        result = await segment_image(image, text_prompt)

        # Handle the tuple result: (sieve.File, Dict[str, sieve.File])
        if isinstance(result, tuple) and len(result) >= 2:
            masks_dict = result[1]  # Second element is a dictionary of masks

            if not isinstance(masks_dict, dict) or not masks_dict:
                raise ValueError("No masks were generated")

            # Get the first mask (assuming it's the main object mask)
            mask_name = list(masks_dict.keys())[0]
            mask_file = masks_dict[mask_name]

            print(f"Using mask: {mask_name}")

            # Download the mask file by accessing .path (this triggers download)
            mask_path = mask_file.path

            # Load the original image
            original_img = Image.open(image_path)

            # Convert to RGBA if not already
            if original_img.mode != "RGBA":
                original_img = original_img.convert("RGBA")

            # Load the mask using the downloaded path
            mask_img = Image.open(mask_path)

            # Convert mask to grayscale if it isn't already
            if mask_img.mode != "L":
                mask_img = mask_img.convert("L")

            # Resize mask to match original image size if needed
            if mask_img.size != original_img.size:
                mask_img = mask_img.resize(original_img.size, Image.Resampling.LANCZOS)

            # Convert images to numpy arrays
            original_array = np.array(original_img)
            mask_array = np.array(mask_img)

            # Create the cutout by setting alpha channel based on mask
            # Where mask is black (0), set alpha to 0 (transparent)
            # Where mask is white (255), set alpha to 255 (opaque)
            cutout_array = original_array.copy()
            cutout_array[:, :, 3] = (
                mask_array  # Directly use mask values as alpha channel
            )

            # Convert back to PIL Image
            cutout_img = Image.fromarray(cutout_array, "RGBA")

            # Save the result
            cutout_img.save(output_path, "PNG")
            print(f"Object cutout saved to: {output_path}")

        else:
            raise ValueError("Unexpected segmentation result format")

    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


async def main() -> None:
    """Main function demonstrating the cut_object function."""
    # Get the absolute path to ensure it works regardless of where the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(script_dir)
    )  # Go up two levels from src/analysis/

    # Example usage of cut_object function
    image_path = os.path.join(
        project_root, "data", "subway surfers", "08-06-25_at_19.33.00", "3.png"
    )
    # text_prompt = "cartoon cop in blue uniform seen from behind"  # Update this to your desired object
    text_prompt = (
        "gold and orange coin with a star emblem"  # Update this to your desired object
    )
    output_path = "cutout_temp.png"  # Where to save the transparent cutout

    print("=== Demonstrating cut_object function ===")
    print(f"Input image: {image_path}")
    print(f"Object to cut: {text_prompt}")
    print(f"Output path: {output_path}")
    print(f"File exists: {os.path.exists(image_path)}")
    print()

    try:
        # Run the cut_object function
        await cut_object(image_path, text_prompt, output_path)

        print("\n=== Success! ===")
        print(f"Transparent cutout saved to: {output_path}")
        print(
            "The cutout contains only the specified object with transparent background."
        )

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
