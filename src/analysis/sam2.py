import asyncio
import contextlib
import io
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

import sieve
from PIL import Image

# Configure logging to suppress Sieve internal messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger("sieve").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Suppress other common noisy loggers
for logger_name in ["httpx", "httpcore", "asyncio", "PIL"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Initialize Sieve with API key
sieve.api_key = os.getenv("SIEVE_API_KEY")


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr if needed."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield


async def get_object_bbox_batch(
    images: List[sieve.File], object_name: str
) -> List[List[int] | None]:
    """Get bounding boxes for an object using YOLOv8 for multiple images concurrently."""
    yolo = sieve.function.get("sieve/yolov8")

    # Push all jobs concurrently
    jobs = []
    for image in images:
        job = yolo.push(
            file=image,
            classes=object_name,
            models="yolov8l-world",
        )
        jobs.append(job)

    # Collect all results
    bounding_boxes = []
    for i, job in enumerate(jobs):
        try:
            response = job.result()
            if not response.get("boxes"):
                print(f"Warning: No {object_name} detected in image {i}")
                bounding_boxes.append(None)
                continue

            box = response["boxes"][0]  # most confident bounding box
            bounding_box = [box["x1"], box["y1"], box["x2"], box["y2"]]
            bounding_boxes.append(bounding_box)
            print(f"✓ Found {object_name} in image {i}")
        except Exception as e:
            print(f"Warning: Error processing image {i}: {e}")
            bounding_boxes.append(None)

    return bounding_boxes


async def segment_images_batch(
    files: List[sieve.File], object_name: str
) -> List[Tuple[sieve.File, Dict[str, sieve.File]] | None]:
    """Segment objects from multiple images using text prompt concurrently."""
    sam = sieve.function.get("sieve/sam2")

    print(f"Fetching bounding boxes for '{object_name}' from {len(files)} images...")
    boxes = await get_object_bbox_batch(files, object_name)

    # Count successful detections
    successful_detections = sum(1 for box in boxes if box is not None)
    print(
        f"Successfully detected {object_name} in {successful_detections}/{len(files)} images"
    )

    # Prepare SAM prompts only for images with successful bounding boxes
    sam_jobs = []
    image_indices = []  # Track which original images we're processing

    for i, (file, box) in enumerate(zip(files, boxes)):
        if box is None:
            continue  # Skip images without detected objects

        sam_prompt = {
            "object_id": 1,
            "frame_index": 0,  # for images, this is always 0
            "box": box,
        }

        sam_job = sam.push(
            file=file,
            prompts=[sam_prompt],
            model_type="large",  # Options: "tiny", "small", "base", "large"
        )
        sam_jobs.append(sam_job)
        image_indices.append(i)

    print(
        f"Running SAM 2 segmentation on {len(sam_jobs)} images with detected objects..."
    )

    # Initialize results list with None for all images
    sam_results = [None] * len(files)

    # Collect SAM results for successful detections
    for job, original_index in zip(sam_jobs, image_indices):
        try:
            sam_out = job.result()
            sam_results[original_index] = sam_out
            print(f"✓ Segmentation completed for image {original_index}")
        except Exception as e:
            print(f"Warning: SAM segmentation failed for image {original_index}: {e}")
            sam_results[original_index] = None

    return sam_results


async def cut_objects_batch(
    image_paths: List[str],
    text_prompt: str,
    output_dir: str = "batch_cutouts",
    suppress_sieve_output: bool = True,
) -> List[str]:
    """
    Cut out objects from multiple images using segmentation and save as transparent PNGs.

    Args:
        image_paths: List of paths to input images
        text_prompt: Text description of the object to segment
        output_dir: Directory where to save the cutout images
        suppress_sieve_output: Whether to suppress Sieve internal debug messages

    Returns:
        List of output file paths for successfully processed images
    """
    try:
        # Validate all input files exist
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create Sieve File objects
        images = [sieve.File(path=path) for path in image_paths]

        print(f"Processing {len(images)} images for '{text_prompt}'...")

        # Run batch segmentation (optionally suppress Sieve output)
        if suppress_sieve_output:
            with suppress_output():
                results = await segment_images_batch(images, text_prompt)
        else:
            results = await segment_images_batch(images, text_prompt)

        output_paths = []
        successful_count = 0
        failed_count = 0

        # Process each result
        for i, (image_path, result) in enumerate(zip(image_paths, results)):
            try:
                if result is None:
                    print(
                        f"⚠ Skipping image {i+1}/{len(images)} ({os.path.basename(image_path)}): No object detected"
                    )
                    failed_count += 1
                    continue

                # Handle the tuple result: (sieve.File, Dict[str, sieve.File])
                if isinstance(result, tuple) and len(result) >= 2:
                    masks_dict = result[1]  # Second element is a dictionary of masks

                    if not isinstance(masks_dict, dict) or not masks_dict:
                        print(
                            f"⚠ Skipping image {i+1}/{len(images)} ({os.path.basename(image_path)}): No masks generated"
                        )
                        failed_count += 1
                        continue

                    # Get the first mask (assuming it's the main object mask)
                    mask_name = list(masks_dict.keys())[0]
                    mask_file = masks_dict[mask_name]

                    print(
                        f"✓ Processing image {i+1}/{len(images)} ({os.path.basename(image_path)}): Using mask {mask_name}"
                    )

                    # Download the mask file by accessing .path (suppress output if requested)
                    if suppress_sieve_output:
                        with suppress_output():
                            mask_path = mask_file.path
                    else:
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
                        mask_img = mask_img.resize(
                            original_img.size, Image.Resampling.LANCZOS
                        )

                    # Convert images to numpy arrays
                    original_array = np.array(original_img)
                    mask_array = np.array(mask_img)

                    # Create the cutout by setting alpha channel based on mask
                    cutout_array = original_array.copy()
                    cutout_array[:, :, 3] = (
                        mask_array  # Use mask values as alpha channel
                    )

                    # Convert back to PIL Image
                    cutout_img = Image.fromarray(cutout_array, "RGBA")

                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(output_dir, f"{base_name}.png")

                    # Save the result
                    cutout_img.save(output_path, "PNG")
                    output_paths.append(output_path)
                    successful_count += 1
                    print(f"✓ Saved cutout to: {output_path}")

                else:
                    print(
                        f"⚠ Skipping image {i+1}/{len(images)} ({os.path.basename(image_path)}): Unexpected segmentation result format"
                    )
                    failed_count += 1

            except Exception as e:
                print(
                    f"✗ Error processing image {i+1}/{len(images)} ({os.path.basename(image_path)}): {e}"
                )
                failed_count += 1
                continue

        # Print summary
        print(f"\n📊 Batch Processing Summary:")
        print(
            f"   ✓ Successfully processed: {successful_count}/{len(image_paths)} images"
        )
        print(f"   ⚠ Failed/Skipped: {failed_count}/{len(image_paths)} images")
        print(f"   📁 Output directory: {output_dir}")

        return output_paths

    except Exception as e:
        print(f"Batch processing error: {e}")
        raise


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
    """Main function demonstrating both single and batch processing."""
    # Get the absolute path to ensure it works regardless of where the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(script_dir)
    )  # Go up two levels from src/analysis/
    text_prompt = "gold and orange coin with a star emblem"

    # Example 1: Single image processing (original functionality)
    # print("=== Single Image Processing ===")
    # image_path = os.path.join(
    #     project_root, "data", "subway surfers", "08-06-25_at_19.33.00", "3.png"
    # )
    # output_path = "cutout_temp.png"

    # print(f"Input image: {image_path}")
    # print(f"Object to cut: {text_prompt}")
    # print(f"Output path: {output_path}")
    # print(f"File exists: {os.path.exists(image_path)}")
    # print()

    # try:
    #     await cut_object(image_path, text_prompt, output_path)
    #     print(f"✓ Single cutout saved to: {output_path}")
    # except Exception as e:
    #     print(f"✗ Single processing failed: {e}")

    # print("\n" + "=" * 50 + "\n")

    # Example 2: Batch processing multiple images
    print("=== Batch Image Processing ===")

    # Find multiple images for batch processing (you can modify this path)
    batch_images = []
    data_dir = os.path.join(
        project_root, "data", "subway surfers", "08-06-25_at_19.33.00"
    )

    # Look for specific image files 3.png through 5.png
    for i in range(3, 6):  # 3, 4, 5 for demo
        filename = f"{i}.png"
        full_path = os.path.join(data_dir, filename)
        if os.path.exists(full_path):
            batch_images.append(full_path)

    if len(batch_images) > 1:
        print(f"Processing {len(batch_images)} images in batch:")
        for i, img_path in enumerate(batch_images):
            print(f"  {i+1}. {os.path.basename(img_path)}")

        try:
            output_paths = await cut_objects_batch(
                image_paths=batch_images,
                text_prompt=text_prompt,
                output_dir="batch_cutouts",
            )

            if output_paths:
                print(f"\n🎉 Batch processing completed successfully!")
                print(f"✓ Generated {len(output_paths)} cutout(s):")
                for output_path in output_paths:
                    print(f"  - {os.path.basename(output_path)}")
            else:
                print(f"\n⚠ Batch processing completed but no cutouts were generated.")
                print(
                    f"The target object '{text_prompt}' was not found in any of the images."
                )

        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Not enough images found for batch demonstration")
        print(f"Available images: {[os.path.basename(p) for p in batch_images]}")


if __name__ == "__main__":
    asyncio.run(main())
