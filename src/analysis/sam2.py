# Standard library imports
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

import google.genai as genai  # Gemini Python SDK

# Third-party imports
import numpy as np
import sieve
from PIL import Image, ImageDraw

from src.analysis.extract_bbox import extract_bbox_from_gemini_response

# Separate confidence thresholds for YOLO (object detection) and SAM 2 (segmentation)
# YOLO is much noisier, so we keep its threshold relatively low.
# SAM 2 currently does not expose a comparable setting via the public
# Sieve wrapper, but we declare the constant for future use and to make
# the intent explicit.

YOLO_CONFIDENCE_THRESHOLD: float = 0.45  # used when calling YOLOv8
# SAM2_CONFIDENCE_THRESHOLD: float = 0.50  # reserved for potential SAM 2 tuning
YOLO_IOU_THRESHOLD        = 0.6 
# Gemini settings
GEMINI_MODEL_NAME: str = "gemini-2.5-pro-preview-06-05"
MAX_CONCURRENT_REQUESTS: int = (
    50  # Increased for better throughput - Gemini can handle high concurrency
)

# Initialise Gemini client once (thread-safe according to SDK docs)
_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Sieve with API key
sieve.api_key = os.getenv("SIEVE_API_KEY")

# Florence-2 task prompt used for guided object detection
FLORENCE_TASK_PROMPT: str = (
    "<CAPTION_TO_PHRASE_GROUNDING>"  # detects only objects named in `text_input`
)

# ---------------------------------------------------------------------------
# Gemini helper utilities
# ---------------------------------------------------------------------------


def _convert_normalised_box_to_pixels(
    normalised_box: List[int | float], image_size: Tuple[int, int]
) -> List[int]:
    """Convert a `[ymin, xmin, ymax, xmax]` box in 0-1000 space to pixel coords.

    The return format is `[xmin, ymin, xmax, ymax]`, matching SAM 2 expectations.
    """

    ymin, xmin, ymax, xmax = normalised_box
    height, width = image_size  # PIL images return (width, height) but we need both

    y0: int = int(round((ymin / 1000) * height))
    x0: int = int(round((xmin / 1000) * width))
    y1: int = int(round((ymax / 1000) * height))
    x1: int = int(round((xmax / 1000) * width))

    return [x0, y0, x1, y1]


async def get_object_bbox_batch(
    images: List[sieve.File],
    object_name: str,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> List[List[int] | None]:
    """Get bounding boxes using Gemini's vision capabilities (batch).

    Parameters
    ----------
    images
        List of :class:`sieve.File` instances to process.
    object_name
        The label to detect within each image.
    confidence_threshold
        **Ignored** but kept for backward compatibility with the previous API.

    Returns
    -------
    list[list[int] | None]
        Pixel-space bounding boxes in `[xmin, ymin, xmax, ymax]` order or
        ``None`` if the object was not detected.
    """

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    object_name_lower = object_name.lower()

    print(
        f"🚀 Starting concurrent Gemini bbox detection for {len(images)} images (max {MAX_CONCURRENT_REQUESTS} concurrent)"
    )

    async def _worker(idx: int, image: sieve.File) -> list[int] | None:
        """Inner coroutine executed with concurrency control."""

        async with semaphore:
            print(f"📤 [Image {idx}] Starting Gemini API call...")

            def _run_sync() -> list[int] | None:
                path = image.path  # local cache path ensured by Sieve

                # Upload the image to Gemini Files API
                try:
                    uploaded = _genai_client.files.upload(file=path)
                    print(f"⬆️  [Image {idx}] File uploaded to Gemini")
                except Exception as exc:  # pragma: no cover – network/SDK errors
                    print(f"Warning: Upload failed for image {idx}: {exc}")
                    return None

                prompt = (
                    f"Detect the most prominent {object_name} object in the image. "
                    "Most prominent means: standalone and not overlapping with a similar object (top priority)"
                    "and out of standalone objects, choose the largest one (lower priority)"
                    "(if there's overlap with a distinctly looking object and that distinctly looking"
                    f"object is behind the {object_name}, that is OK bc it's still prominent)"
                    f"If there are multiple {object_name} objects, make sure your bounding box is around the most prominent/standalone one"
                    f"Respond **only** with a JSON dict which is either empty (if there is no compellingly prominent {object_name} in the image) or the JSON dict should have two keys: "
                    "'label' and 'box_2d' (format [ymin, xmin, ymax, xmax] "
                    "normalized 0-1000). Return nothing else. DO NOT return a list even if there are multiple objects."
                    "DO NOT return a small object - one where both width or height are less than 5% of the image size."
                    "Choose the most prominent one. Make absolutely sure the box you output is around the image, not intersecting with it."
                    "If you're uncertain, prefer to return an empty dict, avoid false positives."
                )

                try:
                    print(f"🤖 [Image {idx}] Calling Gemini generate_content...")
                    response = _genai_client.models.generate_content(
                        model=GEMINI_MODEL_NAME,
                        contents=[uploaded, prompt],
                    )
                    print(f"✅ [Image {idx}] Gemini response received")
                except Exception as exc:
                    print(f"Warning: Gemini call failed for image {idx}: {exc}")
                    return None

                # Debug: Show exactly what Gemini returned for this image
                # print(f"\n📤 [Gemini][image {idx}] raw response:\n{response.text}\n")

                normalised_box = extract_bbox_from_gemini_response(
                    response.text, object_name_lower
                )
                print(f"📦 [Image {idx}] Normalized box: {normalised_box}")
                if normalised_box is None:
                    return None

                # Convert to pixel coordinates using PIL Image for size
                with Image.open(path) as img:
                    pixel_box = _convert_normalised_box_to_pixels(
                        normalised_box, (img.height, img.width)
                    )
                print(f"🎯 [Image {idx}] Pixel box: {pixel_box}")
                return pixel_box

            # Run blocking Gemini/sdk I/O in executor to keep event loop responsive
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _run_sync)

    # Schedule all worker tasks
    tasks = [_worker(idx, img) for idx, img in enumerate(images)]
    print(f"⏳ Awaiting {len(tasks)} concurrent Gemini API calls...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    bounding_boxes: list[list[int] | None] = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"Warning: Error processing image {idx}: {res}")
            bounding_boxes.append(None)
        elif res is None:
            print(f"Warning: No {object_name} detected in image {idx}")
            bounding_boxes.append(None)
        else:
            print(f"✓ Found {object_name} in image {idx}")
            bounding_boxes.append(res)

    print(
        f"🏁 Completed bbox detection: {sum(1 for b in bounding_boxes if b is not None)}/{len(bounding_boxes)} successful"
    )
    return bounding_boxes


async def segment_images_batch(
    files: List[sieve.File],
    object_name: str,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> List[Tuple[sieve.File, Dict[str, sieve.File]] | None]:
    """Segment objects from multiple images using text prompt concurrently."""
    sam = sieve.function.get("sieve/sam2")

    print(f"Fetching bounding boxes for '{object_name}' from {len(files)} images...")
    boxes = await get_object_bbox_batch(files, object_name, confidence_threshold)

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

    # Collect SAM results concurrently instead of sequentially
    async def _get_sam_result(job: Any, original_index: int) -> Tuple[int, Any]:
        """Get SAM result for a single job."""
        try:
            # Run the blocking job.result() in an executor to keep it async
            loop = asyncio.get_running_loop()
            sam_out = await loop.run_in_executor(None, job.result)
            print(f"✓ Segmentation completed for image {original_index}")
            return original_index, sam_out
        except Exception as e:
            print(f"Warning: SAM segmentation failed for image {original_index}: {e}")
            return original_index, None

    # Process all SAM jobs concurrently
    if sam_jobs:
        job_tasks = [
            _get_sam_result(job, original_index)
            for job, original_index in zip(sam_jobs, image_indices)
        ]
        job_results = await asyncio.gather(*job_tasks, return_exceptions=True)

        # Populate results array
        for result in job_results:
            if isinstance(result, Exception):
                print(f"Warning: Error in SAM job: {result}")
                continue
            original_index, sam_out = result
            sam_results[original_index] = sam_out

    return sam_results


async def cut_objects_batch(
    image_paths: List[str],
    text_prompt: str,
    output_dir: str = "batch_cutouts",
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> Tuple[List[str], List[str]]:
    """
    Cut out objects from multiple images using segmentation and save as transparent PNGs.

    Args:
        image_paths: List of paths to input images
        text_prompt: Text description of the object to segment
        output_dir: Directory where to save the cutout images
        confidence_threshold: Confidence threshold for YOLOv8 detection

    Returns:
        Tuple of (output_paths, corresponding_original_paths) for successfully processed images
    """
    # -------------------------------------------------------------
    # Debug: Show which prompt this batch is using
    # -------------------------------------------------------------
    print(
        f"\n🔍 [cut_objects_batch] Text prompt: '{text_prompt}' (images: {len(image_paths)})"
    )

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

        # Run batch segmentation
        results = await segment_images_batch(images, text_prompt, confidence_threshold)

        output_paths = []
        corresponding_original_paths = []
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

                    # Download the mask file by accessing .path
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
                    corresponding_original_paths.append(image_path)
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

        return output_paths, corresponding_original_paths

    except Exception as e:
        print(f"Batch processing error: {e}")
        raise


# ---------------------------------------------------------------------------
# Bounding-box visualization utilities
# ---------------------------------------------------------------------------


async def draw_bboxes_batch(
    image_paths: List[str],
    text_prompt: str,
    output_dir: str = "batch_bboxes",
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> List[str]:
    """Overlay bounding boxes for *text_prompt* on *image_paths* and save results.

    Parameters
    ----------
    image_paths
        List of file paths pointing to the input images.
    text_prompt
        The object description to detect (e.g. "gold coin").
    output_dir
        Base directory where annotated images will be written. A sub-folder
        named after *text_prompt* (spaces preserved) is created within it to
        mimic the segmentation output layout.
    confidence_threshold
        Currently ignored, preserved for API symmetry with segmentation helpers.

    Returns
    -------
    list[str]
        File paths of images that were successfully annotated and saved.
    """

    # -------------------------------------------------------------
    # Debug: show prompt and number of images
    # -------------------------------------------------------------
    print(
        f"\n🔲 [draw_bboxes_batch] Text prompt: '{text_prompt}' (images: {len(image_paths)})"
    )

    # Validate paths and create output structure
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

    prompt_dir: str = os.path.join(output_dir, text_prompt)
    os.makedirs(prompt_dir, exist_ok=True)

    # Prepare Sieve files and fetch bboxes in one go
    images: list[sieve.File] = [sieve.File(path=p) for p in image_paths]
    print("Fetching bounding boxes …")
    bboxes: list[list[int] | None] = await get_object_bbox_batch(
        images, text_prompt, confidence_threshold
    )

    annotated_paths: list[str] = []
    success: int = 0
    failure: int = 0

    for idx, (img_path, bbox) in enumerate(zip(image_paths, bboxes)):
        base_name: str = os.path.splitext(os.path.basename(img_path))[0]
        out_path: str = os.path.join(prompt_dir, f"{base_name}.png")

        if bbox is None:
            print(
                f"⚠ Skipping image {idx+1}/{len(image_paths)} ({os.path.basename(img_path)}): No object detected"
            )
            failure += 1
            continue

        try:
            # Open original image
            with Image.open(img_path) as pil_img:
                # Always work in RGBA to preserve alpha if present
                if pil_img.mode != "RGBA":
                    pil_img = pil_img.convert("RGBA")

                draw = ImageDraw.Draw(pil_img)

                # bbox order is [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = bbox

                # Choose a contrasting color (red) with full opacity
                outline_color: tuple[int, int, int, int] = (255, 0, 0, 255)

                # Dynamic line width relative to image size (min 2 px)
                line_width: int = max(2, min(pil_img.size) // 200)

                draw.rectangle(
                    [xmin, ymin, xmax, ymax],
                    outline=outline_color,
                    width=line_width,
                )

                # Optionally, draw label text above the box
                # Commented out for clarity, uncomment if desired
                # font = ImageFont.load_default()
                # text = text_prompt
                # text_size = font.getsize(text)
                # text_bg = (0, 0, 0, 160)
                # draw.rectangle(
                #     [
                #         xmin,
                #         max(0, ymin - text_size[1] - 4),
                #         xmin + text_size[0] + 4,
                #         ymin,
                #     ],
                #     fill=text_bg,
                # )
                # draw.text((xmin + 2, max(0, ymin - text_size[1] - 2)), text, fill="white", font=font)

                # Save annotated image
                pil_img.save(out_path)

            annotated_paths.append(out_path)
            success += 1
            print(f"✓ Saved annotated image {idx+1}/{len(image_paths)} to: {out_path}")

        except Exception as exc:
            print(
                f"✗ Error processing image {idx+1}/{len(image_paths)} ({os.path.basename(img_path)}): {exc}"
            )
            failure += 1

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n📊 Bounding-box Overlay Summary:")
    print(f"   ✓ Successfully annotated: {success}/{len(image_paths)} images")
    print(f"   ⚠ Failed/Skipped: {failure}/{len(image_paths)} images")
    print(f"   📁 Output directory: {prompt_dir}")

    return annotated_paths


async def postfilter_batch(
    output_paths: List[str],
    original_paths: List[str],
    object_name: str,
) -> List[str]:
    """
    Concurrently filters images using Gemini, deleting those that don't match the object.

    Args:
        output_paths: List of paths to the cutout images to be filtered.
        original_paths: List of paths to the original images (for comparison).
        object_name: The name of the object to verify in the images.

    Returns:
        A list of paths for the images that were kept.
    """
    if len(output_paths) != len(original_paths):
        raise ValueError("output_paths and original_paths must have the same length")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    object_name_lower = object_name.lower()

    print(
        f"🔬 Starting concurrent post-filtering for {len(output_paths)} images (max {MAX_CONCURRENT_REQUESTS} concurrent)"
    )

    async def _worker(idx: int, cutout_path: str, original_path: str) -> str | None:
        """Inner coroutine to verify and potentially delete an image."""
        async with semaphore:
            print(f"🔍 [Filter {idx}] Starting post-filter analysis...")

            def _run_sync() -> str | None:
                if not os.path.exists(cutout_path):
                    print(
                        f"Warning: Cutout file not found for post-filtering, skipping: {cutout_path}"
                    )
                    return None

                if not os.path.exists(original_path):
                    print(
                        f"Warning: Original file not found for post-filtering, skipping: {original_path}"
                    )
                    return None

                try:
                    print(f"⬆️  [Filter {idx}] Uploading images to Gemini...")
                    uploaded_cutout = _genai_client.files.upload(file=cutout_path)
                    uploaded_original = _genai_client.files.upload(file=original_path)
                    print(f"✅ [Filter {idx}] Both images uploaded")
                except Exception as exc:
                    print(
                        f"Warning: Upload failed for image {idx} ({cutout_path}): {exc}"
                    )
                    return cutout_path  # Keep the file if upload fails

                prompt = (
                    "You are postprocessing the results of a segmentation model which cuts out objects from images but is sometimes imprecise, and we need a clear result."
                    f"I will show you two images: first is the original image, second is a cutout from that image. "
                    f"Does the cutout image clearly show a {object_name_lower}? "
                    f"The cutout should be a clean extraction of a {object_name_lower} with minimal artifacts. "
                    f"If there are cutouts in the {object_name_lower} (appearing as black holes or transparent areas): "
                    f"if the object is still clearly recognizable as a {object_name_lower} then answer YES, otherwise NO. "
                    f"If there are ANY OTHER OBJECTS that partially obscure the {object_name_lower}, answer NO. "
                    f"If the cutout clearly does not correspond to a {object_name_lower} (sometimes the segmentation model will miss it completely), answer NO. "
                    "Answer with JSON dict containing 'reasoning': (string of your reasoning) and 'decision': (Literal, either 'YES' or 'NO')."
                )

                try:
                    print(
                        f"🤖 [Filter {idx}] Calling Gemini for post-filter analysis..."
                    )
                    response = _genai_client.models.generate_content(
                        model=GEMINI_MODEL_NAME,
                        contents=[uploaded_original, uploaded_cutout, prompt],
                    )
                    print(f"✅ [Filter {idx}] Gemini response received")
                    response_text = response.text.strip()
                    response_json = json.loads(response_text)
                    decision = response_json["decision"].strip().upper()
                    reasoning = response_json["reasoning"].strip()
                    print(
                        f"🔎 Post-filter [image {idx}]: Gemini says '{decision}' for {os.path.basename(cutout_path)}: {reasoning}"
                    )

                    if decision != "YES":
                        print(
                            f"✗ Deleting image {idx} as it does not seem to be a good {object_name_lower} cutout: {cutout_path}"
                        )
                        # TODO
                        # os.remove(cutout_path)
                        return None
                    else:
                        return cutout_path

                except Exception as exc:
                    print(
                        f"Warning: Gemini call failed for post-filtering image {idx} ({cutout_path}): {exc}"
                    )
                    return cutout_path  # Keep the file on error

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _run_sync)

    tasks = [
        _worker(idx, cutout_path, original_path)
        for idx, (cutout_path, original_path) in enumerate(
            zip(output_paths, original_paths)
        )
    ]
    print(f"⏳ Awaiting {len(tasks)} concurrent post-filter analysis tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    kept_paths: list[str] = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"Warning: Error post-filtering image {idx}: {res}")
            # The original path is still in `output_paths`. We should probably keep it if an error occurs.
            kept_paths.append(output_paths[idx])
        elif res is not None:
            kept_paths.append(res)

    print("\n✅ Post-filtering complete.")
    print(f"   Kept {len(kept_paths)}/{len(output_paths)} images.")

    return kept_paths


async def main() -> None:
    """Main function demonstrating batch processing."""
    parser = argparse.ArgumentParser(
        description="Run segmentation or bounding box drawing on a batch of images."
    )
    parser.add_argument(
        "-s",
        "--segment",
        action="store_true",
        help="Run segmentation (cut out objects).",
    )
    parser.add_argument(
        "-b",
        "--bboxes",
        action="store_true",
        help="Run bounding box drawing.",
    )
    args = parser.parse_args()

    if not args.segment and not args.bboxes:
        parser.print_help()
        return

    # Get the absolute path to ensure it works regardless of where the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(script_dir)
    )  # Go up two levels from src/analysis/
    text_prompt = '"Gold Coin"'

    # Example 2: Batch processing multiple images
    print("=== Batch Image Processing ===")

    # Find multiple images for batch processing (you can modify this path)
    batch_images = []
    data_dir = os.path.join(
        project_root, "data", "subway surfers", "08-06-25_at_19.33.00", "frames"
    )

    # Filter for images ending with _time.png
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith("_time.png"):
                batch_images.append(os.path.join(data_dir, filename))
        batch_images.sort()  # Sort for consistent ordering

    if len(batch_images) > 1:
        print(f"Processing {len(batch_images)} images in batch:")
        for i, img_path in enumerate(batch_images):
            print(f"  {i+1}. {os.path.basename(img_path)}")

        if args.segment:
            print("\n=== Cutting Objects (Segmentation) ===")
            try:
                output_paths, corresponding_original_paths = await cut_objects_batch(
                    image_paths=batch_images,
                    text_prompt=text_prompt,
                    output_dir=f"batch_cutouts/{text_prompt}",
                    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                )

                if output_paths:
                    print("\n🔬 Post-filtering cutouts...")
                    filtered_paths = await postfilter_batch(
                        output_paths, corresponding_original_paths, text_prompt
                    )

                    print("\n🎉 Batch processing completed successfully!")
                    print(
                        f"✓ Generated {len(filtered_paths)} final cutout(s) after filtering:"
                    )
                    for output_path in filtered_paths:
                        print(f"  - {os.path.basename(output_path)}")
                else:
                    print(
                        "\n⚠ Batch processing completed but no cutouts were generated."
                    )
                    print(
                        f"The target object '{text_prompt}' was not found in any of the images."
                    )
            except Exception as e:
                print(f"✗ Segmentation processing failed: {e}")
                import traceback

                traceback.print_exc()

        if args.bboxes:
            # ------------------------------------------------------------------
            # Draw bounding boxes on the originals for comparison
            # ------------------------------------------------------------------
            print("\n=== Drawing Bounding Boxes on Originals ===")
            try:
                bbox_output_paths = await draw_bboxes_batch(
                    image_paths=batch_images,
                    text_prompt=text_prompt,
                    output_dir="batch_bboxes",
                    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                )

                if bbox_output_paths:
                    print("\n🎉 Bounding-box overlay completed successfully!")
                    print(f"✓ Generated {len(bbox_output_paths)} annotated image(s):")
                    for p in bbox_output_paths:
                        print(f"  - {os.path.basename(p)}")
                else:
                    print(
                        "\n⚠ Bounding-box overlay completed but no annotated images were generated."
                    )
            except Exception as exc:
                print(f"✗ Bounding-box overlay failed: {exc}")

    else:
        print("Not enough images found for batch demonstration")
        print(f"Available images: {[os.path.basename(p) for p in batch_images]}")


if __name__ == "__main__":
    asyncio.run(main())
