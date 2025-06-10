# Standard library imports
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Tuple

import google.genai as genai  # Gemini Python SDK

# Third-party imports
import numpy as np
import sieve
from PIL import Image, ImageDraw

# Separate confidence thresholds for YOLO (object detection) and SAM 2 (segmentation)
# YOLO is much noisier, so we keep its threshold relatively low.
# SAM 2 currently does not expose a comparable setting via the public
# Sieve wrapper, but we declare the constant for future use and to make
# the intent explicit.

YOLO_CONFIDENCE_THRESHOLD: float = 0.20  # used when calling YOLOv8
# SAM2_CONFIDENCE_THRESHOLD: float = 0.50  # reserved for potential SAM 2 tuning

# Gemini settings
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"  # model optimised for speed + vision
MAX_CONCURRENT_REQUESTS: int = 100  # requested by user for batch processing

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
    normalised_box: list[int | float], image_size: tuple[int, int]
) -> list[int]:
    """Convert a `[ymin, xmin, ymax, xmax]` box in 0-1000 space to pixel coords.

    The return format is `[xmin, ymin, xmax, ymax]`, matching SAM 2 expectations.
    """

    ymin, xmin, ymax, xmax = normalised_box
    height, width = image_size  # PIL images return (width, height) but we need both

    y0 = int(round((ymin / 1000) * height))
    x0 = int(round((xmin / 1000) * width))
    y1 = int(round((ymax / 1000) * height))
    x1 = int(round((xmax / 1000) * width))

    return [x0, y0, x1, y1]


def _extract_bbox_from_gemini_response(
    response_text: str, object_name_lower: str
) -> list[int] | None:
    """Return the largest bounding box that matches *object_name_lower*.

    The Gemini Vision API usually responds with a JSON payload wrapped inside
    triple-back-tick code fences, for example::

        ```json
        [
          {"label": "gold coin", "box_2d": [10, 20, 30, 40]}
        ]
        ```

    Directly feeding such a string to :pyfunc:`json.loads` fails because of the
    leading back-ticks.  This helper therefore attempts the following in order:

    1. Extract the *first* fenced code block (```json fenced block). If present, try
       ``json.loads`` on its contents.
    2. If no code fences are found, attempt ``json.loads`` on the *entire*
       response (this covers the ideal case where Gemini complied exactly).
    3. As a last resort, iterate over every substrings that *look* like a JSON
       array or object (starting with ``[`` or ``{``) and try to parse them.

    Once valid JSON is obtained, the routine searches for the *largest* bounding box
    amongst those that contain the queried object name.
    """

    print(f"[DEBUG] Raw response_text: '{response_text}'")
    print(f"[DEBUG] Looking for object: '{object_name_lower}'")

    # ------------------------------------------------------------------
    # Step 1: try to extract the payload from a ```json fenced block
    # ------------------------------------------------------------------
    code_fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    fenced_blocks = code_fence_re.findall(response_text)

    # Candidates we will attempt to parse in order of likelihood
    json_candidates: list[str] = []
    if fenced_blocks:
        json_candidates.extend(fenced_blocks)
    # Always also try the raw response as-is (covers the compliant case)
    json_candidates.append(response_text.strip())

    # ------------------------------------------------------------------
    # Step 2: brute-force parsing of each candidate
    # ------------------------------------------------------------------
    data: Any | None = None
    for idx, candidate in enumerate(json_candidates):
        try:
            data = json.loads(candidate)
            print(f"[DEBUG] ✓ Parsed candidate {idx} successfully")
            break  # Stop at the first successful parse
        except json.JSONDecodeError as exc:
            print(f"[DEBUG] Candidate {idx} JSON parse failed: {exc}")

    # As a last resort, attempt to locate a substring that starts with '[' or '{'
    if data is None:
        potential_start = min(
            (
                pos
                for pos in (response_text.find("["), response_text.find("{"))
                if pos != -1
            ),
            default=-1,
        )
        if potential_start != -1:
            substring = response_text[potential_start:]
            try:
                data = json.loads(substring)
                print("[DEBUG] ✓ Parsed fallback substring successfully")
            except json.JSONDecodeError as exc:
                print(f"[DEBUG] Fallback substring parse failed: {exc}")

    if data is None:
        print("[DEBUG] Unable to locate any valid JSON in Gemini response")
        return None

    # ------------------------------------------------------------------
    # Normalise to a list[dict] for uniform downstream processing
    # ------------------------------------------------------------------
    if isinstance(data, dict):
        items: list[dict[str, Any]] = [data]
    elif isinstance(data, list):
        items = data  # type: ignore[assignment]
    else:
        print(f"[DEBUG] Parsed JSON is neither dict nor list (type={type(data)})")
        return None

    print(f"[DEBUG] Processing {len(items)} items from parsed JSON")

    # ------------------------------------------------------------------
    # Search for the *largest* bounding box amongst those that match label
    # ------------------------------------------------------------------
    best_box: list[float] | None = None
    best_area: float = -1.0

    for i, item in enumerate(items):
        print(f"[DEBUG] Item {i}: {item}")

        # Extract box and label (support both 'box_2d' and 'bbox')
        box = item.get("box_2d") or item.get("bbox")
        label = str(item.get("label", "")).lower()

        print(f"[DEBUG] Item {i} label: '{label}', box: {box}")

        if box is None:
            continue  # Nothing to work with

        # Ensure the label matches (or is absent)
        if label and object_name_lower not in label:
            print(
                f"[DEBUG] Item {i} skipped because label '{label}' does not contain '{object_name_lower}'"
            )
            continue

        # Validate box format: must be four numeric values
        if not (
            isinstance(box, (list, tuple))
            and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)
        ):
            print(f"[DEBUG] Item {i} has an invalid box format: {box}")
            continue

        # Compute the area (remember the order is [ymin, xmin, ymax, xmax])
        ymin, xmin, ymax, xmax = (float(v) for v in box)
        width: float = max(0.0, xmax - xmin)
        height: float = max(0.0, ymax - ymin)
        area: float = width * height

        print(f"[DEBUG] Item {i} area: {area}")

        if area > best_area:
            best_area = area
            # Cast to int to satisfy the declared return type while preserving order
            best_box = [int(round(v)) for v in (ymin, xmin, ymax, xmax)]
            print(f"[DEBUG] Item {i} currently has the largest area")

    if best_box is not None:
        print(f"[DEBUG] ✓ Returning largest box with area {best_area}: {best_box}")
        return best_box

    print("[DEBUG] No matching box found in any JSON items")
    return None


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

    async def _worker(idx: int, image: sieve.File) -> list[int] | None:
        """Inner coroutine executed with concurrency control."""

        async with semaphore:

            def _run_sync() -> list[int] | None:
                path = image.path  # local cache path ensured by Sieve

                # Upload the image to Gemini Files API
                try:
                    uploaded = _genai_client.files.upload(file=path)
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
                    response = _genai_client.models.generate_content(
                        model=GEMINI_MODEL_NAME,
                        contents=[uploaded, prompt],
                    )
                except Exception as exc:
                    print(f"Warning: Gemini call failed for image {idx}: {exc}")
                    return None

                # -------------------------------------------------------------
                # Debug: Show exactly what Gemini returned for this image
                # -------------------------------------------------------------
                print(f"\n📤 [Gemini][image {idx}] raw response:\n{response.text}\n")

                normalised_box = _extract_bbox_from_gemini_response(
                    response.text, object_name_lower
                )
                print(f"Normalised box: {normalised_box}")
                if normalised_box is None:
                    return None

                # Convert to pixel coordinates using PIL Image for size
                with Image.open(path) as img:
                    pixel_box = _convert_normalised_box_to_pixels(
                        normalised_box, (img.height, img.width)
                    )
                return pixel_box

            # Run blocking Gemini/sdk I/O in executor to keep event loop responsive
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _run_sync)

    # Schedule all worker tasks
    tasks = [_worker(idx, img) for idx, img in enumerate(images)]
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
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> List[str]:
    """
    Cut out objects from multiple images using segmentation and save as transparent PNGs.

    Args:
        image_paths: List of paths to input images
        text_prompt: Text description of the object to segment
        output_dir: Directory where to save the cutout images
        confidence_threshold: Confidence threshold for YOLOv8 detection

    Returns:
        List of output file paths for successfully processed images
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


async def get_object_bbox(
    image: sieve.File,
    object_name: str,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> List[int]:
    """Get a single bounding box for *object_name* using Gemini vision."""

    # Re-use the batch helper for consistency but for a single image
    bboxes = await get_object_bbox_batch([image], object_name, confidence_threshold)
    bbox = bboxes[0]
    if bbox is None:
        raise ValueError(f"No {object_name} detected in the image")
    return bbox


async def segment_image(
    file: sieve.File,
    object_name: str,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> Tuple[sieve.File, Dict[str, sieve.File]]:
    """Segment an object from an image using text prompt."""
    sam = sieve.function.get("sieve/sam2")

    print(f"Fetching bounding box for '{object_name}'...")
    box = await get_object_bbox(file, object_name, confidence_threshold)
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
        model_type="large",  # Options: "tiny", "small", "base", "large" - larger models are more accurate but slower
        # Other available parameters to control sensitivity/quality:
        # preview_mode=False,  # If True, returns lower quality but faster results
        # normalize_coords=True,  # Whether to normalize coordinates (default: True)
        # max_edge_length=1024,  # Maximum edge length for processing (affects quality vs speed)
        # return_confidence=True,  # Whether to return confidence scores (available in some versions)
    )

    # .result() is synchronous according to Sieve docs
    sam_out = sam_job.result()
    return sam_out


async def cut_object(
    image_path: str,
    text_prompt: str,
    output_path: str,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
) -> None:
    """
    Cut out an object from an image using segmentation and save as transparent PNG.

    Args:
        image_path: Path to the input image
        text_prompt: Text description of the object to segment
        output_path: Path where to save the cutout image with transparency
        confidence_threshold: Confidence threshold for YOLOv8 detection
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Create Sieve File object
        image = sieve.File(path=image_path)

        # Run segmentation to get masks
        print(f"Segmenting '{text_prompt}' from {image_path}...")
        result = await segment_image(image, text_prompt, confidence_threshold)

        # Handle the tuple result: (sieve.File, Dict[str, sieve.File])
        if isinstance(result, tuple) and len(result) >= 2:
            masks_dict = result[1]  # Second element is a dictionary of masks

            if not isinstance(masks_dict, dict) or not masks_dict:
                raise ValueError("No masks were generated")

            # Get the first mask (assuming it's the main object mask)
            mask_name = list(masks_dict.keys())[0]
            mask_file = masks_dict[mask_name]

            print(f"Using mask: {mask_name}")

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


async def main() -> None:
    """Main function demonstrating both single and batch processing."""
    # Get the absolute path to ensure it works regardless of where the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(script_dir)
    )  # Go up two levels from src/analysis/
    text_prompt = "hoverboard"

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

        try:
            # TODO uncomment

            # output_paths = await cut_objects_batch(
            #     image_paths=batch_images,
            #     text_prompt=text_prompt,
            #     output_dir=f"batch_cutouts/{text_prompt}",
            #     confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
            # )

            # if output_paths:
            #     print(f"\n🎉 Batch processing completed successfully!")
            #     print(f"✓ Generated {len(output_paths)} cutout(s):")
            #     for output_path in output_paths:
            #         print(f"  - {os.path.basename(output_path)}")
            # else:
            #     print(f"\n⚠ Batch processing completed but no cutouts were generated.")
            #     print(
            #         f"The target object '{text_prompt}' was not found in any of the images."
            #     )

            # ------------------------------------------------------------------
            # Additionally, draw bounding boxes on the originals for comparison
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
                    print(f"\n🎉 Bounding-box overlay completed successfully!")
                    print(f"✓ Generated {len(bbox_output_paths)} annotated image(s):")
                    for p in bbox_output_paths:
                        print(f"  - {os.path.basename(p)}")
                else:
                    print(
                        f"\n⚠ Bounding-box overlay completed but no annotated images were generated."
                    )
            except Exception as exc:
                print(f"✗ Bounding-box overlay failed: {exc}")

        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Not enough images found for batch demonstration")
        print(f"Available images: {[os.path.basename(p) for p in batch_images]}")


if __name__ == "__main__":
    asyncio.run(main())
