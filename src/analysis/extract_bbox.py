import json
import re
from typing import Any


def extract_bbox_from_gemini_response(
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
