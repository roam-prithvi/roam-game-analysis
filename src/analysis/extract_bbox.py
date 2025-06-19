import json
import re
from typing import Any, Dict, List


def extract_bbox_from_gemini_response(
    response_text: str, object_name_lower: str
) -> List[int] | None:
    """Return the largest bounding box from the Gemini response.

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

    Once valid JSON is obtained, the routine returns the *largest* bounding box
    from all available boxes, assuming any returned box is for the requested object.
    """

    # print(f"[DEBUG] Raw response_text: '{response_text}'")
    # print(f"[DEBUG] Looking for object: '{object_name_lower}'")

    # ------------------------------------------------------------------
    # Step 1: try to extract the payload from a ```json fenced block
    # ------------------------------------------------------------------
    code_fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    fenced_blocks = code_fence_re.findall(response_text)

    # Candidates we will attempt to parse in order of likelihood
    json_candidates: List[str] = []
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
            # print(f"[DEBUG] ✓ Parsed candidate {idx} successfully")
            break  # Stop at the first successful parse
        except json.JSONDecodeError as exc:
            # print(f"[DEBUG] Candidate {idx} JSON parse failed: {exc}")
            pass

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
                # print("[DEBUG] ✓ Parsed fallback substring successfully")
            except json.JSONDecodeError as exc:
                pass
                # print(f"[DEBUG] Fallback substring parse failed: {exc}")

    if data is None:
        # print("[DEBUG] Unable to locate any valid JSON in Gemini response")
        return None

    # ------------------------------------------------------------------
    # Normalise to a list[dict] for uniform downstream processing
    # ------------------------------------------------------------------
    if isinstance(data, dict):
        items: List[Dict[str, Any]] = [data]
    elif isinstance(data, list):
        items = data  # type: ignore[assignment]
    else:
        # print(f"[DEBUG] Parsed JSON is neither dict nor list (type={type(data)})")
        return None

    # print(f"[DEBUG] Processing {len(items)} items from parsed JSON")

    # ------------------------------------------------------------------
    # Search for the *largest* bounding box (assume all boxes are valid)
    # ------------------------------------------------------------------
    best_box: List[float] | None = None
    best_area: float = -1.0

    for i, item in enumerate(items):
        # print(f"[DEBUG] Item {i}: {item}")

        # Extract box (support both 'box_2d' and 'bbox')
        box = item.get("box_2d") or item.get("bbox")

        # print(f"[DEBUG] Item {i} box: {box}")

        if box is None:
            continue  # Nothing to work with

        # Validate box format: must be four numeric values
        if not (
            isinstance(box, (list, tuple))
            and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)
        ):
            # print(f"[DEBUG] Item {i} has an invalid box format: {box}")
            continue

        # Compute the area (remember the order is [ymin, xmin, ymax, xmax])
        ymin, xmin, ymax, xmax = (float(v) for v in box)
        width: float = max(0.0, xmax - xmin)
        height: float = max(0.0, ymax - ymin)
        area: float = width * height

        # print(f"[DEBUG] Item {i} area: {area}")

        if area > best_area:
            best_area = area
            # Cast to int to satisfy the declared return type while preserving order
            best_box = [int(round(v)) for v in (ymin, xmin, ymax, xmax)]
            # print(f"[DEBUG] Item {i} currently has the largest area")

    if best_box is not None:
        # print(f"[DEBUG] ✓ Returning largest box with area {best_area}: {best_box}")
        return best_box

    # print("[DEBUG] No valid bounding box found in any JSON items")
    return None
