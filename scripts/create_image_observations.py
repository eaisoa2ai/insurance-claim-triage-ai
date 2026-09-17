"""
Inspect every image in data/images/ with gpt-4o and write structured visual
observations to data/image_observations.json.

This script produces raw visual evidence only — no insurance decisions,
no coverage judgements, no risk scores. Downstream agents consume this file.

Usage:
    uv run python scripts/create_image_observations.py

Requires:
    OPENAI_API_KEY in .env or environment
"""

# load_dotenv() must run before importing openai so the API key is present
# when the openai module initialises its default client.
from dotenv import load_dotenv
load_dotenv()

import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import openai
from pydantic import BaseModel, Field


# ── Output schema ─────────────────────────────────────────────────────────────

class ImageObservation(BaseModel):
    image_id: str = Field(description="Filename without extension, e.g. '0005'")
    filename: str = Field(description="Filename with extension, e.g. '0005.jpg'")
    visible_damage: list[str] = Field(
        description="Damage types observed, e.g. ['dent', 'scrape', 'cracked_glass']"
    )
    damaged_parts: list[str] = Field(
        description="Car parts with visible damage, e.g. ['rear_bumper', 'trunk_lid']"
    )
    severity_hint: str = Field(
        description="Estimated visual severity: low | medium | high | unclear"
    )
    confidence: float = Field(
        description="Confidence in this assessment, 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    uncertainty: str = Field(
        description="What is unclear or hard to assess from this image alone"
    )
    short_visual_summary: str = Field(
        description="One or two sentences describing only what is visually present"
    )


class ObservationsFile(BaseModel):
    generated_at: str
    model: str
    total_images: int
    observations: list[ImageObservation]


# ── Vision prompt ──────────────────────────────────────────────────────────────

VISION_PROMPT = """\
You are a vehicle damage documentation assistant. Your only job is to describe \
what you can see in this image.

Do NOT make insurance decisions.
Do NOT assess liability or fault.
Do NOT estimate repair costs.
Do NOT determine whether a claim is valid or fraudulent.

Respond with a JSON object containing exactly these fields:

{
  "visible_damage": ["list", "of", "damage", "types", "observed"],
  "damaged_parts": ["list", "of", "car", "parts", "with", "visible", "damage"],
  "severity_hint": "low | medium | high | unclear",
  "confidence": 0.0,
  "uncertainty": "describe what is unclear or hard to assess from this image",
  "short_visual_summary": "One or two sentences describing only what is visually present."
}

Guidelines:
- visible_damage values: dent, scrape, cracked_glass, shattered_glass, crumple,
  paint_transfer, burn_damage, structural_deformation, no_visible_damage, debris_impact
- damaged_parts values: front_bumper, rear_bumper, hood, trunk, roof, driver_door,
  passenger_door, rear_door, front_quarter_panel, rear_quarter_panel, windshield,
  rear_windshield, side_window, headlight, taillight, wheel, undercarriage
- severity_hint:
    low    = cosmetic only, no structural concern visible
    medium = panel damage, likely needs body work
    high   = structural damage, airbag deployment, or total-loss indicators visible
    unclear = cannot reliably assess severity from this image
- confidence: your overall confidence that you have correctly described the damage
- uncertainty: be specific — e.g. "angle obscures left side", "lighting makes depth hard to judge"
- short_visual_summary: factual, visual description only — no opinions on fault or validity

Return only valid JSON. No markdown. No explanation text outside the JSON.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), media_type


def call_vision(client: openai.OpenAI, image_path: Path) -> dict:
    """Call gpt-4o with a single image and return the parsed JSON dict."""
    b64_data, media_type = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_data}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def observe_image(client: openai.OpenAI, image_path: Path) -> ImageObservation:
    """Build one ImageObservation for a given image file."""
    image_id = image_path.stem
    filename = image_path.name

    try:
        vision_data = call_vision(client, image_path)
        return ImageObservation(
            image_id=image_id,
            filename=filename,
            visible_damage=vision_data.get("visible_damage", []),
            damaged_parts=vision_data.get("damaged_parts", []),
            severity_hint=vision_data.get("severity_hint", "unclear"),
            confidence=float(vision_data.get("confidence", 0.0)),
            uncertainty=vision_data.get("uncertainty", ""),
            short_visual_summary=vision_data.get("short_visual_summary", ""),
        )
    except Exception as exc:
        # Log the error but keep going — a failed image gets a placeholder entry.
        print(f"  ERROR processing {filename}: {exc}", file=sys.stderr)
        return ImageObservation(
            image_id=image_id,
            filename=filename,
            visible_damage=[],
            damaged_parts=[],
            severity_hint="unclear",
            confidence=0.0,
            uncertainty=f"Vision model call failed: {exc}",
            short_visual_summary="Could not assess — vision model error.",
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set. Add it to .env or the environment.")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    project_root = Path(__file__).parent.parent
    images_dir = project_root / "data" / "images"
    output_path = project_root / "data" / "image_observations.json"

    if not images_dir.exists():
        print(f"ERROR: images directory not found: {images_dir}")
        sys.exit(1)

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No image files found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {images_dir}")
    print(f"Model: gpt-4o  |  Output: {output_path}\n")

    observations: list[ImageObservation] = []
    for i, image_path in enumerate(image_files, start=1):
        print(f"[{i:2d}/{len(image_files)}] {image_path.name} ...", end=" ", flush=True)
        obs = observe_image(client, image_path)
        observations.append(obs)
        print(f"severity={obs.severity_hint}  confidence={obs.confidence:.2f}")

    result = ObservationsFile(
        generated_at=datetime.now(timezone.utc).isoformat(),
        model="gpt-4o",
        total_images=len(observations),
        observations=observations,
    )

    output_path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. Observations written to {output_path}")
    failed = [o for o in observations if o.confidence == 0.0]
    if failed:
        print(f"WARNING: {len(failed)} image(s) failed: {[o.filename for o in failed]}")


if __name__ == "__main__":
    main()
