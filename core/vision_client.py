"""
vision_client.py — OpenAI gpt-4o vision wrapper.

Responsibilities:
- VISION_PROMPT: the structured extraction prompt sent with every image. Stored
  here so DamageEvidenceAgent does not embed prompt text.
- call_vision_model(b64_image, prompt): POST to OpenAI chat completions with a
  multimodal message (image_url + text). Return the raw JSON string from the
  model response.
- parse_vision_response(raw_json): validate the JSON string into VisionFindings
  via model_validate_json(). Raise ValueError if the response cannot be parsed.
- Image encoding helper: encode_image_to_base64(image_path) → str.

Error contract:
  Any exception from the OpenAI SDK is allowed to propagate. DamageEvidenceAgent
  catches it and sets confidence=0.0, image_claim_mismatch=True.
"""

from __future__ import annotations

import base64
import pathlib

import openai

from core.claim_state import VisionFindings


VISION_PROMPT = """
You are a vehicle damage assessment expert. Inspect this car damage image and return a JSON object with:
- damage_type: one of [collision, scrape, dent, glass, fire, flood, vandalism, total_loss, none_visible]
- severity: one of [none, minor, moderate, severe, total_loss]
- affected_components: list of damaged parts visible (e.g. ["rear_bumper", "trunk_lid"])
- estimated_repair_scope: one of [cosmetic, panel_repair, structural, total_loss]
- confidence: float 0.0-1.0 representing your confidence in this assessment
- observations: list of specific visual evidence strings (max 5)
Return only valid JSON. No explanation text.
""".strip()


def encode_image_to_base64(image_path: str | pathlib.Path) -> str:
    """Read image bytes from disk and return a base64-encoded string."""
    raise NotImplementedError


def call_vision_model(b64_image: str, prompt: str = VISION_PROMPT) -> str:
    """Send image + prompt to gpt-4o. Return raw response content string."""
    raise NotImplementedError


def parse_vision_response(raw_json: str) -> VisionFindings:
    """Validate raw JSON string into a VisionFindings model."""
    raise NotImplementedError
