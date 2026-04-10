"""
Shared utilities for Aiyer modules
"""

import io
import json
import re
from typing import Type

from ..providers import get_pil_image


def clean_json_response(content: str) -> str:
    """Strip markdown code fences if present"""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def build_schema_example(model: Type) -> str:
    """Build a JSON example plus a field description guide from a Pydantic model.

    The example shows the exact JSON shape with placeholder values whose JSON
    types match the schema (integer -> 0, number -> 0.0, boolean -> false,
    string -> "..."). Descriptions are NOT inlined inside the example because
    LLMs tend to concatenate them with the placeholder value (producing things
    like the string "0 2 packages remaining" for an integer field).

    Instead, descriptions are emitted in a separate "Field descriptions"
    section keyed by dotted paths (with [] for arrays) so the model can read
    the semantics without any chance of mixing them into the example values.
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    example = _schema_to_example(schema, defs)
    descriptions = _collect_field_descriptions(schema, defs)

    rendered = json.dumps(example, indent=2)
    if not descriptions:
        return rendered

    guide_lines = ["", "Field descriptions:"]
    for path, type_label, desc in descriptions:
        guide_lines.append(f"- {path} ({type_label}): {desc}")
    return rendered + "\n" + "\n".join(guide_lines)


def _schema_to_example(schema: dict, defs: dict):
    """Recursively build a typed JSON example (no descriptions inlined)."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        return _schema_to_example(defs[ref_name], defs)

    # anyOf / oneOf (Union, Optional)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for variant in schema[key]:
                if variant.get("type") == "null":
                    continue
                return _schema_to_example(variant, defs)
            return None

    # enum (Literal)
    if "enum" in schema:
        options = ", ".join(str(v) for v in schema["enum"])
        return f"<one of: {options}>"

    # const
    if "const" in schema:
        return schema["const"]

    schema_type = schema.get("type", "string")

    if schema_type == "object":
        return {
            prop_name: _schema_to_example(prop_schema, defs)
            for prop_name, prop_schema in schema.get("properties", {}).items()
        }

    if schema_type == "array":
        items = schema.get("items", {"type": "string"})
        return [_schema_to_example(items, defs)]

    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None

    return "..."


def _collect_field_descriptions(
    schema: dict,
    defs: dict,
    path: str = "",
    out: list | None = None,
) -> list:
    """Walk the schema and collect (dotted_path, type_label, description) entries."""
    if out is None:
        out = []

    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        return _collect_field_descriptions(defs[ref_name], defs, path, out)

    # anyOf / oneOf — pick the first non-null variant, mirroring the example builder
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for variant in schema[key]:
                if variant.get("type") == "null":
                    continue
                # carry the parent description into the chosen variant if it has none
                if "description" not in variant and schema.get("description"):
                    variant = {**variant, "description": schema["description"]}
                _collect_field_descriptions(variant, defs, path, out)
                break
            return out

    description = schema.get("description")
    if path and description:
        out.append((path, _type_label(schema), description))

    schema_type = schema.get("type", "string")

    if schema_type == "object":
        for prop_name, prop_schema in schema.get("properties", {}).items():
            child_path = f"{path}.{prop_name}" if path else prop_name
            _collect_field_descriptions(prop_schema, defs, child_path, out)
        return out

    if schema_type == "array":
        items = schema.get("items", {"type": "string"})
        _collect_field_descriptions(items, defs, f"{path}[]", out)
        return out

    return out


def _type_label(schema: dict) -> str:
    """Human-readable type label for the field guide."""
    if "enum" in schema:
        options = ", ".join(repr(v) for v in schema["enum"])
        return f"one of {options}"
    return schema.get("type", "string")


def resize_image(image_bytes: bytes, max_size: int = 384, quality: int = 75) -> bytes:
    """Resize image keeping aspect ratio and compress as JPEG"""

    pil_img = get_pil_image()
    img = pil_img.open(io.BytesIO(image_bytes))

    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))

    if img.mode == "RGBA":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
