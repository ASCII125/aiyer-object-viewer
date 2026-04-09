"""
Shared utilities for Aiyer modules
"""

import io
import json
import re
from typing import Type


def clean_json_response(content: str) -> str:
    """Strip markdown code fences if present"""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def build_schema_example(model: Type) -> str:
    """Build a JSON example with placeholder values from a Pydantic model schema"""
    schema = model.model_json_schema()
    example = _schema_to_example(schema, schema.get("$defs", {}))
    return json.dumps(example, indent=2)


def _schema_to_example(schema: dict, defs: dict):
    """Recursively build an example object from a JSON schema"""
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
    description = schema.get("description")

    if schema_type == "object":
        result = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            result[prop_name] = _schema_to_example(prop_schema, defs)
        return result

    if schema_type == "array":
        items = schema.get("items", {"type": "string"})
        return [_schema_to_example(items, defs)]

    if description:
        return f"<{description}>"

    if schema_type in ("number", "integer"):
        return 0

    if schema_type == "boolean":
        return False

    if schema_type == "null":
        return None

    return "..."


def resize_image(image_bytes: bytes, max_size: int = 384, quality: int = 75) -> bytes:
    """Resize image keeping aspect ratio and compress as JPEG"""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    img = Image.open(io.BytesIO(image_bytes))

    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))

    if img.mode == "RGBA":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
