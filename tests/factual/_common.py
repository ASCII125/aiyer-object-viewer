"""
Shared schema and output helpers for ollama factual tests
"""

import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class DetectedItem(BaseModel):
    """Item schema"""
    name: str = Field(description="Object name")
    description: str = Field(description="Brief description of the object")
    quantity_occurrences: int = Field(description="Number of occurrences")
    confidence: float = Field(description="Confidence level 0-1")


class ImageAnalysis(BaseModel):
    """Image analysis schema"""
    summary: str = Field(description="General description of the scene")
    objects: List[DetectedItem] = Field(default_factory=list, description="Detected objects")
    total_object_occurrences: int = Field(description="Sum objects.quantity_occurrences")
    environment: Optional[str] = Field(default=None, description="Environment type")


def load_image() -> bytes:
    """Load image from CLI args"""
    if len(sys.argv) < 2:
        print(f"Usage: python -m {sys.modules['__main__'].__package__} <image_path>")  # pylint: disable=no-member
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    image_bytes = image_path.read_bytes()
    print(f"Image loaded: {image_path} ({len(image_bytes)} bytes)")
    return image_bytes


def print_result(label: str, result, elapsed: float):
    """Print formatted result"""
    print(f"\n=== {label} ({elapsed:.2f}s) ===")
    print(f"Summary: {result.view.summary}")
    print(f"Environment: {result.view.environment}")
    print(f"Objects ({len(result.view.objects)}):")
    for obj in result.view.objects:
        print(f"  - {obj.name}: {obj.description} ({obj.confidence:.0%})")
    print(f"\nRaw JSON:\n{result.view.model_dump_json(indent=2)}")
    print(f"\nTotal time: {elapsed:.2f}s")
