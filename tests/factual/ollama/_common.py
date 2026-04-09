"""
Shared schema and output helpers for ollama factual tests
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from tests.setup import setup


class DetectedItem(BaseModel):
    name: str = Field(description="Object name")
    description: str = Field(description="Brief description of the object")
    confidence: float = Field(description="Confidence level 0-1")


class ImageAnalysis(BaseModel):
    summary: str = Field(description="General description of the scene")
    objects: List[DetectedItem] = Field(default_factory=list, description="Detected objects")
    environment: Optional[str] = Field(default=None, description="Environment type")


def load_image() -> bytes:
    """Load image from CLI args"""
    if len(sys.argv) < 2:
        print(f"Usage: python -m {sys.modules['__main__'].__package__} <image_path>")
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
