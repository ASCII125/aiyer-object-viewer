"""
Factual test - AiyerLite (LLM only, no detector)

Usage:
    python -m tests.factual.ollama.test_lite <image_path>
"""

import asyncio
import logging
import time

from aiyer.adapters.ollama import OllamaAdapter
from aiyer.modules.aiyer_lite import AiyerLite
from tests.setup import setup
from ._common import ImageAnalysis, load_image, print_result


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    image_bytes = load_image()

    model = OllamaAdapter(
        model=setup.test_ollama_vision_model,
        ollama_ip=setup.test_ollama_url,
        ollama_port=setup.test_ollama_port,
    )

    aiyer = AiyerLite(model=model)

    print(f"Running AiyerLite with {setup.test_ollama_vision_model}...")
    start = time.perf_counter()
    result = await aiyer.view(image_bytes, ImageAnalysis)
    elapsed = time.perf_counter() - start

    print_result("AiyerLite", result, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
