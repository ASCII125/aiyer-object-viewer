"""
Factual test - AiyerZero (max performance, resized image, single LLM call)

Usage:
    python -m tests.factual.ollama.test_zero <image_path>
"""

import asyncio
import logging
import time

from aiyer.adapters.ollama import OllamaAdapter
from aiyer.modules.aiyer_zero import AiyerZero
from tests.setup import setup
from .._common import ImageAnalysis, load_image, print_result


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    image_bytes = load_image()

    model = OllamaAdapter(
        model=setup.test_ollama_vision_model,
        ollama_ip=setup.test_ollama_url,
        ollama_port=setup.test_ollama_port,
        ollama_api_key=setup.test_ollama_api_key,
        https=setup.test_ollama_https,
    )

    aiyer = AiyerZero(model=model)

    print(f"Running AiyerZero with {setup.test_ollama_vision_model}...")
    start = time.perf_counter()
    result = await aiyer.view(image_bytes, ImageAnalysis)
    elapsed = time.perf_counter() - start

    print_result("AiyerZero", result, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
