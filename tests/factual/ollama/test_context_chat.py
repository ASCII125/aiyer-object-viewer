"""
Factual test - ContextChat (incremental conversation about an image)

Usage:
    python -m tests.factual.ollama.test_context_chat <image_path>
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

    print(f"Running ContextChat with {setup.test_ollama_vision_model}...")
    start = time.perf_counter()

    context = aiyer.view_chat(image_bytes, ImageAnalysis)
    context.add("Focus on the main subjects in the foreground.")
    context.add("Describe colors and positions in detail.")
    result = await context.get_result()

    elapsed = time.perf_counter() - start

    print_result("ContextChat", result, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
