"""
Factual test - AiyerZero with Groq adapter

Usage:
    python -m tests.factual.groq.test_zero <image_path>
"""

import asyncio
import logging
import time

from aiyer.adapters.groq import GroqAdapter
from aiyer.modules.aiyer_zero import AiyerZero
from tests.setup import setup
from tests.factual.ollama._common import ImageAnalysis, load_image, print_result


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    image_bytes = load_image()

    model = GroqAdapter(
        model=setup.test_groq_vision_model,
        api_key=setup.test_groq_token,
    )

    aiyer = AiyerZero(model=model)

    print(f"Running AiyerZero with Groq ({setup.test_groq_vision_model})...")
    start = time.perf_counter()
    result = await aiyer.view(image_bytes, ImageAnalysis)
    elapsed = time.perf_counter() - start

    print_result("AiyerZero (Groq)", result, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
