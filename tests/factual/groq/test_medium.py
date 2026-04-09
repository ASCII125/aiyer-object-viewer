"""
Factual test - AiyerMedium with Groq adapter (2 LLM calls: analysis + enrichment)

Usage:
    python -m tests.factual.groq.test_medium <image_path>
"""

import asyncio
import logging
import time

from aiyer.adapters.groq import GroqAdapter
from aiyer.modules.aiyer_medium import AiyerMedium
from tests.setup import setup
from tests.factual.ollama._common import ImageAnalysis, load_image, print_result


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    image_bytes = load_image()

    model = GroqAdapter(
        model=setup.test_groq_vision_model,
        api_key=setup.test_groq_token,
    )

    aiyer = AiyerMedium(model=model)

    print(f"Running AiyerMedium with Groq ({setup.test_groq_vision_model})...")
    start = time.perf_counter()
    result = await aiyer.view(image_bytes, ImageAnalysis)
    elapsed = time.perf_counter() - start

    print_result("AiyerMedium (Groq)", result, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
