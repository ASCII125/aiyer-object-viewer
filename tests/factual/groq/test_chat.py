"""
Factual test - Groq adapter basic chat

Usage:
    python -m tests.factual.groq.test_chat
"""

import asyncio
import logging
import time

from aiyer.adapters.groq import GroqAdapter
from aiyer.interfaces.models import Message
from tests.setup import setup


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = GroqAdapter(
        model=setup.test_groq_model,
        api_key=setup.test_groq_token,
    )

    print(f"Running Groq chat with {setup.test_groq_model}...")
    start = time.perf_counter()

    result = await model.achat([
        Message(role="system", content="You are a helpful assistant. Respond concisely."),
        Message(role="user", content="What is 2 + 2? Answer in one word."),
    ])

    elapsed = time.perf_counter() - start

    print(f"\n=== Groq Chat Result ({elapsed:.2f}s) ===")
    print(f"Role: {result.role}")
    print(f"Content: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
