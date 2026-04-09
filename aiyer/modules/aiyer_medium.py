"""
AiyerMedium - 2 LLM calls (analysis + enrichment)
"""

import base64
import logging
import time
from typing import Type, TypeVar

from ..interfaces.aiyer import Aiyer, VisionResponse
from ..interfaces.models import Message
from ._utils import clean_json_response, build_schema_example


T = TypeVar("T")

logger = logging.getLogger(__name__)


class AiyerMedium(Aiyer):
    """
    Medium Aiyer: 2 LLM calls (free analysis + enrichment review).

    Higher quality, LLM reviews and improves its own output.
    """

    async def view(self, image, expected_view: Type[T]) -> VisionResponse[T]:
        """
        Flow:
        1. 1st LLM call: free analysis
        2. 2nd LLM call: review and enrich
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        image_bytes = bytes(image)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        logger.info("[benchmark] encode: %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        schema_example = build_schema_example(expected_view)

        analysis_prompt = (
            f"Analyze this image. Fill in the following JSON with real data:\n{schema_example}"
        )
        logger.info("[benchmark] prompt_build: %.3fs", time.perf_counter() - t0)

        # 1st LLM call: free analysis
        t0 = time.perf_counter()
        first_response = await self.model.achat(
            [
                Message(
                    role="system",
                    content=(
                        "You are a vision analyst. Respond with a single JSON object "
                        "filled with real data from the image. "
                        "Never return a schema definition, only actual values."
                    ),
                ),
                Message(
                    role="user",
                    content=analysis_prompt,
                    images=[image_b64],
                ),
            ],
            format="json",
        )
        logger.info("[benchmark] llm_analysis: %.3fs", time.perf_counter() - t0)

        # 2nd LLM call: enrichment review
        t0 = time.perf_counter()
        enrichment_prompt = (
            f"Your first analysis:\n{first_response.content}\n\n"
            f"Now look at the image again MORE CAREFULLY. Your task is to IMPROVE your first analysis:\n"
            f"1. Split grouped items into individual entries (e.g. 'several people' -> one entry per person).\n"
            f"2. Add specific details you missed: exact colors, positions (left/center/right, foreground/background), actions, clothing.\n"
            f"3. Add small or background objects you overlooked (text overlays, signs, containers, tools).\n"
            f"4. Fix any inaccuracies from your first pass.\n"
            f"5. Make every description more specific than before — never copy descriptions unchanged.\n"
            f"6. NEVER invent or hallucinate objects. Only include what you can CLEARLY and CONFIDENTLY see.\n\n"
            f"Respond with the complete, improved JSON:\n{schema_example}"
        )

        final_response = await self.model.achat(
            [
                Message(
                    role="system",
                    content=(
                        "You are a senior vision analyst doing a thorough second review. "
                        "Your job is to find everything your first pass missed and add richer detail. "
                        "Every description must be more specific than the original. "
                        "Respond with a single JSON object. Never return a schema definition."
                    ),
                ),
                Message(
                    role="user",
                    content=analysis_prompt,
                    images=[image_b64],
                ),
                Message(
                    role="assistant",
                    content=first_response.content,
                ),
                Message(
                    role="user",
                    content=enrichment_prompt,
                ),
            ],
            format="json",
        )
        logger.info("[benchmark] llm_enrichment: %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        raw_content = clean_json_response(final_response.content)
        view_instance = expected_view.model_validate_json(raw_content)
        logger.info("[benchmark] parse_response: %.3fs", time.perf_counter() - t0)

        logger.info("[benchmark] total: %.3fs", time.perf_counter() - t_total)

        return VisionResponse(image_bytes=image_bytes, view=view_instance)
