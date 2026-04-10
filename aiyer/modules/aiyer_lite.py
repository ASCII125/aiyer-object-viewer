"""
AiyerLite - LLM only, no object detection
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


class AiyerLite(Aiyer):
    """
    Lightweight Aiyer: single LLM call, no detector.

    Fastest option (~30-40s), relies entirely on LLM vision.
    """

    async def view(self, image, expected_view: Type[T]) -> VisionResponse[T]:
        """
        Flow:
        1. Single LLM call with image + schema
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        image_bytes = bytes(image)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        logger.info("[benchmark] encode: %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        schema_example = build_schema_example(expected_view)

        user_prompt = (
            f"Analyze this image. Fill in the following JSON with real data:\n{schema_example}"
        )
        logger.info("[benchmark] prompt_build: %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        response = await self.model.achat(
            [
                Message(
                    role="system",
                    content=(
                        "You are a vision analyst. Respond with a single JSON object "
                        "filled with real data from the image. "
                        "Never return a schema definition, only actual values. "
                        "Match each field's JSON type exactly: integers must be plain "
                        "numbers (e.g. 3), not strings. The 'Field descriptions' section "
                        "is provided only as semantic context — never copy its text into "
                        "your response."
                    ),
                ),
                Message(
                    role="user",
                    content=user_prompt,
                    images=[image_b64],
                ),
            ],
            format="json",
        )
        logger.info("[benchmark] llm_analysis: %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        raw_content = clean_json_response(response.content)
        view_instance = expected_view.model_validate_json(raw_content)
        logger.info("[benchmark] parse_response: %.3fs", time.perf_counter() - t0)

        logger.info("[benchmark] total: %.3fs", time.perf_counter() - t_total)

        return VisionResponse(image_bytes=image_bytes, view=view_instance)
