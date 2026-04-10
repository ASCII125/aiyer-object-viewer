"""
AiyerZero - Maximum performance, single LLM call with resized image
"""

import base64
import logging
import time
from typing import Type, TypeVar

from ..interfaces.aiyer import Aiyer, VisionResponse
from ..interfaces.models import Message
from ._utils import clean_json_response, build_schema_example, resize_image


T = TypeVar("T")

logger = logging.getLogger(__name__)


class AiyerZero(Aiyer):
    """
    Zero-overhead Aiyer: resized image + single minimal LLM call.

    Target: < 10s. No detector, compressed image, minimal prompt.
    """

    def __init__(self, model, max_image_size: int = 384):
        super().__init__(model)
        self.max_image_size = max_image_size

    async def view(self, image, expected_view: Type[T]) -> VisionResponse[T]:
        """
        Flow:
        1. Resize image
        2. Single LLM call with minimal prompt
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        image_bytes = bytes(image)
        resized = resize_image(image_bytes, self.max_image_size)
        image_b64 = base64.b64encode(resized).decode("utf-8")
        logger.info(
            "[benchmark] resize+encode: %.3fs (%d -> %d bytes)",
            time.perf_counter() - t0, len(image_bytes), len(resized),
        )

        t0 = time.perf_counter()
        schema_example = build_schema_example(expected_view)
        logger.info("[benchmark] prompt_build: %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        response = await self.model.achat(
            [
                Message(
                    role="system",
                    content=(
                        "Respond with JSON only. Match each field's JSON type exactly "
                        "(integers as plain numbers, not strings). The 'Field descriptions' "
                        "section is context only — never copy its text into your response."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Analyze this image. Fill JSON with real data:\n{schema_example}",
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
