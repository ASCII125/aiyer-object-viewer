"""
Core module interface
"""

import base64
from typing import TypeVar, Generic, SupportsBytes, Type, List

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .models import ILLModel, Message


T = TypeVar("T", bound=BaseModel)


class VisionResponse(BaseModel, Generic[T]):
    """
    Return context image
    """

    model_config = {"arbitrary_types_allowed": True}

    image_bytes: bytes = Field(description="Image bytes")
    view: T = Field(description="View instance")


class ContextChat(Generic[T]):
    """
    Incremental conversation about an image.

    Allows adding text context before resolving to a structured response.
    """

    def __init__(
        self,
        model: ILLModel,
        image_b64: str,
        image_bytes: bytes,
        expected_view: Type[T],
    ):
        self._model = model
        self._image_b64 = image_b64
        self._image_bytes = image_bytes
        self._expected_view = expected_view
        self._messages: List[str] = []

    def add(self, content: str) -> "ContextChat[T]":
        """Add user context or instructions"""
        self._messages.append(content)
        return self

    async def get_result(self) -> VisionResponse[T]:
        """Send everything to LLM and return structured response"""
        from ..modules._utils import build_schema_example, clean_json_response  # pylint: disable=import-outside-toplevel

        schema_example = build_schema_example(self._expected_view)

        messages = [
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
                content=f"Analyze this image. Fill in the following JSON with real data:\n{schema_example}",
                images=[self._image_b64],
            ),
        ]

        for msg in self._messages:
            messages.append(Message(role="user", content=msg))

        response = await self._model.achat(messages, format="json")

        raw_content = clean_json_response(response.content)
        view_instance = self._expected_view.model_validate_json(raw_content)

        return VisionResponse(image_bytes=self._image_bytes, view=view_instance)


class VisionTask(Generic[T]):
    """
    Task vision
    """
    prompt: str
    schema: Type[T]


class Aiyer(ABC):
    """
    Aiyer main class
    """

    def __init__(self, model: ILLModel):
        self.model = model

    @abstractmethod
    async def view(self, image: SupportsBytes, expected_view: Type[T]) -> VisionResponse[T]:
        """
        Analyze image and return context
        """

    def view_chat(self, image: SupportsBytes, expected_view: Type[T]) -> ContextChat[T]:
        """
        Start an incremental conversation about an image.
        Use .add() to provide context, then .get_result() to resolve.
        """
        image_bytes = bytes(image)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return ContextChat(self.model, image_b64, image_bytes, expected_view)
