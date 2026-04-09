"""
Models interfaces
"""

from typing import List, Literal, Optional, Union
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    Message
    """
    role: Literal["user", "assistant", "system"] = Field(description="Role")
    content: str = Field(description="Content")
    images: Optional[List[Union[str, bytes]]] = Field(
        default=None,
        description="Images as base64 strings, file paths, or raw bytes",
    )


class ILLModel(ABC):
    """
    Base LLM class
    """

    @abstractmethod
    async def achat(self, messages: List[Message], **kwargs) -> Message:
        """
        Async chat
        """
