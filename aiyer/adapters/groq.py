"""
Groq LLM adapter
"""

import re
from typing import List, Optional

from ..providers import get_groq
from ..interfaces.models import ILLModel, Message


class GroqAdapter(ILLModel):
    """
    Groq LLM adapter
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 2,
        think: bool = False,
    ):
        self.ctx_client = get_groq()
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.think = think

        try:
            self.client = self.init()
        except Exception as err:
            raise ConnectionError("Error initializing Groq client") from err

    def init(self):
        """
        Init Groq async client
        """
        return self.ctx_client(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    async def achat(self, messages: List[Message], **kwargs) -> Message:
        """
        Async chat
        """
        kwargs.setdefault("temperature", 0)

        if not self.think and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = "none"

        if kwargs.pop("format", None) == "json":
            kwargs["response_format"] = {"type": "json_object"}

        formatted = [self._format_message(m) for m in messages]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted,
                **kwargs,
            )
        except Exception as err:
            if "reasoning_effort" in str(err):
                kwargs.pop("reasoning_effort", None)
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted,
                    **kwargs,
                )
            else:
                raise

        choice = response.choices[0].message
        content = self._strip_think(choice.content)

        return Message(role=choice.role, content=content)

    @staticmethod
    def _strip_think(content: str) -> str:
        """Strip <think>...</think> blocks from model output"""
        return re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()

    @staticmethod
    def _format_message(msg: Message) -> dict:
        """Convert Message to Groq/OpenAI format with multipart content for images"""
        if not msg.images:
            return {"role": msg.role, "content": msg.content}

        content = [{"type": "text", "text": msg.content}]
        for image in msg.images:
            if isinstance(image, bytes):
                import base64  # pylint: disable=import-outside-toplevel
                image = base64.b64encode(image).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            })

        return {"role": msg.role, "content": content}
