"""
Ollama LLM adapter
"""

import os

from typing import List, Optional

from ..providers import get_ollama
from ..interfaces.models import ILLModel, Message


class OllamaAdapter(ILLModel):
    """
    Ollama LLM adapter
    """

    def __init__(
        self,
        model: str,
        ollama_ip: str,
        ollama_port: int = 11434,
        ollama_api_key: Optional[str] = None,
        https: bool = False,
    ):
        self.ctx_client = get_ollama()
        self.ollama_api_key = ollama_api_key
        self.ollama_ip = ollama_ip
        self.ollama_port = ollama_port
        self.model = model
        self.https = https

        try:
            self.client = self.init()
        except Exception as err:
            raise ConnectionError("Error initializing ollama client") from err

    def init(self):
        """
        Init client
        """

        if self.ollama_api_key:
            os.environ["OLLAMA_API_KEY"] = self.ollama_api_key

        client = self.ctx_client(
            host=f"http{'s' if self.https else ''}://{self.ollama_ip}:{self.ollama_port}",
        )

        return client

    async def achat(self, messages: List[Message], **kwargs) -> Message:
        """
        Async chat
        """
        options = kwargs.pop("options", {})
        options.setdefault("temperature", 0)

        kwargs.setdefault("think", False)

        response = await self.client.chat(
            model=self.model,
            messages=[m.model_dump(exclude_none=True) for m in messages],
            options=options,
            **kwargs
        )

        return Message(role=response.message.role, content=response.message.content)

if __name__=="__main__":
    import asyncio

    async def main():
        """test and example usage"""

        adapter = OllamaAdapter(
            ollama_port=11434,
            https=False,
            model="qwen3.5:4b",
            ollama_ip="10.100.0.240",
            ollama_api_key=None,
        )

        result = await adapter.achat([Message(role="user", content="Hello!")])
        print(result)

    asyncio.run(main())
