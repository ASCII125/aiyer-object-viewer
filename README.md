# Aiyer

> **Alpha** – Functional demonstration. API may change.

Aiyer is a lightweight Python library for structured image analysis using LLMs. Define a Pydantic model, send an image, and get back structured data.

It works with any LLM provider through adapters (Ollama, Groq) and supports multiple analysis strategies with different speed/quality trade-offs.

## Installation

```bash
pip install aiyer            # Core only
pip install aiyer[ollama]    # With Ollama support (local LLMs)
pip install aiyer[groq]      # With Groq support (cloud API)
pip install aiyer[all]       # All providers
```

## Quick Start

```python
import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from aiyer.adapters.ollama import OllamaAdapter
from aiyer.modules import AiyerLite


# Define your schema
class SceneAnalysis(BaseModel):
    summary: str = Field(description="General description of the scene")
    objects: List[str] = Field(description="List of detected objects")
    environment: Optional[str] = Field(description="Environment type")
    danger_level: Literal["low", "medium", "high"] = Field(description="Danger level")


async def main():
    # Create an adapter
    model = OllamaAdapter(
        model="qwen3.5:4b",
        ollama_ip="localhost",
    )

    # Initialize the analyzer
    aiyer = AiyerLite(model=model)

    # Analyze an image
    with open("photo.jpg", "rb") as f:
        result = await aiyer.view(f.read(), SceneAnalysis)

    # result is a VisionResponse[SceneAnalysis]
    # result.image_bytes -> the original image as bytes
    # result.view        -> your SceneAnalysis instance with the LLM output

    print(result.view.summary)
    print(result.view.objects)
    print(result.view.danger_level)


asyncio.run(main())
```

## VisionResponse

Every call to `view()` or `get_result()` returns a `VisionResponse[T]`:

```python
class VisionResponse(BaseModel, Generic[T]):
    image_bytes: bytes   # The original image
    view: T              # Your typed Pydantic model with the LLM analysis
```

`T` is the schema you passed in. So if you call `aiyer.view(img, SceneAnalysis)`, you get back a `VisionResponse[SceneAnalysis]` where `result.view` is a `SceneAnalysis` instance.

## Adapters

**Ollama** (local):

```python
from aiyer.adapters.ollama import OllamaAdapter

model = OllamaAdapter(
    model="qwen3.5:4b",
    ollama_ip="localhost",
    ollama_port=11434,       # optional, default 11434
    ollama_api_key=None,     # optional
    https=False,             # optional
)
```

**Groq** (cloud):

```python
from aiyer.adapters.groq import GroqAdapter

model = GroqAdapter(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key="your-api-key",
    timeout=30.0,            # optional
    max_retries=2,           # optional
    think=False,             # optional, enables reasoning
)
```

## Analysis Modes

| Mode | LLM Calls | Speed | Quality | Use Case |
|------|-----------|-------|---------|----------|
| `AiyerZero` | 1 (resized image) | Fastest | Basic | Quick triage, real-time |
| `AiyerLite` | 1 | Fast | Good | General use, best cost/benefit |
| `AiyerMedium` | 2 (analysis + enrichment) | Slower | Best | When accuracy matters |

```python
from aiyer.modules import AiyerZero, AiyerLite, AiyerMedium

# Fastest – resizes image before sending
aiyer = AiyerZero(model=model, max_image_size=384)

# Balanced – single LLM call, full resolution
aiyer = AiyerLite(model=model)

# Best quality – LLM analyzes, then reviews its own output
aiyer = AiyerMedium(model=model)

result = await aiyer.view(image_bytes, YourSchema)
```

## ContextChat

Use `view_chat` to add context before getting results:

```python
from pydantic import BaseModel, Field
from typing import Literal

class GateStatus(BaseModel):
    status: Literal["open", "closed", "partially_open"] = Field(description="Gate status")
    description: str = Field(description="Gate description")

result = await aiyer.view_chat(image_bytes, GateStatus) \
    .add("Focus on the gate in the center of the image.") \
    .add("Is it open or closed?") \
    .get_result()

# Same VisionResponse – result.view is a GateStatus
print(result.view.status)       # "partially_open"
print(result.view.description)
```

## Schema Features

Aiyer generates smart examples from your Pydantic schema to guide the LLM:

```python
class Report(BaseModel):
    weather: Literal["sunny", "cloudy", "rainy"] = Field(description="Weather condition")
    count: int = Field(description="Number of people")
    items: List[str] = Field(description="Detected items")
```

The LLM receives:
```json
{
  "weather": "<one of: sunny, cloudy, rainy>",
  "count": "<Number of people>",
  "items": ["<Detected items>"]
}
```

`Literal`, `Optional`, `Union`, nested models, and all standard types are supported.

## Custom Adapters

Implement `ILLModel` to add any LLM provider:

```python
from aiyer.interfaces.models import ILLModel, Message

class MyAdapter(ILLModel):
    async def achat(self, messages: list[Message], **kwargs) -> Message:
        # Call your LLM here
        ...
        return Message(role="assistant", content=response_text)
```

## Requirements

- Python >= 3.11
- pydantic >= 2.12

## License

MIT
