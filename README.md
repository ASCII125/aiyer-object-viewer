# Aiyer

> **Alpha** - Functional demonstration. API may change.

Aiyer is a lightweight Python library for structured image analysis using LLMs. Define a Pydantic model, send an image, and get back structured data.

It works with any LLM provider through adapters (Ollama, Groq) and supports multiple analysis strategies with different speed/quality trade-offs.

## Installation

```bash
# Core only
pip install aiyer

# With Ollama support (local LLMs)
pip install aiyer[ollama]

# With Groq support (cloud API)
pip install aiyer[groq]

# All providers
pip install aiyer[all]
```

## Quick Start

### 1. Define your schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SceneAnalysis(BaseModel):
    summary: str = Field(description="General description of the scene")
    objects: List[str] = Field(description="List of detected objects")
    environment: Optional[str] = Field(description="Environment type")
    danger_level: Literal["low", "medium", "high"] = Field(description="Danger level")
```

### 2. Analyze an image

```python
import asyncio
from aiyer.adapters.ollama import OllamaAdapter
from aiyer.modules import AiyerLite

async def main():
    model = OllamaAdapter(
        model="qwen3.5:4b",
        ollama_ip="localhost",
    )

    aiyer = AiyerLite(model=model)

    with open("photo.jpg", "rb") as f:
        result = await aiyer.view(f.read(), SceneAnalysis)

    print(result.view.summary)
    print(result.view.objects)
    print(result.view.danger_level)

asyncio.run(main())
```

### Using Groq (cloud)

```python
from aiyer.adapters.groq import GroqAdapter
from aiyer.modules import AiyerLite

model = GroqAdapter(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key="your-api-key",
)

aiyer = AiyerLite(model=model)
result = await aiyer.view(image_bytes, SceneAnalysis)
```

## Analysis Modes

| Mode | LLM Calls | Speed | Quality | Use Case |
|------|-----------|-------|---------|----------|
| `AiyerZero` | 1 (resized image) | Fastest | Basic | Quick triage, real-time |
| `AiyerLite` | 1 | Fast | Good | General use, best cost/benefit |
| `AiyerMedium` | 2 (analysis + enrichment) | Slower | Best | When accuracy matters |

```python
from aiyer.modules import AiyerZero, AiyerLite, AiyerMedium

# Fastest - resizes image before sending
aiyer = AiyerZero(model=model, max_image_size=384)

# Balanced - single LLM call, full resolution
aiyer = AiyerLite(model=model)

# Best quality - LLM analyzes, then reviews its own output
aiyer = AiyerMedium(model=model)
```

## ContextChat

For more control, use `view_chat` to add context before getting results:

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

print(result.view.status)  # "partially_open"
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
