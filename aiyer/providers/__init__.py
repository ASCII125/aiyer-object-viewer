"""
Lazy import providers
"""

from ._connectors import get_ollama, get_groq

__all__ = ["get_ollama", "get_groq"]
