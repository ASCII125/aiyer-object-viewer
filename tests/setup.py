"""
Setup configuration
"""

from dotenv import dotenv_values


class Setup:
    """
    Setup teste configuration
    """

    def __init__(self):
        self.env_values = dotenv_values(".env")
        self.test_ollama_url = self.env_values.get("TEST_OLLAMA_URL")
        self.test_ollama_port = int(self.env_values.get("TEST_OLLAMA_PORT", 11434))
        self.test_ollama_api_key = self.env_values.get("TEST_OLLAMA_API_KEY")
        self.test_ollama_https = self.env_values.get("TEST_OLLAMA_HTTPS", "false").lower() == "true"
        self.test_ollama_vision_model = self.env_values.get("TEST_OLLAMA_VISION_MODEL", "gemma3:4b")
        self.test_groq_token = self.env_values.get("TEST_GROQ_TOKEN")
        self.test_groq_model = self.env_values.get("TEST_GROQ_MODEL", "qwen/qwen3-32b")
        self.test_groq_vision_model = self.env_values.get("TEST_GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

setup = Setup()
