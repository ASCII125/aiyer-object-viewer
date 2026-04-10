"""
Connectors proxys
"""

def get_ollama():
    """
    Get ollama async client
    """
    try:
        from ollama import AsyncClient  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError("Ollama is not installed [pip install ollama]") from err
    return AsyncClient



def get_groq():
    """
    Get Groq async client
    """
    try:
        from groq import AsyncGroq  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError("Groq is not installed [pip install groq]") from err
    return AsyncGroq

def get_pil_image():
    """
    Get PIL image
    """
    try:
        from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError("PIL is not installed [pip install Pillow]") from err
    return Image
