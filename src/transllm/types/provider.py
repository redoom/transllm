"""Provider enumeration for LLM providers with IDE autocomplete support"""

from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEX_AI = "vertex_ai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    NVIDIA_NIM = "nvidia_nim"
