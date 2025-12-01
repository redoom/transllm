"""TransLLM - Universal LLM Format Converter

A brand-neutral intermediate representation (IR) for converting between
any two LLM API formats.
"""

from .adapters import OpenAIAdapter
from .utils.provider_registry import ProviderRegistry

# Register built-in adapters
ProviderRegistry.register("openai", OpenAIAdapter)

__version__ = "0.1.0"
__all__ = [
    "OpenAIAdapter",
    "ProviderRegistry",
]
