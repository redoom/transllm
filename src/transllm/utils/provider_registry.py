"""Provider registry for centralized adapter management"""

from __future__ import annotations

from typing import Dict, Type

from ..core.base_adapter import BaseAdapter
from ..core.exceptions import UnsupportedProviderError


class ProviderRegistry:
    """Central registry for all provider adapters"""

    _adapters: Dict[str, Type[BaseAdapter]] = {}

    @classmethod
    def register(cls, provider_name: str, adapter_class: Type[BaseAdapter]) -> None:
        """Register a provider adapter class

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            adapter_class: Adapter class for this provider
        """
        if not issubclass(adapter_class, BaseAdapter):
            raise TypeError(
                f"Adapter must be a subclass of BaseAdapter, got {adapter_class}"
            )

        cls._adapters[provider_name.lower()] = adapter_class

    @classmethod
    def get_adapter(cls, provider_name: str) -> BaseAdapter:
        """Get an instance of the adapter for a provider

        Args:
            provider_name: Name of the provider

        Returns:
            An instance of the provider's adapter

        Raises:
            UnsupportedProviderError: If provider is not registered
        """
        provider_key = provider_name.lower()

        if provider_key not in cls._adapters:
            raise UnsupportedProviderError(
                provider_name,
                cls.list_supported_providers(),
            )

        adapter_class = cls._adapters[provider_key]
        return adapter_class(provider_name)

    @classmethod
    def list_supported_providers(cls) -> list[str]:
        """List all supported providers

        Returns:
            List of provider names
        """
        return list(cls._adapters.keys())

    @classmethod
    def is_supported(cls, provider_name: str) -> bool:
        """Check if a provider is supported

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider is supported, False otherwise
        """
        return provider_name.lower() in cls._adapters

    @classmethod
    def unregister(cls, provider_name: str) -> None:
        """Unregister a provider adapter

        Args:
            provider_name: Name of the provider to unregister
        """
        provider_key = provider_name.lower()
        if provider_key in cls._adapters:
            del cls._adapters[provider_key]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters"""
        cls._adapters.clear()


# Convenience functions
def register_adapter(provider_name: str, adapter_class: Type[BaseAdapter]) -> None:
    """Register a provider adapter"""
    ProviderRegistry.register(provider_name, adapter_class)


def get_adapter(provider_name: str) -> BaseAdapter:
    """Get an adapter instance for a provider"""
    return ProviderRegistry.get_adapter(provider_name)


def list_providers() -> list[str]:
    """List all supported providers"""
    return ProviderRegistry.list_supported_providers()


def is_provider_supported(provider_name: str) -> bool:
    """Check if a provider is supported"""
    return ProviderRegistry.is_supported(provider_name)
