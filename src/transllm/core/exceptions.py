"""TransLLM exception classes for conversion errors"""

from __future__ import annotations

from typing import Any, Dict, Optional


class TransLLMError(Exception):
    """Base exception for all TransLLM errors"""
    pass


class ConversionError(TransLLMError):
    """Raised when conversion between formats fails"""

    def __init__(
        self,
        message: str,
        from_provider: str,
        to_provider: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.from_provider = from_provider
        self.to_provider = to_provider
        self.details = details or {}


class UnsupportedProviderError(TransLLMError):
    """Raised when an unsupported provider is requested"""

    def __init__(self, provider: str, supported_providers: list[str]) -> None:
        self.provider = provider
        self.supported_providers = supported_providers
        message = (
            f"Unsupported provider: '{provider}'. "
            f"Supported providers: {', '.join(supported_providers)}"
        )
        super().__init__(message)


class UnsupportedFeatureError(TransLLMError):
    """Raised when a feature is not supported by the target provider"""

    def __init__(
        self,
        feature: str,
        provider: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feature = feature
        self.provider = provider
        self.details = details or {}
        message = f"Provider '{provider}' does not support feature: '{feature}'"
        super().__init__(message)


class ValidationError(TransLLMError):
    """Raised when data validation fails"""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[list[str]] = None,
    ) -> None:
        super().__init__(message)
        self.validation_errors = validation_errors or []


class IdempotencyError(TransLLMError):
    """Raised when idempotency test fails (A -> IR -> A)"""

    def __init__(
        self,
        original_data: Dict[str, Any],
        final_data: Dict[str, Any],
        differences: list[str],
    ) -> None:
        self.original_data = original_data
        self.final_data = final_data
        self.differences = differences
        message = (
            "Idempotency test failed. Data changed after round-trip conversion:\n"
            + "\n".join(differences)
        )
        super().__init__(message)
