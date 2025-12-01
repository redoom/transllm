"""Test idempotency of IR ↔ OpenAI conversions"""

import pytest
from src.transllm.converters.request_converter import RequestConverter
from src.transllm.converters.response_converter import ResponseConverter
from src.transllm.utils.provider_registry import ProviderRegistry
from src.transllm.fixtures.openai import (
    OPENAI_CHAT_REQUEST,
    OPENAI_CHAT_RESPONSE,
    OPENAI_TOOL_REQUEST,
    OPENAI_TOOL_RESPONSE,
    OPENAI_STREAM_EVENTS,
    OPENAI_MULTIMODAL_REQUEST,
    OPENAI_FULL_REQUEST,
)


class TestOpenAIRequestIdempotency:
    """Test OpenAI request format conversion idempotency"""

    def test_openai_request_idempotency(self):
        """Test that OpenAI request conversion is idempotent"""
        converter = RequestConverter()
        test_cases = [
            ("Simple Chat Request", OPENAI_CHAT_REQUEST),
            ("Tool Request", OPENAI_TOOL_REQUEST),
            ("Multimodal Request", OPENAI_MULTIMODAL_REQUEST),
            ("Full Request", OPENAI_FULL_REQUEST),
        ]

        for name, data in test_cases:
            is_idempotent = converter.check_idempotency(data, "openai")
            assert is_idempotent, f"Failed idempotency test: {name}"


class TestOpenAIResponseIdempotency:
    """Test OpenAI response format conversion idempotency"""

    def test_openai_response_idempotency(self):
        """Test that OpenAI response conversion is idempotent"""
        converter = ResponseConverter()
        test_cases = [
            ("Simple Chat Response", OPENAI_CHAT_RESPONSE),
            ("Tool Response", OPENAI_TOOL_RESPONSE),
        ]

        for name, data in test_cases:
            is_idempotent = converter.check_idempotency(data, "openai")
            assert is_idempotent, f"Failed idempotency test: {name}"


class TestOpenAIStreamEventIdempotency:
    """Test OpenAI stream event format conversion idempotency"""

    def test_openai_stream_event_idempotency(self):
        """Test that OpenAI stream event conversion is idempotent"""
        adapter = ProviderRegistry.get_adapter("openai")

        for i, event_data in enumerate(OPENAI_STREAM_EVENTS):
            # Convert to unified IR
            unified_event = adapter.to_unified_stream_event(event_data)

            # Convert back to OpenAI format
            converted_event = adapter.from_unified_stream_event(unified_event)

            # Compare
            is_idempotent = adapter.check_idempotency(event_data, converted_event, "stream_event")
            assert is_idempotent, f"Failed stream event idempotency test at index {i}"
