"""Anthropic adapter streaming event tests

Tests verify that streaming events are correctly converted between
Anthropic SSE format and unified IR StreamEvent format.
"""

import pytest
from src.transllm.adapters.anthropic import AnthropicAdapter
from src.transllm.fixtures.anthropic import (
    ANTHROPIC_STREAMING_EVENTS,
    ANTHROPIC_STREAMING_TOOL_EVENTS,
    ANTHROPIC_STREAMING_THINKING_EVENTS,
)


class TestAnthropicStreamingConversion:
    """Test streaming event conversion"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_text_streaming_events_conversion(self):
        """Test that text streaming events are correctly converted"""
        events = ANTHROPIC_STREAMING_EVENTS

        unified_events = []
        sequence_id = 0
        timestamp = 0.0

        for event in events:
            unified = self.adapter.to_unified_stream_event(event, sequence_id, timestamp)
            unified_events.append(unified)
            sequence_id += 1
            timestamp += 0.01

        # Should have events for message_start, content_block_start, deltas, stops, message_stop
        assert len(unified_events) > 0

        # First event should be metadata_update for message_start
        assert unified_events[0].type.value == "metadata_update"
        assert unified_events[0].metadata.get("event") == "message_start"

        # Should have content_delta events for text chunks
        content_deltas = [e for e in unified_events if e.type.value == "content_delta"]
        assert len(content_deltas) == 2  # "Hello" and " world"
        assert content_deltas[0].content_delta == "Hello"
        assert content_deltas[1].content_delta == " world"

        # Should have content_finish and stream_end events
        content_finishes = [e for e in unified_events if e.type.value == "content_finish"]
        assert len(content_finishes) >= 1

        stream_ends = [e for e in unified_events if e.type.value == "stream_end"]
        assert len(stream_ends) >= 1

    def test_text_streaming_events_round_trip(self):
        """Test that text streaming events round-trip correctly"""
        events = ANTHROPIC_STREAMING_EVENTS

        for i, event in enumerate(events):
            # Convert to unified
            unified = self.adapter.to_unified_stream_event(event, i, 0.01 * i)

            # Convert back to Anthropic
            result = self.adapter.from_unified_stream_event(unified)

            # Check that key event type conversions are preserved
            # All standard Anthropic event types should be represented
            assert result["type"] in [
                "content_block_delta",
                "content_block_start",
                "content_block_stop",
                "message_stop",
                "message_start",
                "message_delta",
                "metadata_update",
            ]

    def test_tool_streaming_events_conversion(self):
        """Test that tool call streaming events are correctly converted"""
        events = ANTHROPIC_STREAMING_TOOL_EVENTS

        unified_events = []
        for i, event in enumerate(events):
            unified = self.adapter.to_unified_stream_event(event, i, 0.01 * i)
            unified_events.append(unified)

        # Should have tool_call_delta events
        tool_deltas = [e for e in unified_events if e.type.value == "tool_call_delta"]
        assert len(tool_deltas) == 2

        # Check that tool argument deltas are captured
        # tool_call_delta is a Pydantic model, access as attribute
        assert '{"location"' in tool_deltas[0].tool_call_delta.arguments_delta
        assert ': "Beijing"}' in tool_deltas[1].tool_call_delta.arguments_delta

    def test_tool_streaming_events_round_trip(self):
        """Test that tool streaming events round-trip correctly"""
        events = ANTHROPIC_STREAMING_TOOL_EVENTS

        for i, event in enumerate(events):
            unified = self.adapter.to_unified_stream_event(event, i, 0.01 * i)
            result = self.adapter.from_unified_stream_event(unified)

            # Event types should be preserved or mapped correctly
            assert result["type"] in [
                "content_block_delta",
                "content_block_start",
                "content_block_stop",
                "message_stop",
                "message_start",
                "message_delta",
                "metadata_update",
            ]

    def test_thinking_streaming_events_conversion(self):
        """Test that thinking block streaming events are correctly converted"""
        events = ANTHROPIC_STREAMING_THINKING_EVENTS

        unified_events = []
        for i, event in enumerate(events):
            unified = self.adapter.to_unified_stream_event(event, i, 0.01 * i)
            unified_events.append(unified)

        # Should have multiple content blocks (thinking and text)
        metadata_updates = [e for e in unified_events if e.type.value == "metadata_update"]
        assert len(metadata_updates) >= 2  # At least content_block_start for thinking and text

    def test_stream_event_sequence_ids(self):
        """Test that sequence_id is properly maintained in stream events"""
        event = ANTHROPIC_STREAMING_EVENTS[2]  # A content_block_delta event

        for seq_id in [0, 1, 5, 10]:
            unified = self.adapter.to_unified_stream_event(event, seq_id, 1.0)
            assert unified.sequence_id == seq_id

    def test_stream_event_timestamps(self):
        """Test that timestamps are properly maintained in stream events"""
        event = ANTHROPIC_STREAMING_EVENTS[2]

        for timestamp in [0.0, 1.5, 10.123]:
            unified = self.adapter.to_unified_stream_event(event, 0, timestamp)
            assert unified.timestamp == timestamp

    def test_stream_event_content_index(self):
        """Test that content_index is properly extracted and preserved"""
        # Event with index 0
        unified_0 = self.adapter.to_unified_stream_event(
            ANTHROPIC_STREAMING_EVENTS[2], 0, 0.0
        )
        assert unified_0.content_index == 0

        # Event with different index
        tool_event_with_index = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "test"},
        }
        unified_1 = self.adapter.to_unified_stream_event(tool_event_with_index, 0, 0.0)
        assert unified_1.content_index == 1

    def test_message_delta_event_handling(self):
        """Test that message_delta events are handled gracefully"""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        }

        unified = self.adapter.to_unified_stream_event(event, 0, 0.0)
        # Should convert to metadata_update or similar
        assert unified is not None
        assert unified.type.value in ["metadata_update", "content_finish"]

    def test_empty_delta_handling(self):
        """Test handling of empty delta fields"""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "",
            },
        }

        unified = self.adapter.to_unified_stream_event(event, 0, 0.0)
        assert unified.type.value == "content_delta"
        assert unified.content_delta == ""

    def test_event_round_trip_preserves_structure(self):
        """Test that event round-trip preserves overall structure"""
        # Test a simple text delta event
        original_event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello, world!",
            },
        }

        unified = self.adapter.to_unified_stream_event(original_event, 0, 0.0)
        result = self.adapter.from_unified_stream_event(unified)

        # Key structure should be preserved
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "text_delta"
        assert result["delta"]["text"] == "Hello, world!"


class TestAnthropicStreamingEdgeCases:
    """Test edge cases in streaming conversion"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_malformed_event_graceful_fallback(self):
        """Test that malformed events fall back gracefully"""
        malformed = {
            "type": "unknown_event_type",
            "data": "some_data",
        }

        unified = self.adapter.to_unified_stream_event(malformed, 0, 0.0)
        # Should not raise, should return some metadata_update event
        assert unified is not None
        assert unified.type.value == "metadata_update"

    def test_missing_fields_handling(self):
        """Test handling of events with missing optional fields"""
        minimal_event = {
            "type": "content_block_delta",
            # Missing index and delta - will use defaults
        }

        unified = self.adapter.to_unified_stream_event(minimal_event, 0, 0.0)
        # Should handle gracefully - will be treated as unknown event type
        # since delta is missing
        assert unified is not None

    def test_event_with_metadata_preservation(self):
        """Test that event metadata is preserved during conversion"""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "test",
            },
            "metadata": {
                "custom_field": "custom_value",
            },
        }

        unified = self.adapter.to_unified_stream_event(event, 0, 0.0)
        assert unified.metadata is not None
        assert unified.metadata.get("custom_field") == "custom_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
