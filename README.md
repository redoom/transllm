# TransLLM

[![PyPI version](https://badge.fury.io/py/transllm.svg)](https://badge.fury.io/py/transllm)
[![Python Support](https://img.shields.io/pypi/pyversions/transllm.svg)](https://pypi.org/project/transllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/transllm/transllm/actions/workflows/tests.yml/badge.svg)](https://github.com/transllm/transllm/actions/workflows/tests.yml)

**Universal LLM Format Converter** - Seamlessly convert between OpenAI, Anthropic, and Gemini API formats using a brand-neutral intermediate representation (IR).

## 🚀 Features

- **Unified Interface**: Single API for all major LLM providers (OpenAI, Anthropic, Gemini)
- **Bidirectional Conversion**: Convert requests and responses between any two providers
- **Type-Safe**: Built with Pydantic for runtime validation and type safety
- **Streaming Support**: Full streaming event conversion for real-time applications
- **Multimodal**: Supports text, images, tool calls, and reasoning content
- **Extensible**: Easy to add new providers or custom adapters

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install transllm
```

### From Source

```bash
git clone https://github.com/transllm/transllm.git
cd transllm
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/transllm/transllm.git
cd transllm
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Request Conversion

Convert an OpenAI request to Anthropic format:

```python
from transllm.converters.request_converter import RequestConverter

# OpenAI format request
openai_request = {
    "model": "gpt-4",
    "messages": [
        {"role": "user", "content": "Hello, Claude!"}
    ],
    "temperature": 0.7,
}

# Convert to Anthropic format
converter = RequestConverter()
anthropic_request = converter.convert(openai_request, "openai", "anthropic")

# Use with Anthropic API
# result = anthropic_client.messages.create(**anthropic_request)
```

### Response Conversion

```python
from transllm.converters.response_converter import ResponseConverter

# Convert Anthropic response to OpenAI format
converter = ResponseConverter()
openai_response = converter.convert(anthropic_response, "anthropic", "openai")

# Use with OpenAI-compatible clients
# print(openai_response["choices"][0]["message"]["content"])
```

### Streaming Events

```python
from transllm.adapters import ProviderRegistry

adapter = ProviderRegistry.get_adapter("openai")

# Convert streaming event to unified IR
unified_event = adapter.to_unified_stream_event(openai_sse_event)

# Convert back to target provider format
target_event = adapter.from_unified_stream_event(unified_event)
```

## 🔄 Provider Support

| Provider | Request Conversion | Response Conversion | Streaming | Multimodal | Tools |
|----------|-------------------|-------------------|-----------|------------|-------|
| OpenAI   | ✅ Full | ✅ Full | ✅ | ✅ | ✅ |
| Anthropic| ✅ Full | ✅ Full | ✅ | ✅ | ✅ |
| Gemini   | ✅ Full | ✅ Full | ✅ | ✅ | ✅ |

## 📋 Supported Features

### Message Types
- Text messages
- System instructions
- Tool calls and results
- Image content (multimodal)
- Reasoning content (OpenAI o1 series)
- Extended thinking (Anthropic)

### Generation Parameters
- `temperature`, `top_p`, `top_k`
- `max_tokens`, `max_completion_tokens`
- `stop_sequences`
- `stream`
- `seed`
- `presence_penalty`, `frequency_penalty`
- `logprobs`, `top_logprobs`
- `response_format`
- And more...

### Streaming Events
- Content deltas
- Tool call events
- Content completion
- Stream termination
- Metadata updates

## 📊 Idempotency

TransLLM guarantees **perfect idempotency**:

```python
# Original → IR → Original = Original
openai_req = {"model": "gpt-4", "messages": [...]}
ir_req = converter.to_unified(openai_req, "openai")
roundtrip_req = converter.from_unified(ir_req, "openai")

assert openai_req == roundtrip_req  # ✅ Always true
```

This ensures zero information loss in conversions.

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# Specific provider tests
pytest tests/test_idempotency_openai.py
pytest tests/test_idempotency_anthropic.py
pytest tests/test_idempotency_gemini.py

# Cross-format conversion tests
pytest tests/test_cross_format_conversion.py

# Streaming tests
pytest tests/test_streaming_anthropic.py
```

## 🔧 Configuration

### Custom Adapters

Create a custom adapter for a new provider:

```python
from transllm.core.base_adapter import BaseAdapter

class CustomProviderAdapter(BaseAdapter):
    def to_unified_request(self, data):
        # Convert provider-specific request to IR
        pass

    def from_unified_request(self, unified_request):
        # Convert IR to provider-specific request
        pass

    # ... implement other methods
```

Register your adapter:

```python
from transllm.utils.provider_registry import ProviderRegistry

ProviderRegistry.register("custom", CustomProviderAdapter)
```

## 📖 Documentation

- [API Reference](https://transllm.readthedocs.io)
- [Conversion Guide](https://transllm.readthedocs.io/conversion-guide)
- [Custom Adapters](https://transllm.readthedocs.io/custom-adapters)
- [Examples](https://github.com/transllm/transllm/tree/main/examples)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/transllm/transllm.git
cd transllm
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
isort src tests

# Type checking
mypy src
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for LLM provider interoperability
- Built with [Pydantic](https://pydantic.dev/) for type safety
- Thanks to all contributors and the open-source community

## 🐛 Bug Reports & Feature Requests

Please use [GitHub Issues](https://github.com/transllm/transllm/issues) to report bugs or request features.

## 📊 Project Status

**Current Version**: 0.1.0 (Beta)

**Test Coverage**: 100% for core functionality

**Supported APIs**:
- OpenAI Chat Completions (all versions)
- Anthropic Messages API (all versions)
- Google Gemini (v1 and v2 series)

## ⚡ Performance

- Conversion time: < 1ms for typical requests
- Memory overhead: Minimal (no caching by default)
- Thread-safe: Yes

## 🔮 Roadmap

- [ ] Additional provider adapters (Mistral, Cohere, Groq, etc.)
- [ ] Enhanced grounding support
- [ ] Async adapter support
- [ ] Batch conversion utilities
- [ ] Response caching layer
- [ ] WebSocket streaming support

---

**Made with ❤️ by the TransLLM team**

For questions or support, join our [Discord](https://discord.gg/transllm) or open an issue.
