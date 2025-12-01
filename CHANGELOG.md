# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Additional provider adapters (Mistral, Cohere, Groq)
- Async adapter support
- Batch conversion utilities
- Response caching layer
- WebSocket streaming support

## [0.1.0] - 2025-12-01

### Added
- Initial release of TransLLM
- Core Intermediate Representation (IR) schema
- OpenAI adapter with full feature support
- Anthropic adapter with full feature support
- Google Gemini adapter with full feature support
- Request conversion between all providers
- Response conversion between all providers
- Streaming event conversion
- Idempotency guarantees (A → IR → A = A)
- Multimodal content support (text, images, tools)
- Tool calls and tool results conversion
- Extended reasoning/thinking support
- Comprehensive test suite (108 tests, 100% pass rate)
- Type-safe conversions with Pydantic validation
- Provider capability matrix

### Features
- **Bidirectional Conversion**: Convert requests/responses between any two providers
- **Streaming Support**: Real-time event conversion for streaming APIs
- **Multimodal**: Full support for images, tool calls, and reasoning content
- **Extensible**: Easy to add new providers with custom adapters
- **Type-Safe**: Pydantic models ensure data integrity
- **Well-Tested**: 100% test coverage for core functionality

### Supported Providers
- OpenAI (Chat Completions API)
- Anthropic (Messages API)
- Google Gemini (v1 and v2 series)

### Supported Features
- Text and system messages
- Image content (multimodal)
- Tool definitions and calls
- Tool execution results
- Streaming responses
- Extended thinking/reasoning
- All generation parameters:
  - temperature, top_p, top_k
  - max_tokens, max_completion_tokens
  - stop_sequences
  - stream
  - seed
  - presence_penalty, frequency_penalty
  - logprobs, top_logprobs
  - response_format
  - And more...

### Bug Fixes
- Fixed OpenAI tool call argument serialization (dict → JSON string)
- Fixed OpenAI streaming event type determination
- Fixed empty content field handling in responses
- Fixed system message extraction across providers
- Fixed ContentBlock construction for multimodal content
- Fixed deep comparison for idempotency tests
- Fixed field mapping (stop → stop_sequences)
- Fixed stream field default value for idempotency

### Development
- Full pytest test suite
- Black code formatting
- isort import sorting
- MyPy type checking
- Coverage reporting
- Pre-commit hooks support

---

## Release Notes Format

Each release includes:
- Version number (following SemVer)
- Release date
- Added features
- Changed/Improved functionality
- Bug fixes
- Breaking changes (if any)
- Development dependencies
- Known issues (if any)

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).
