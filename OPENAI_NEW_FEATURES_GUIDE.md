# OpenAI New Features Quick Reference

## Usage Examples

### 1. JSON Mode (response_format)
```python
from src.transllm.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter()

# OpenAI format
request = {
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": "Generate JSON"}],
    "response_format": {"type": "json_object"}
}

# Convert to IR
unified = adapter.to_unified_request(request)
assert unified.generation_params.response_format == {"type": "json_object"}

# Convert back to OpenAI
openai_request = adapter.from_unified_request(unified)
assert openai_request["response_format"] == {"type": "json_object"}
```

### 2. Logit Bias (Token Probability Control)
```python
request = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Say good things"}],
    "logit_bias": {
        "1234": 50,     # Boost token 1234 probability
        "5678": -100    # Suppress token 5678
    }
}

unified = adapter.to_unified_request(request)
assert unified.generation_params.logit_bias["1234"] == 50
```

### 3. Multiple Completions (n > 1)
```python
request = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Complete this"}],
    "n": 3  # Request 3 different completions
}

unified = adapter.to_unified_request(request)
assert unified.generation_params.n == 3

# Response will have 3 different choices
response = {"choices": [
    {"index": 0, "message": {"role": "assistant", "content": "Option 1"}},
    {"index": 1, "message": {"role": "assistant", "content": "Option 2"}},
    {"index": 2, "message": {"role": "assistant", "content": "Option 3"}},
]}
```

### 4. Parallel Tool Calls
```python
request = {
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": "Get weather and news"}],
    "tools": [
        {"type": "function", "function": {"name": "get_weather", ...}},
        {"type": "function", "function": {"name": "get_news", ...}}
    ],
    "parallel_tool_calls": True  # Allow both tools simultaneously
}

unified = adapter.to_unified_request(request)
assert unified.parallel_tool_calls is True
```

### 5. Reasoning Content (o1 Models)
```python
# OpenAI o1 model response with thinking
response = {
    "id": "chatcmpl-xxx",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "The answer is 42",
            "reasoning_content": "Let me think through this step by step..."
        }
    }]
}

unified = adapter.to_unified_response(response)
choice = unified.choices[0]
assert choice.message.reasoning_content == "Let me think through this step by step..."
```

### 6. Log Probabilities
```python
request = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "logprobs": True,       # Enable log probabilities
    "top_logprobs": 2       # Return top 2 alternatives for each token
}

unified = adapter.to_unified_request(request)
assert unified.generation_params.logprobs is True
assert unified.generation_params.top_logprobs == 2
```

### 7. Max Completion Tokens (Alternative to max_tokens)
```python
request = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Write a story"}],
    "max_completion_tokens": 1000  # Newer API version
}

unified = adapter.to_unified_request(request)
assert unified.generation_params.max_completion_tokens == 1000
```

### 8. Stream Options
```python
request = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
    "stream_options": {"include_usage": True}  # Include usage in stream
}

unified = adapter.to_unified_request(request)
assert unified.generation_params.stream_options == {"include_usage": True}
```

### 9. Combined Features
```python
# Use multiple new features together
request = {
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": "Generate multiple JSON responses"}],
    "response_format": {"type": "json_object"},
    "logit_bias": {"100": 10},
    "n": 2,
    "max_completion_tokens": 500,
    "logprobs": True,
    "top_logprobs": 2
}

unified = adapter.to_unified_request(request)

# All features preserved
assert unified.generation_params.response_format is not None
assert unified.generation_params.logit_bias is not None
assert unified.generation_params.n == 2
assert unified.generation_params.max_completion_tokens == 500
assert unified.generation_params.logprobs is True
assert unified.generation_params.top_logprobs == 2
```

## Feature Support Matrix

| Feature | Type | Support | Notes |
|---------|------|---------|-------|
| `logit_bias` | Request | ✅ Full | Dict[str, float] |
| `response_format` | Request | ✅ Full | JSON mode, schema |
| `parallel_tool_calls` | Request | ✅ Full | Boolean |
| `logprobs` | Request | ✅ Full | Boolean |
| `top_logprobs` | Request | ✅ Full | Integer >= 0 |
| `n` | Request | ✅ Full | Integer >= 1 |
| `max_completion_tokens` | Request | ✅ Full | Alternative to max_tokens |
| `stream_options` | Request | ✅ Full | Dict[str, Any] |
| `reasoning_content` | Response | ✅ Full | String (o1 models) |
| `thinking_blocks` | Response | ✅ Full | List[Dict] |

## API Changes

### GenerationParameters
```python
# New fields (all optional)
logit_bias: Optional[Dict[str, float]] = None
response_format: Optional[Dict[str, Any]] = None
logprobs: Optional[bool] = None
top_logprobs: Optional[int] = None
n: Optional[int] = None
max_completion_tokens: Optional[int] = None
stream_options: Optional[Dict[str, Any]] = None
```

### CoreRequest
```python
# New field (optional)
parallel_tool_calls: Optional[bool] = None
```

### ResponseMessage
```python
# New fields (optional)
reasoning_content: Optional[str] = None
thinking_blocks: Optional[List[Dict[str, Any]]] = None
```

## Testing

Run all tests:
```bash
pytest tests/test_idempotency_openai.py tests/test_openai_new_features.py -v
```

Run specific feature test:
```bash
pytest tests/test_openai_new_features.py::test_json_mode_idempotency -v
```

## Backward Compatibility

✅ All changes are fully backward compatible:
- All new fields are optional
- No modifications to existing fields
- No changes to method signatures
- Existing code requires no updates

## Performance

- No performance impact
- Same conversion speed as before
- Minimal memory overhead from new fields

---

**For more details, see:**
- `OPENAI_FEATURE_GAP_ANALYSIS.md` - Feature comparison with litellm
- `PHASE_1_1_OPENAI_ENHANCEMENT_SUMMARY.md` - Implementation details
- `tests/test_openai_new_features.py` - Complete test examples
