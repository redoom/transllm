# OpenAI Feature Gap Analysis: TransLLM vs litellm

## Overview
Comparison of OpenAI adapter implementation between TransLLM and litellm to identify missing features.

## Request Parameters Comparison

### ✅ Already Implemented
| Parameter | TransLLM | litellm | Type | Purpose |
|-----------|----------|---------|------|---------|
| `model` | CoreRequest.model | ChatCompletionRequest | string | Model identifier |
| `messages` | CoreRequest.messages | ChatCompletionRequest | List[Message] | Conversation history |
| `tools` | CoreRequest.tools | ChatCompletionRequest | List[Tool] | Available tools |
| `tool_choice` | CoreRequest.tool_choice | ChatCompletionRequest | string/dict | Tool selection strategy |
| `system_instruction` | CoreRequest.system_instruction | N/A | string | System-level instruction |
| `temperature` | GenerationParameters.temperature | ChatCompletionRequest | float [0-2] | Sampling randomness |
| `max_tokens` | GenerationParameters.max_tokens | ChatCompletionRequest | int | Max output length |
| `top_p` | GenerationParameters.top_p | ChatCompletionRequest | float [0-1] | Nucleus sampling |
| `top_k` | GenerationParameters.top_k | N/A | int | Top-K sampling |
| `stop_sequences` | GenerationParameters.stop_sequences | ChatCompletionRequest (stop) | List[str] | Stop generation tokens |
| `seed` | GenerationParameters.seed | ChatCompletionRequest | int | Random seed |
| `presence_penalty` | GenerationParameters.presence_penalty | ChatCompletionRequest | float | Presence penalty |
| `frequency_penalty` | GenerationParameters.frequency_penalty | ChatCompletionRequest | float | Frequency penalty |
| `metadata` | CoreRequest.metadata | ChatCompletionRequest | dict | Request metadata |

### ❌ Missing in GenerationParameters
| Parameter | litellm | Required? | Notes |
|-----------|---------|-----------|-------|
| `logit_bias` | dict | High | Modify token probability distribution |
| `response_format` | dict | High | JSON mode and response schema |
| `logprobs` | bool | Medium | Return token log probabilities |
| `top_logprobs` | int | Medium | Number of log probs to return |
| `n` | int | Medium | Number of completions to generate |
| `max_completion_tokens` | int | Low | Alternative to max_tokens (newer API) |
| `stream_options` | dict | Low | Streaming configuration (e.g., include_usage) |
| `service_tier` | str | Low | Request tier (auto/default) |
| `safety_identifier` | str | Low | Safety setting identifier |

### ❌ Missing in CoreRequest
| Parameter | litellm | Required? | Notes |
|-----------|---------|-----------|-------|
| `parallel_tool_calls` | bool | High | Allow parallel tool execution |
| `function_call` | str/dict | Low | Deprecated but still used |
| `functions` | List | Low | Deprecated function definitions |
| `user` | str | Low | User identifier for usage tracking |

---

## Response Fields Comparison

### ✅ Already Implemented
| Field | TransLLM | litellm | Type | Purpose |
|-------|----------|---------|------|---------|
| `id` | CoreResponse.id | ChatCompletionResponse | string | Response identifier |
| `object` | CoreResponse.object | ChatCompletionResponse | string | Object type |
| `created` | CoreResponse.created | ChatCompletionResponse | int | Creation timestamp |
| `model` | CoreResponse.model | ChatCompletionResponse | string | Model used |
| `choices` | CoreResponse.choices | ChatCompletionResponse | List[Choice] | Response choices |
| `usage` | CoreResponse.usage | ChatCompletionUsageBlock | UsageStatistics | Token usage |
| `metadata` | CoreResponse.metadata | N/A | dict | Response metadata |

### ✅ Already in Choices
| Field | TransLLM | litellm | Type |
|-------|----------|---------|------|
| `index` | Choice.index | integer | Choice index |
| `message` | Choice.message | ChatCompletionResponseMessage | Response message |
| `finish_reason` | Choice.finish_reason | enum | Completion reason |
| `logprobs` | Choice.logprobs | dict | Log probabilities |

### ❌ Missing in ResponseMessage
| Field | litellm | Required? | Notes |
|-------|---------|-----------|-------|
| `reasoning_content` | string | Medium | OpenAI o1 thinking content |
| `thinking_blocks` | List | Low | Structured thinking blocks |
| `provider_specific_fields` | dict | Low | Provider-specific extensions |

### ❌ Missing in UsageStatistics
| Field | litellm | Required? | Notes |
|-------|---------|-----------|-------|
| `prompt_tokens_details` | dict | Low | Breakdown of prompt tokens |
| `completion_tokens_details` | dict | Low | Breakdown of completion tokens |

---

## Stream Event Comparison

### ✅ Already Implemented
| Field | TransLLM | Notes |
|-------|----------|-------|
| `type` | StreamEventType | Event type enum |
| `sequence_id` | int | Event ordering |
| `timestamp` | float | Event timestamp |
| `content_delta` | str | Incremental content |
| `tool_call_delta` | ToolCallDelta | Tool call increments |
| `finish_reason` | str | Completion reason |
| `content_index` | int | Content block index |
| `error` | Error | Error information |

### ❌ Missing
| Field | Purpose | Priority |
|-------|---------|----------|
| `usage` | Streaming usage stats | Medium |
| `logprobs` | Log probabilities in stream | Medium |

---

## Priority-Based Implementation Plan

### 🔴 High Priority (Critical for feature parity)
1. **logit_bias** - Essential for controlling token probabilities
   - Add to GenerationParameters
   - Type: Dict[str, float]
   - Maps token IDs to bias values [-100, 100]

2. **response_format** - Essential for JSON mode
   - Add to GenerationParameters or CoreRequest
   - Type: Dict[str, Any]
   - Supports: {"type": "json_object"} or {"type": "text"}

3. **parallel_tool_calls** - Critical for tool handling
   - Add to CoreRequest
   - Type: bool
   - Allow/disallow multiple tools in single response

### 🟡 Medium Priority (Important features)
1. **reasoning_content** - OpenAI o1 model support
   - Add to ResponseMessage
   - Type: Optional[str]
   - Thinking content from reasoning models

2. **thinking_blocks** - Structured reasoning (o1 feature)
   - Add to ResponseMessage
   - Type: Optional[List[Dict[str, Any]]]

3. **logprobs / top_logprobs** - Advanced analysis
   - Add to GenerationParameters
   - logprobs: bool, top_logprobs: int

4. **n** - Multiple completions
   - Add to GenerationParameters
   - Type: int
   - Number of different chat completions

5. **user** - Usage tracking
   - Add to CoreRequest
   - Type: Optional[str]
   - End-user identifier

### 🟢 Low Priority (Nice-to-have)
1. **max_completion_tokens** - Alternative to max_tokens
   - Add to GenerationParameters
   - Newer API version

2. **stream_options** - Streaming configuration
   - Add to GenerationParameters
   - Type: Dict[str, Any]

3. **service_tier**, **safety_identifier** - Advanced settings
   - Can be added to metadata for now

4. **function_call, functions** - Deprecated but supported
   - Can be added but deprecated

---

## Impact Assessment

### Files to Modify
1. **src/transllm/ir/schema.py** - Auto-generated, regenerate from OpenAPI
2. **src/transllm/ir/openapi.yaml** - Main schema definition
3. **src/transllm/adapters/openai.py** - Adapter logic
4. **src/transllm/ir/aliases.py** - Provider field mappings
5. **tests/test_idempotency_openai.py** - Validation tests

### Breaking Changes
- ⚠️ GenerationParameters schema changes (all changes are additive/optional)
- ⚠️ CoreRequest schema changes (all changes are additive/optional)
- ⚠️ ResponseMessage schema changes (all changes are additive/optional)
- ✅ No breaking changes - all new fields are Optional

### Backward Compatibility
All new fields are optional, ensuring full backward compatibility with existing code.

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Total litellm OpenAI params | 24 |
| Already implemented | 13 |
| Missing request params | 9 |
| Missing response fields | 2 |
| Missing stream fields | 2 |

**Feature Parity: 54.2%** → Target: 90%+ with high priority items
