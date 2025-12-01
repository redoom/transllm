"""Utility functions for Gemini adapter"""

from __future__ import annotations

import base64
import mimetypes
import re
import uuid
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse


def is_gemini_3_or_newer(model: str) -> bool:
    """Check if model is Gemini 3.x or newer

    Args:
        model: Model name string

    Returns:
        True if model is Gemini 3.x or newer
    """
    gemini_3_models = ["gemini-2.0", "gemini-3.0", "gemini-exp", "gemini-1.5-pro"]
    return any(model.startswith(prefix) for prefix in gemini_3_models)


def is_gcs_uri(url: str) -> bool:
    """Check if URL is a Google Cloud Storage URI

    Args:
        url: URL to check

    Returns:
        True if URL is a GCS URI
    """
    return url.startswith("gs://")


def is_http_url(url: str) -> bool:
    """Check if URL is HTTP/HTTPS

    Args:
        url: URL to check

    Returns:
        True if URL is HTTP/HTTPS
    """
    parsed = urlparse(url)
    return parsed.scheme in ["http", "https"]


def is_base64_data(url: str) -> bool:
    """Check if URL is base64 data URI

    Args:
        url: URL to check

    Returns:
        True if URL is base64 data URI
    """
    return url.startswith("data:")


def detect_media_type(url: str) -> Optional[str]:
    """Detect media type from URL

    Args:
        url: URL or data URI

    Returns:
        Media type string or None
    """
    if is_base64_data(url):
        # Extract from data URI
        match = re.match(r'data:([^;]+)', url)
        if match:
            return match.group(1)

    # Try to detect from URL extension
    parsed = urlparse(url)
    path = parsed.path
    media_type, _ = mimetypes.guess_type(path)
    return media_type


def convert_image_url_to_gemini(
    image_url: str,
    detail: Optional[str] = None,
    is_vertex: bool = False
) -> Dict[str, Any]:
    """Convert OpenAI image_url to Gemini format

    Args:
        image_url: OpenAI image URL or data URI
        detail: Detail level (low, high, auto)
        is_vertex: Whether using Vertex AI

    Returns:
        Gemini image part dictionary
    """
    if is_base64_data(image_url):
        # Base64 data URI → inline_data
        match = re.match(r'data:([^;]+);base64,(.+)', image_url)
        if match:
            media_type = match.group(1)
            data = match.group(2)
            return {
                "inline_data": {
                    "mime_type": media_type,
                    "data": data
                }
            }
    elif is_gcs_uri(image_url):
        # GCS URI → file_data
        return {
            "file_data": {
                "mime_type": detect_media_type(image_url) or "image/jpeg",
                "file_uri": image_url
            }
        }
    elif is_http_url(image_url):
        # HTTP URL → file_data (Vertex) or base64 (Google AI Studio)
        if is_vertex:
            return {
                "file_data": {
                    "mime_type": detect_media_type(image_url) or "image/jpeg",
                    "file_uri": image_url
                }
            }
        else:
            # Would need to fetch and convert to base64
            raise ValueError(
                "HTTP URLs not supported for Google AI Studio. "
                "Use base64 data URI or GCS URI instead."
            )
    else:
        # Assume local path, try to read as base64
        raise ValueError(
            "Invalid image URL format. Use base64 data URI, GCS URI, or HTTP URL (Vertex only)."
        )

    # Default fallback
    raise ValueError(f"Unable to process image URL: {image_url}")


def map_detail_to_media_resolution(detail: Optional[str]) -> Optional[str]:
    """Map OpenAI detail to Gemini media resolution

    Args:
        detail: OpenAI detail level

    Returns:
        Gemini media resolution or None
    """
    if detail is None:
        return None

    detail_lower = detail.lower()
    if detail_lower == "low":
        return "low"
    elif detail_lower == "high":
        return "high"
    elif detail_lower == "auto":
        return "low"  # Default to low for auto
    else:
        return None


def generate_tool_call_id() -> str:
    """Generate unique tool call ID for Gemini

    Gemini doesn't return tool call IDs in responses, so we need to
    generate UUID-based IDs. For thinking blocks, we encode with
    thoughtSignature.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def encode_thought_signature(tool_call_id: str) -> str:
    """Encode tool call ID as thoughtSignature for thinking blocks

    Args:
        tool_call_id: Tool call ID to encode

    Returns:
        Base64 encoded signature
    """
    # Simple encoding - in litellm this uses a more sophisticated signature
    signature_data = f"toolcall:{tool_call_id}"
    return base64.b64encode(signature_data.encode()).decode()


def decode_thought_signature(signature: str) -> Optional[str]:
    """Decode thoughtSignature to extract tool call ID

    Args:
        signature: Base64 encoded signature

    Returns:
        Original tool call ID or None if invalid
    """
    try:
        decoded = base64.b64decode(signature.encode()).decode()
        if decoded.startswith("toolcall:"):
            return decoded.split(":", 1)[1]
    except Exception:
        pass
    return None


def convert_tool_to_gemini_function(
    tool: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert OpenAI tool to Gemini function declaration

    Args:
        tool: OpenAI tool dictionary
        parameters: Optional schema for parameters

    Returns:
        Gemini function declaration dictionary
    """
    if tool.get("type") == "function":
        function = tool["function"]

        result = {
            "name": function["name"],
            "description": function.get("description", "")
        }

        if "parameters" in function or parameters:
            result["parameters"] = parameters or function["parameters"]

        return result

    raise ValueError(f"Unsupported tool type: {tool.get('type')}")


def convert_gemini_function_to_tool(function_decl: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Gemini function declaration to OpenAI tool

    Args:
        function_decl: Gemini function declaration

    Returns:
        OpenAI tool dictionary
    """
    return {
        "type": "function",
        "function": {
            "name": function_decl["name"],
            "description": function_decl.get("description", ""),
            "parameters": function_decl.get("parameters", {})
        }
    }


def is_candidate_token_count_inclusive(
    prompt_tokens: int,
    candidates_tokens: int,
    total_tokens: int
) -> bool:
    """Check if candidate tokens are included in total token count

    Gemini 3.x models include thinking tokens in the candidate count,
    while Gemini 2.x models may not. This function detects which
    behavior applies to avoid double-counting.

    Args:
        prompt_tokens: Token count for prompt
        candidates_tokens: Token count for candidates
        total_tokens: Total token count

    Returns:
        True if candidate tokens are already included in total
    """
    # If prompt + candidates equals total, they're already counted separately
    if prompt_tokens + candidates_tokens == total_tokens:
        return False
    elif candidates_tokens == total_tokens:
        # If candidates equal total, they include everything
        return True
    else:
        # Default to assuming they're separate (safe approach)
        return False


def merge_duplicate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive messages with the same role

    Gemini prefers fewer, more complete messages rather than
    many small messages.

    Args:
        messages: List of message dictionaries

    Returns:
        Merged messages list
    """
    if not messages:
        return []

    merged = []
    current = messages[0].copy()
    current_parts = []

    # Extract parts if present
    if isinstance(current.get("parts"), list):
        current_parts = current["parts"]
        # Remove parts from current to process separately
        current = {k: v for k, v in current.items() if k != "parts"}

    for msg in messages[1:]:
        if msg.get("role") == current.get("role"):
            # Same role - merge parts
            if isinstance(msg.get("parts"), list):
                current_parts.extend(msg["parts"])
        else:
            # Different role - save current and start new
            if current_parts and "parts" not in current:
                current["parts"] = current_parts
            elif current_parts:
                current["parts"].extend(current_parts)
            merged.append(current)

            current = msg.copy()
            current_parts = []
            if isinstance(current.get("parts"), list):
                current_parts = current["parts"]
                current = {k: v for k, v in current.items() if k != "parts"}

    # Don't forget the last message
    if current_parts and "parts" not in current:
        current["parts"] = current_parts
    elif current_parts:
        current["parts"].extend(current_parts)
    merged.append(current)

    return merged


def validate_gemini_request(request: Dict[str, Any]) -> None:
    """Validate Gemini request for common issues

    Args:
        request: Gemini request dictionary

    Raises:
        ValueError: If request is invalid
    """
    # Check for required fields
    if "contents" not in request:
        raise ValueError("Gemini request must contain 'contents' field")

    contents = request["contents"]
    if not isinstance(contents, list) or not contents:
        raise ValueError("Gemini 'contents' must be a non-empty list")

    # Check for empty parts
    for content in contents:
        if "parts" not in content:
            raise ValueError("Each content must have 'parts' field")
        parts = content["parts"]
        if not isinstance(parts, list) or not parts:
            raise ValueError("Parts must be a non-empty list")

        # Check each part has required fields
        for part in parts:
            if not any(key in part for key in ["text", "inline_data", "file_data", "function_call"]):
                raise ValueError("Each part must have at least one of: text, inline_data, file_data, function_call")


# Beta Headers Detection Functions (based on litellm)

def is_cache_control_set(messages: List[Dict[str, Any]]) -> bool:
    """Check if cache control is set in messages

    Args:
        messages: List of message dictionaries

    Returns:
        True if cache control is detected
    """
    for msg in messages:
        # Check metadata or custom fields for cache control
        metadata = msg.get("metadata", {})
        if isinstance(metadata, dict):
            # Check for cache control indicators
            if "cache_control" in metadata or "cache" in str(metadata).lower():
                return True

        # Check for Anthropic-style cache control
        content = msg.get("content", "")
        if isinstance(content, str):
            if "cache_control" in content.lower():
                return True

        # Check tool calls for cache control
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            if "cache" in str(tool_call).lower():
                return True

    return False


def computer_tool_used(messages: List[Dict[str, Any]]) -> bool:
    """Check if computer tool is used

    Args:
        messages: List of message dictionaries

    Returns:
        True if computer tool is detected
    """
    for msg in messages:
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name", "").lower()
            if any(keyword in tool_name for keyword in ["computer", "desktop", "screenshot", "mouse"]):
                return True

    return False


def file_id_used(messages: List[Dict[str, Any]]) -> bool:
    """Check if file IDs are used

    Args:
        messages: List of message dictionaries

    Returns:
        True if file IDs are detected
    """
    for msg in messages:
        # Check for file references in content
        content = msg.get("content", "")
        if isinstance(content, str):
            # Look for file ID patterns (e.g., file-abc123, file://)
            if "file_" in content or "file://" in content:
                return True

        # Check metadata for file references
        metadata = msg.get("metadata", {})
        if "file" in str(metadata).lower():
            return True

    return False


def mcp_server_used(messages: List[Dict[str, Any]]) -> bool:
    """Check if MCP server is used

    Args:
        messages: List of message dictionaries

    Returns:
        True if MCP server is detected
    """
    for msg in messages:
        # Check for MCP-specific patterns
        metadata = msg.get("metadata", {})
        if "mcp" in str(metadata).lower():
            return True

        # Check tool names for MCP patterns
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name", "").lower()
            if "mcp" in tool_name or "server" in tool_name:
                return True

    return False


def tool_search_used(messages: List[Dict[str, Any]]) -> bool:
    """Check if tool search is used

    Args:
        messages: List of message dictionaries

    Returns:
        True if tool search is detected
    """
    for msg in messages:
        # Check for search-related tool usage
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name", "").lower()
            if any(keyword in tool_name for keyword in ["search", "find", "query", "lookup"]):
                return True

        # Check content for search queries
        content = msg.get("content", "")
        if isinstance(content, str):
            if "search" in content.lower() or "find" in content.lower():
                return True

    return False


def effort_used(messages: List[Dict[str, Any]], request: Dict[str, Any]) -> bool:
    """Check if thinking effort is used

    Args:
        messages: List of message dictionaries
        request: Request dictionary

    Returns:
        True if thinking effort is detected
    """
    # Check for thinking config
    thinking_config = request.get("thinkingConfig")
    if thinking_config:
        if thinking_config.get("includeThoughts") or thinking_config.get("thinkingBudget"):
            return True

    # Check for reasoning effort parameter
    reasoning_effort = request.get("reasoning_effort")
    if reasoning_effort:
        return True

    # Check for thinking blocks in messages
    for msg in messages:
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            # Check if thinking-related tools are used
            function = tool_call.get("function", {})
            tool_name = function.get("name", "").lower()
            if "reasoning" in tool_name or "thinking" in tool_name:
                return True

    return False


def get_gemini_headers(
    messages: List[Dict[str, Any]],
    request: Dict[str, Any],
    is_vertex: bool = False,
    user_betas: Optional[Set[str]] = None
) -> Dict[str, str]:
    """Get Gemini beta headers based on feature usage

    Args:
        messages: List of message dictionaries
        request: Request dictionary
        is_vertex: Whether using Vertex AI
        user_betas: User-provided beta headers

    Returns:
        Dictionary of headers
    """
    headers = {}

    # Only add beta headers for Google AI Studio (not Vertex AI)
    if is_vertex:
        return headers

    # Collect beta features based on usage
    betas = set()

    if is_cache_control_set(messages):
        betas.add("prompt-caching-2024-07-31")

    if computer_tool_used(messages):
        betas.add("computer-use-2024-10-22")

    if file_id_used(messages):
        betas.add("files-api-2025-04-14")

    if mcp_server_used(messages):
        betas.add("mcp-client-2025-04-04")

    if tool_search_used(messages):
        betas.add("tool-search-2025-11-19")

    if effort_used(messages, request):
        betas.add("effort-2025-11-24")

    # Merge user-provided betas
    if user_betas:
        betas.update(user_betas)

    # Add to headers if any betas detected
    if betas:
        headers["anthropic-beta"] = ",".join(sorted(betas))

    return headers


# Empty/Missing Content Error Handling (based on litellm)

def handle_empty_content(
    content: Any,
    default: str = "",
    is_required: bool = False
) -> str:
    """Handle empty or missing content gracefully

    Args:
        content: Content to check
        default: Default value if content is empty
        is_required: Whether content is required

    Returns:
        Processed content string

    Raises:
        ValueError: If content is required but missing
    """
    # Handle None
    if content is None:
        if is_required:
            raise ValueError("Required content is missing")
        return default

    # Handle empty string
    if content == "":
        if is_required:
            raise ValueError("Required content is empty")
        return default

    # Handle list
    if isinstance(content, list):
        if not content:
            if is_required:
                raise ValueError("Required content list is empty")
            return default
        # Check if all items are empty
        if all(not item for item in content):
            return default

    # Handle dict
    if isinstance(content, dict):
        if not content:
            if is_required:
                raise ValueError("Required content dict is empty")
            return default

    # Return as string
    if isinstance(content, str):
        return content
    else:
        return str(content)


def handle_missing_text(text: Optional[str], fallback: str = "") -> str:
    """Handle missing or empty text fields

    Args:
        text: Text to check
        fallback: Fallback text if missing

    Returns:
        Processed text
    """
    if not text or not text.strip():
        return fallback
    return text


def detect_circular_reference(
    obj: Any,
    visited: Optional[Set[int]] = None,
    depth: int = 0,
    max_depth: int = 50
) -> bool:
    """Detect circular references in nested objects

    Args:
        obj: Object to check
        visited: Set of visited object ids
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        True if circular reference detected

    Raises:
        ValueError: If circular reference or max depth exceeded
    """
    if visited is None:
        visited = set()

    if depth > max_depth:
        raise ValueError(f"Max recursion depth ({max_depth}) exceeded")

    obj_id = id(obj)
    if obj_id in visited:
        return True

    if isinstance(obj, (dict, list)):
        visited.add(obj_id)

        try:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if detect_circular_reference(value, visited.copy(), depth + 1, max_depth):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if detect_circular_reference(item, visited.copy(), depth + 1, max_depth):
                        return True
        finally:
            visited.discard(obj_id)

    return False


def validate_empty_properties(schema: Dict[str, Any]) -> None:
    """Validate schema for empty properties

    Args:
        schema: Schema to validate

    Raises:
        ValueError: If schema has empty properties
    """
    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")

    if schema_type == "object":
        properties = schema.get("properties", {})
        if not properties:
            raise ValueError("Object schema cannot have empty properties")

        # Recursively validate properties
        for prop_name, prop_schema in properties.items():
            if not prop_schema:
                raise ValueError(f"Property '{prop_name}' has empty schema")

            # Check for empty required list
            if "required" in prop_schema:
                required = prop_schema["required"]
                if isinstance(required, list) and not required:
                    raise ValueError(f"Property '{prop_name}' has empty required list")

            validate_empty_properties(prop_schema)

    elif schema_type == "array":
        items = schema.get("items")
        if not items:
            # Set default empty object schema
            schema["items"] = {"type": "object"}

        # Recursively validate items
        if items:
            validate_empty_properties(items)

    elif "anyOf" in schema:
        anyof = schema["anyOf"]
        if len(anyof) == 0:
            raise ValueError("anyOf cannot be empty")

        if len(anyof) == 1:
            only_item = anyof[0]
            if isinstance(only_item, dict) and only_item.get("type") == "null":
                raise ValueError("anyOf cannot contain only null type")

        # Recursively validate anyOf items
        for item in anyof:
            validate_empty_properties(item)

    elif "allOf" in schema:
        allof = schema["allOf"]
        if not allof:
            raise ValueError("allOf cannot be empty")

        # Recursively validate allOf items
        for item in allof:
            validate_empty_properties(item)


def safe_get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely get nested dictionary value using dot notation

    Args:
        data: Dictionary to query
        path: Dot-separated path (e.g., "content.parts.0.text")
        default: Default value if path not found

    Returns:
        Value at path or default
    """
    keys = path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            index = int(key)
            if 0 <= index < len(current):
                current = current[index]
            else:
                return default
        else:
            return default

    return current


def validate_request_structure(request: Dict[str, Any]) -> None:
    """Comprehensive validation of Gemini request structure

    Args:
        request: Request to validate

    Raises:
        ValueError: If request structure is invalid
    """
    # Check required top-level fields
    if "contents" not in request:
        raise ValueError("Gemini request must contain 'contents' field")

    contents = request["contents"]
    if not isinstance(contents, list):
        raise ValueError("'contents' must be a list")

    if not contents:
        raise ValueError("'contents' cannot be empty")

    # Validate each content item
    for idx, content in enumerate(contents):
        if not isinstance(content, dict):
            raise ValueError(f"Content item {idx} must be a dict")

        if "role" not in content:
            raise ValueError(f"Content item {idx} missing 'role'")

        if "parts" not in content:
            raise ValueError(f"Content item {idx} missing 'parts'")

        parts = content["parts"]
        if not isinstance(parts, list):
            raise ValueError(f"Content item {idx} 'parts' must be a list")

        if not parts:
            raise ValueError(f"Content item {idx} 'parts' cannot be empty")

        # Validate each part
        for part_idx, part in enumerate(parts):
            if not isinstance(part, dict):
                raise ValueError(
                    f"Content {idx} part {part_idx} must be a dict"
                )

            # Check for valid part types
            valid_keys = {"text", "inline_data", "file_data", "function_call"}
            if not any(key in part for key in valid_keys):
                raise ValueError(
                    f"Content {idx} part {part_idx} must have at least one of: {', '.join(valid_keys)}"
                )

    # Validate tools if present
    if "tools" in request:
        tools = request["tools"]
        if not isinstance(tools, list):
            raise ValueError("'tools' must be a list")

        for tool_idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"Tool {tool_idx} must be a dict")

            if "function_declarations" not in tool:
                raise ValueError(f"Tool {tool_idx} missing 'function_declarations'")

            func_decls = tool["function_declarations"]
            if not isinstance(func_decls, list) or not func_decls:
                raise ValueError(
                    f"Tool {tool_idx} 'function_declarations' must be a non-empty list"
                )

    # Validate system instruction if present
    if "system_instruction" in request:
        sys_inst = request["system_instruction"]
        if not isinstance(sys_inst, dict):
            raise ValueError("'system_instruction' must be a dict")

        if "parts" not in sys_inst:
            raise ValueError("'system_instruction' missing 'parts'")

        sys_parts = sys_inst["parts"]
        if not isinstance(sys_parts, list) or not sys_parts:
            raise ValueError("'system_instruction' 'parts' cannot be empty")

    # Validate generation config if present
    if "generationConfig" in request:
        gen_config = request["generationConfig"]
        if not isinstance(gen_config, dict):
            raise ValueError("'generationConfig' must be a dict")
