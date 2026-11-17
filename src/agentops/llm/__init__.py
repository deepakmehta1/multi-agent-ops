"""The model interface, its OpenAI implementation and a scripted fake for tests."""

from agentops.llm.base import (
    Completion,
    LLMClient,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = ["Completion", "LLMClient", "Message", "ToolCall", "ToolSpec", "Usage"]
