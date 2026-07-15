"""Basic unit tests for gateway request/response shaping (no GPU needed)."""
import os

os.environ.setdefault("GATEWAY_ALLOW_NO_AUTH", "1")

from gateway.app import messages_to_prompt, ChatMessage


def test_messages_to_prompt_basic():
    msgs = [ChatMessage(role="user", content="Hello")]
    prompt = messages_to_prompt(msgs)
    assert "<|user|>" in prompt
    assert "Hello" in prompt
    assert prompt.strip().endswith("<|assistant|>")


def test_messages_to_prompt_multiturn():
    msgs = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="Hi"),
    ]
    prompt = messages_to_prompt(msgs)
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
