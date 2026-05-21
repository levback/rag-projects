"""Unit tests for LLM clients and model factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.base_llm import LLMConfig, Message
from src.core.model_factory import ModelProvider, create_llm


# ─── BaseLLM / Message / LLMConfig ───────────────────────────────────────────

class TestMessage:
    def test_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(model="gpt-4o")
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.stream is False

    def test_custom_values(self):
        cfg = LLMConfig(model="claude-3-5-sonnet-20241022", temperature=0.0, max_tokens=512)
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512


# ─── GPTClient ────────────────────────────────────────────────────────────────

class TestGPTClient:
    @patch("src.core.gpt_client.OpenAI")
    @patch("src.core.gpt_client.AsyncOpenAI")
    def test_complete_returns_response(self, mock_async, mock_sync):
        from src.core.gpt_client import GPTClient

        # Build a minimal mock matching the OpenAI response shape
        mock_choice = MagicMock()
        mock_choice.message.content = "Paris"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15

        mock_sync.return_value.chat.completions.create.return_value = mock_resp

        client = GPTClient(LLMConfig(model="gpt-4o"), api_key="sk-test")
        result = client.complete([Message(role="user", content="Capital of France?")])

        assert result.content == "Paris"
        assert result.model == "gpt-4o"
        assert result.usage["total_tokens"] == 15

    @patch("src.core.gpt_client.OpenAI")
    @patch("src.core.gpt_client.AsyncOpenAI")
    def test_chat_convenience(self, mock_async, mock_sync):
        from src.core.gpt_client import GPTClient

        mock_choice = MagicMock()
        mock_choice.message.content = "42"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 1
        mock_resp.usage.total_tokens = 6

        mock_sync.return_value.chat.completions.create.return_value = mock_resp

        client = GPTClient(LLMConfig(model="gpt-4o"), api_key="sk-test")
        answer = client.chat("What is 6x7?")
        assert answer == "42"


# ─── ClaudeClient ─────────────────────────────────────────────────────────────

class TestClaudeClient:
    @patch("src.core.claude_client.anthropic.Anthropic")
    @patch("src.core.claude_client.anthropic.AsyncAnthropic")
    def test_complete_returns_response(self, mock_async, mock_sync):
        from src.core.claude_client import ClaudeClient

        mock_content = MagicMock()
        mock_content.text = "Lyon"
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_resp.model = "claude-3-5-sonnet-20241022"
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 8
        mock_resp.usage.output_tokens = 3

        mock_sync.return_value.messages.create.return_value = mock_resp

        client = ClaudeClient(
            LLMConfig(model="claude-3-5-sonnet-20241022"), api_key="ant-test"
        )
        result = client.complete([Message(role="user", content="Second city of France?")])
        assert result.content == "Lyon"
        assert result.finish_reason == "end_turn"

    @patch("src.core.claude_client.anthropic.Anthropic")
    @patch("src.core.claude_client.anthropic.AsyncAnthropic")
    def test_system_prompt_separated(self, mock_async, mock_sync):
        from src.core.claude_client import ClaudeClient, _split_system

        messages = [
            Message(role="system", content="Be concise."),
            Message(role="user", content="Hello"),
        ]
        system, turns = _split_system(messages)
        assert system == "Be concise."
        assert len(turns) == 1
        assert turns[0]["role"] == "user"


# ─── ModelFactory ─────────────────────────────────────────────────────────────

class TestModelFactory:
    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm("invalid_provider")

    @patch("src.core.gpt_client.OpenAI")
    @patch("src.core.gpt_client.AsyncOpenAI")
    def test_create_openai(self, mock_async, mock_sync):
        llm = create_llm("openai", model="gpt-4o-mini")
        from src.core.gpt_client import GPTClient

        assert isinstance(llm, GPTClient)

    @patch("src.core.claude_client.anthropic.Anthropic")
    @patch("src.core.claude_client.anthropic.AsyncAnthropic")
    def test_create_anthropic(self, mock_async, mock_sync):
        llm = create_llm("anthropic")
        from src.core.claude_client import ClaudeClient

        assert isinstance(llm, ClaudeClient)

    def test_provider_enum_values(self):
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.LOCAL.value == "local"
