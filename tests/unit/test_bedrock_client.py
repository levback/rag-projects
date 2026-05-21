"""Unit tests for BedrockClient and model_factory Bedrock path."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.core.base_llm import LLMConfig, LLMResponse, Message
from src.core.bedrock_client import BedrockClient, _to_bedrock_messages
from src.core.model_factory import ModelProvider, _resolve_bedrock_kwargs, create_llm


# ── _to_bedrock_messages ──────────────────────────────────────────────────────

class TestToBedrickMessages:
    def test_user_only(self):
        msgs = [Message(role="user", content="Hello")]
        system, converted = _to_bedrock_messages(msgs)
        assert system == ""
        assert converted == [{"role": "user", "content": [{"text": "Hello"}]}]

    def test_system_extracted(self):
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Tell me about AI."),
        ]
        system, converted = _to_bedrock_messages(msgs)
        assert system == "You are helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_assistant_included(self):
        msgs = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello there."),
            Message(role="user", content="How are you?"),
        ]
        system, converted = _to_bedrock_messages(msgs)
        assert system == ""
        assert len(converted) == 3
        assert converted[1]["role"] == "assistant"

    def test_empty_messages(self):
        system, converted = _to_bedrock_messages([])
        assert system == ""
        assert converted == []


# ── BedrockClient helpers (mocked boto3) ──────────────────────────────────────

def _make_mock_bedrock_client():
    """Build a BedrockClient with a fully mocked boto3 bedrock-runtime client."""
    mock_boto_client = MagicMock()
    with patch("src.core.bedrock_client.BedrockClient._build_boto_client", return_value=mock_boto_client):
        client = BedrockClient(
            LLMConfig(model="anthropic.claude-3-5-sonnet-20241022-v2:0"),
            region_name="us-east-1",
        )
    client._client = mock_boto_client
    return client, mock_boto_client


class TestBedrockClientComplete:
    def test_complete_returns_llm_response(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "The answer is 42."}]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "stopReason": "end_turn",
        }

        result = client.complete([Message(role="user", content="What is the answer?")])

        assert isinstance(result, LLMResponse)
        assert result.content == "The answer is 42."
        assert result.usage["total_tokens"] == 15
        assert result.finish_reason == "end_turn"
        assert result.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_complete_with_system_message(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse.return_value = {
            "output": {"message": {"content": [{"text": "Sure."}]}},
            "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
            "stopReason": "end_turn",
        }

        messages = [
            Message(role="system", content="Be concise."),
            Message(role="user", content="Summarise AI."),
        ]
        client.complete(messages)

        call_kwargs = mock_boto.converse.call_args[1]
        assert "system" in call_kwargs
        assert call_kwargs["system"][0]["text"] == "Be concise."

    def test_complete_multiple_content_blocks_concatenated(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Part one."}, {"text": " Part two."}]
                }
            },
            "usage": {},
            "stopReason": "stop",
        }

        result = client.complete([Message(role="user", content="Go")])
        assert result.content == "Part one. Part two."

    def test_complete_missing_usage_defaults_to_zero(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "stop",
        }

        result = client.complete([Message(role="user", content="test")])
        assert result.usage["total_tokens"] == 0


class TestBedrockClientStream:
    def test_stream_yields_tokens(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse_stream.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockDelta": {"delta": {"text": " world"}}},
                {"messageStop": {}},
            ]
        }

        tokens = list(client.stream([Message(role="user", content="Hi")]))
        assert tokens == ["Hello", " world"]

    def test_stream_with_system_message(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse_stream.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hi"}}}
            ]
        }

        messages = [
            Message(role="system", content="Be friendly."),
            Message(role="user", content="Hello"),
        ]
        tokens = list(client.stream(messages))
        call_kwargs = mock_boto.converse_stream.call_args[1]
        assert "system" in call_kwargs
        assert tokens == ["Hi"]

    def test_stream_empty_events_skipped(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse_stream.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": ""}}},  # empty text skipped
                {"metadata": {}},                                 # non-delta skipped
                {"contentBlockDelta": {"delta": {"text": "ok"}}},
            ]
        }

        tokens = list(client.stream([Message(role="user", content="test")]))
        assert tokens == ["ok"]


class TestBedrockClientAsync:
    def test_acomplete_calls_complete(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse.return_value = {
            "output": {"message": {"content": [{"text": "async result"}]}},
            "usage": {},
            "stopReason": "stop",
        }

        result = asyncio.run(
            client.acomplete([Message(role="user", content="async?")])
        )
        assert result.content == "async result"

    def test_astream_yields_tokens(self):
        client, mock_boto = _make_mock_bedrock_client()

        mock_boto.converse_stream.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "a"}}},
                {"contentBlockDelta": {"delta": {"text": "b"}}},
            ]
        }

        async def collect():
            tokens = []
            async for t in client.astream([Message(role="user", content="go")]):
                tokens.append(t)
            return tokens

        tokens = asyncio.run(collect())
        assert tokens == ["a", "b"]


class TestBedrockClientBuildBotoClient:
    def test_explicit_credentials_create_session(self):
        """Explicit key+secret must pass them to boto3.Session."""
        with patch("boto3.Session") as mock_session_cls:
            mock_session_cls.return_value.client.return_value = MagicMock()
            BedrockClient._build_boto_client(
                region_name="us-west-2",
                profile_name=None,
                aws_access_key_id="AKID",
                aws_secret_access_key="SECRET",
                aws_session_token=None,
            )
        call_kwargs = mock_session_cls.call_args[1]
        assert call_kwargs["aws_access_key_id"] == "AKID"
        assert call_kwargs["aws_secret_access_key"] == "SECRET"

    def test_named_profile_creates_session(self):
        with patch("boto3.Session") as mock_session_cls:
            mock_session_cls.return_value.client.return_value = MagicMock()
            BedrockClient._build_boto_client(
                region_name="us-east-1",
                profile_name="my-profile",
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_session_token=None,
            )
        call_kwargs = mock_session_cls.call_args[1]
        assert call_kwargs["profile_name"] == "my-profile"

    def test_default_chain_uses_region_only(self):
        with patch("boto3.Session") as mock_session_cls:
            mock_session_cls.return_value.client.return_value = MagicMock()
            BedrockClient._build_boto_client(
                region_name="eu-west-1",
                profile_name=None,
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_session_token=None,
            )
        call_kwargs = mock_session_cls.call_args[1]
        assert call_kwargs["region_name"] == "eu-west-1"
        assert "profile_name" not in call_kwargs

    def test_missing_boto3_raises_import_error(self):
        import builtins
        real_import = builtins.__import__

        def _block_boto3(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("mocked missing boto3")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_boto3):
            with pytest.raises(ImportError, match="boto3 is required"):
                BedrockClient._build_boto_client(
                    region_name="us-east-1",
                    profile_name=None,
                    aws_access_key_id=None,
                    aws_secret_access_key=None,
                    aws_session_token=None,
                )


class TestBedrockInferenceConfig:
    def test_default_inference_config(self):
        client, _ = _make_mock_bedrock_client()
        cfg = client._inference_config()
        assert cfg["maxTokens"] == client.config.max_tokens
        assert cfg["temperature"] == client.config.temperature
        assert "topP" not in cfg  # top_p == 1.0 by default

    def test_top_p_included_when_not_one(self):
        with patch("src.core.bedrock_client.BedrockClient._build_boto_client", return_value=MagicMock()):
            client = BedrockClient(
                LLMConfig(model="x", top_p=0.9),
            )
        cfg = client._inference_config()
        assert "topP" in cfg
        assert cfg["topP"] == pytest.approx(0.9)


# ── model_factory — Bedrock-specific paths ────────────────────────────────────

class TestResolveBedrickKwargs:
    def test_splits_bedrock_keys_from_config_extra(self):
        extra = {"region_name": "eu-west-1", "temperature": 0.5, "custom_key": "x"}
        bedrock_kw, config_extra = _resolve_bedrock_kwargs(extra)
        assert bedrock_kw["region_name"] == "eu-west-1"
        assert "custom_key" in config_extra
        assert "temperature" in config_extra
        assert "temperature" not in bedrock_kw

    def test_falls_back_to_env_for_region(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
        bedrock_kw, _ = _resolve_bedrock_kwargs({})
        assert bedrock_kw["region_name"] == "ap-southeast-1"

    def test_uses_aws_region_env_fallback(self, monkeypatch):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.setenv("AWS_REGION", "ca-central-1")
        bedrock_kw, _ = _resolve_bedrock_kwargs({})
        assert bedrock_kw["region_name"] == "ca-central-1"

    def test_default_region_us_east_1(self, monkeypatch):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        bedrock_kw, _ = _resolve_bedrock_kwargs({})
        assert bedrock_kw["region_name"] == "us-east-1"

    def test_aws_profile_from_env(self, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "my-env-profile")
        bedrock_kw, _ = _resolve_bedrock_kwargs({})
        assert bedrock_kw["profile_name"] == "my-env-profile"

    def test_no_profile_env_key_not_set(self, monkeypatch):
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        bedrock_kw, _ = _resolve_bedrock_kwargs({})
        assert "profile_name" not in bedrock_kw


class TestCreateLLMBedrock:
    def test_create_bedrock_returns_bedrock_client(self, monkeypatch):
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)

        with patch("boto3.Session") as mock_session_cls:
            mock_session_cls.return_value.client.return_value = MagicMock()
            llm = create_llm("bedrock", model="anthropic.claude-3-haiku-20240307-v1:0")

        assert isinstance(llm, BedrockClient)
        assert llm.config.model == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_create_bedrock_passes_region(self, monkeypatch):
        monkeypatch.delenv("AWS_PROFILE", raising=False)

        with patch("boto3.Session") as mock_session_cls:
            mock_session_cls.return_value.client.return_value = MagicMock()
            llm = create_llm("bedrock", region_name="eu-central-1")

        assert llm._region == "eu-central-1"

    def test_bedrock_enum_value(self):
        assert ModelProvider.BEDROCK.value == "bedrock"
