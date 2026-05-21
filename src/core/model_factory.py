"""Factory that selects and instantiates the correct LLM backend."""
from __future__ import annotations

import logging
import os
from enum import Enum

from src.core.base_llm import BaseLLM, LLMConfig

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    BEDROCK = "bedrock"


# Map provider → (module path, class name) for lazy imports
_PROVIDER_MAP: dict[ModelProvider, tuple[str, str]] = {
    ModelProvider.OPENAI: ("src.core.gpt_client", "GPTClient"),
    ModelProvider.ANTHROPIC: ("src.core.claude_client", "ClaudeClient"),
    ModelProvider.LOCAL: ("src.core.local_llm", "LocalLLM"),
    ModelProvider.BEDROCK: ("src.core.bedrock_client", "BedrockClient"),
}

# Reasonable defaults per provider
_DEFAULT_MODELS: dict[ModelProvider, str] = {
    ModelProvider.OPENAI: "gpt-4o",
    ModelProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    ModelProvider.LOCAL: "llama-3.1-8b-instruct",
    ModelProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
}

# Bedrock-specific constructor kwargs extracted from **extra before building LLMConfig
_BEDROCK_INIT_KWARGS = {
    "region_name",
    "profile_name",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
}


def _resolve_api_key(provider: ModelProvider) -> str | None:
    """Return the API key from environment variables, or None for credential-chain providers."""
    env_map = {
        ModelProvider.OPENAI: "OPENAI_API_KEY",
        ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var is None:
        return None
    key = os.environ.get(env_var)
    if not key:
        logger.warning("Environment variable %s is not set.", env_var)
    return key


def _resolve_bedrock_kwargs(extra: dict) -> tuple[dict, dict]:
    """Split *extra* into Bedrock constructor kwargs and LLMConfig.extra.

    Bedrock credentials / region are passed to :class:`BedrockClient.__init__`
    directly; anything else stays in ``LLMConfig.extra``.
    Falls back to environment variables for region and profile when not supplied.

    Returns:
        ``(bedrock_kwargs, config_extra)``
    """
    bedrock_kwargs: dict = {}
    config_extra: dict = {}
    for key, value in extra.items():
        if key in _BEDROCK_INIT_KWARGS:
            bedrock_kwargs[key] = value
        else:
            config_extra[key] = value

    # Fill from environment when caller did not explicitly pass them
    if "region_name" not in bedrock_kwargs:
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        bedrock_kwargs["region_name"] = region
    if "profile_name" not in bedrock_kwargs:
        profile = os.environ.get("AWS_PROFILE")
        if profile:
            bedrock_kwargs["profile_name"] = profile

    return bedrock_kwargs, config_extra


def create_llm(
    provider: str | ModelProvider,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
    **extra,
) -> BaseLLM:
    """Instantiate an LLM client for *provider*.

    Args:
        provider: One of ``"openai"``, ``"anthropic"``, ``"local"``, or ``"bedrock"``.
        model: Model identifier; uses a sensible default when omitted.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        stream: Whether to enable streaming by default.
        **extra: Additional provider-specific kwargs. For Bedrock these may include
            ``region_name``, ``profile_name``, ``aws_access_key_id``,
            ``aws_secret_access_key``, and ``aws_session_token``.

    Returns:
        A :class:`~src.core.base_llm.BaseLLM` instance.

    Raises:
        ValueError: If *provider* is not recognised.
    """
    try:
        provider_enum = ModelProvider(provider)
    except ValueError:
        valid = [p.value for p in ModelProvider]
        raise ValueError(f"Unknown provider {provider!r}. Choose from: {valid}")

    resolved_model = model or _DEFAULT_MODELS[provider_enum]
    config = LLMConfig(
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        extra=extra,
    )

    module_path, class_name = _PROVIDER_MAP[provider_enum]
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if provider_enum == ModelProvider.BEDROCK:
        bedrock_kwargs, config_extra = _resolve_bedrock_kwargs(extra)
        # Rebuild config without bedrock-specific keys in extra
        config = LLMConfig(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            extra=config_extra,
        )
        instance = cls(config, **bedrock_kwargs)
    elif provider_enum == ModelProvider.LOCAL:
        instance = cls(config)
    else:
        api_key = _resolve_api_key(provider_enum)
        instance = cls(config, api_key=api_key)

    logger.info("Created %s (model=%s)", class_name, resolved_model)
    return instance
