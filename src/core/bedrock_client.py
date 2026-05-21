"""Amazon Bedrock client using the unified Converse API."""
from __future__ import annotations

import logging
from typing import AsyncIterator, Iterator

from src.core.base_llm import BaseLLM, LLMConfig, LLMResponse, Message

logger = logging.getLogger(__name__)

# ── Message conversion ────────────────────────────────────────────────────────

def _to_bedrock_messages(messages: list[Message]) -> tuple[str, list[dict]]:
    """Separate the optional system prompt and convert to Bedrock message format.

    Returns:
        ``(system_text, bedrock_messages)`` where ``system_text`` is empty string
        when no system message was found.
    """
    system_text = ""
    bedrock_msgs: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_text = m.content
        else:
            bedrock_msgs.append(
                {"role": m.role, "content": [{"text": m.content}]}
            )
    return system_text, bedrock_msgs


class BedrockClient(BaseLLM):
    """Wraps the AWS Bedrock Converse / ConverseStream APIs.

    Supports every foundation model available on Bedrock (Anthropic Claude,
    Amazon Nova / Titan, Meta Llama, Mistral, Cohere, AI21 …) through the
    same unified interface as the other LLM clients in this project.

    Authentication — resolved in priority order:
    1. **Explicit keys**: pass ``aws_access_key_id`` + ``aws_secret_access_key``
       (and optionally ``aws_session_token``) directly to the constructor.
    2. **Named CLI profile**: pass ``profile_name`` to use a specific
       ``~/.aws/credentials`` / ``~/.aws/config`` profile.
    3. **Default boto3 credential chain** (no arguments required): environment
       variables → ``~/.aws/credentials`` → ``~/.aws/config`` → IAM instance
       role / ECS task role / … — identical to the AWS CLI default behaviour.

    Requires: ``pip install boto3``

    Example (default chain — same as AWS CLI)::

        client = BedrockClient(LLMConfig(model="anthropic.claude-3-5-sonnet-20241022-v2:0"))
        response = client.chat("Summarise this document.")

    Example (named CLI profile)::

        client = BedrockClient(config, profile_name="my-dev-profile")

    Example (explicit API key pair)::

        client = BedrockClient(
            config,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    """

    def __init__(
        self,
        config: LLMConfig,
        region_name: str = "us-east-1",
        profile_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ) -> None:
        super().__init__(config)
        self._region = region_name
        self._client = self._build_boto_client(
            region_name=region_name,
            profile_name=profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

    # ── boto3 client construction ─────────────────────────────────────────────

    @staticmethod
    def _build_boto_client(
        region_name: str,
        profile_name: str | None,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        aws_session_token: str | None,
    ):
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for BedrockClient. "
                "Install with: pip install boto3"
            ) from exc

        if aws_access_key_id and aws_secret_access_key:
            # Explicit credentials — highest priority
            logger.debug("BedrockClient: using explicit API key credentials")
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
        elif profile_name:
            # Named AWS CLI profile
            logger.debug("BedrockClient: using profile '%s'", profile_name)
            session = boto3.Session(
                profile_name=profile_name,
                region_name=region_name,
            )
        else:
            # Default boto3 credential chain (env vars, ~/.aws/credentials, IAM role …)
            logger.debug("BedrockClient: using default boto3 credential chain")
            session = boto3.Session(region_name=region_name)

        return session.client("bedrock-runtime")

    # ── Inference config helper ───────────────────────────────────────────────

    def _inference_config(self) -> dict:
        cfg: dict = {
            "maxTokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.top_p != 1.0:
            cfg["topP"] = self.config.top_p
        return cfg

    # ── Sync ─────────────────────────────────────────────────────────────────

    def complete(self, messages: list[Message]) -> LLMResponse:
        self._logger.debug(
            "Sending %d messages to Bedrock model %s", len(messages), self.config.model
        )
        system_text, bedrock_msgs = _to_bedrock_messages(messages)

        kwargs: dict = dict(
            modelId=self.config.model,
            messages=bedrock_msgs,
            inferenceConfig=self._inference_config(),
        )
        if system_text:
            kwargs["system"] = [{"text": system_text}]

        response = self._client.converse(**kwargs)

        content = response["output"]["message"]["content"]
        text = "".join(block.get("text", "") for block in content)
        usage = response.get("usage", {})

        return LLMResponse(
            content=text,
            model=self.config.model,
            usage={
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            },
            finish_reason=response.get("stopReason", "stop"),
            raw=response,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self._logger.debug("Streaming from Bedrock model %s", self.config.model)
        system_text, bedrock_msgs = _to_bedrock_messages(messages)

        kwargs: dict = dict(
            modelId=self.config.model,
            messages=bedrock_msgs,
            inferenceConfig=self._inference_config(),
        )
        if system_text:
            kwargs["system"] = [{"text": system_text}]

        response = self._client.converse_stream(**kwargs)
        for event in response.get("stream", []):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield text

    # ── Async ─────────────────────────────────────────────────────────────────
    # boto3 is synchronous; we delegate to a thread executor.

    async def acomplete(self, messages: list[Message]) -> LLMResponse:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, messages)

    async def astream(self, messages: list[Message]) -> AsyncIterator[str]:
        import asyncio

        loop = asyncio.get_event_loop()
        tokens: list[str] = await loop.run_in_executor(
            None, lambda: list(self.stream(messages))
        )
        for token in tokens:
            yield token
