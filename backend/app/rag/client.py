"""OpenAI-chat-compatible LLM client.

One code path targets:

* OpenAI (default -- ``https://api.openai.com/v1``)
* Anthropic's OpenAI-compat proxy
* Groq (``https://api.groq.com/openai/v1``)
* Together (``https://api.together.xyz/v1``)
* Local Ollama (``http://localhost:11434/v1``)

The compatibility surface used is deliberately small:

* ``POST /embeddings`` -- ``{model, input}`` -> ``{data: [{embedding: [...]}]}``.
* ``POST /chat/completions`` -- ``{model, messages, stream}`` -> SSE stream
  of ``{choices: [{delta: {content: ...}}]}`` events, terminated by
  ``data: [DONE]``.

Async httpx throughout so the streaming chat call yields to the event
loop between chunks -- other requests keep flowing while a slow LLM
answers.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Sequence

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM is called but no API key is configured."""


class LLMClient:
    """Thin wrapper around the two endpoints ChatService needs.

    Instances are cheap; one per request is fine. Connection reuse
    happens inside the ``httpx.AsyncClient`` we build per call --
    ChatService could later hold a long-lived client for connection
    pooling, but that requires a shutdown hook and isn't worth the
    complexity for the current call rate.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._base_url = s.llm_base_url.rstrip("/")
        self._api_key = s.llm_api_key.get_secret_value() if s.llm_api_key else None
        self._chat_model = s.llm_chat_model
        self._embedding_model = s.llm_embedding_model
        self._timeout = s.llm_request_timeout_seconds

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is not set. /api/chat + the embedder require it. "
                "See DEPLOYMENT.md for the env var contract."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ embeddings
    async def embed(self, text: str) -> list[float]:
        """One-shot single-text embedding. Batch API is intentionally
        omitted -- callers embed one alert or one user question at a
        time; batching adds latency-of-the-slowest-in-the-batch
        without saving much."""
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self._embedding_model, "input": list(texts)}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                f"{self._base_url}/embeddings",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()["data"]
        # Response order matches input order per the OpenAI contract.
        return [entry["embedding"] for entry in data]

    # ------------------------------------------------------------------ chat
    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield content chunks from a streaming chat completion.

        Consumes the OpenAI SSE format:

            data: {"choices":[{"delta":{"content":"..."}}]}
            data: [DONE]

        and yields only the content strings. Non-content deltas (role,
        tool_calls, finish_reason) are dropped -- callers get pure
        text they can concatenate. Skips over the SSE keepalive comment
        lines (``: ping``) that some backends emit.
        """
        payload = {
            "model": self._chat_model,
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for raw in response.aiter_lines():
                    if not raw or raw.startswith(":"):
                        continue  # SSE keepalive / blank
                    if not raw.startswith("data: "):
                        # Unexpected line -- log and continue rather
                        # than raising, so a single format oddity
                        # doesn't kill the stream mid-answer.
                        logger.debug("stream: non-data line %r", raw)
                        continue
                    data = raw[6:]
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("stream: could not parse JSON %r", data)
                        continue
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content


__all__ = ["LLMClient", "LLMConfigurationError"]
