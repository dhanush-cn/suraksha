"""ChatService -- the RAG orchestrator.

Flow for one ``/api/chat`` request:

  1. Embed the user question (LLMClient.embed)
  2. Retrieve top-K alerts by cosine distance (retrieval.top_k_similar)
  3. Build a prompt: system instruction + retrieved context + question
  4. Stream the LLM answer back to the caller (LLMClient.stream_chat)

The service ITSELF is an async generator that yields text chunks; the
main.py handler wraps it in StreamingResponse. Keeping the streaming
inside the service means the metadata (retrieved alert ids, model
name) can be emitted as a JSON header event BEFORE the answer body,
letting the frontend show "found N relevant alerts" while the LLM
composes the actual sentence.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.rag.client import LLMClient, LLMConfigurationError
from app.rag.retrieval import RetrievedAlert, top_k_similar

logger = logging.getLogger(__name__)

# Kept as a module constant so eval / grading can compare the exact
# system prompt across runs. The specific wording deliberately says
# "grounded in ... retrieved alerts" so the LLM leans on the context
# and doesn't hallucinate mine names from its training data.
SYSTEM_PROMPT = (
    "You are RockfallGuard's on-call assistant. Answer the operator's "
    "question about recent mine alert activity, grounded ONLY in the "
    "retrieved alert history provided below. If the retrieved context "
    "does not contain enough information, say so plainly -- do not "
    "invent mine names, dates, or sensor readings. Prefer concise, "
    "operator-actionable phrasing over exhaustive detail."
)


def _build_context_block(retrieved: list[RetrievedAlert]) -> str:
    """Format retrieval hits into the ``## Context`` block the LLM sees.

    Numbered, one alert per line, with a similarity score so the model
    can implicitly downweight weaker matches. Kept deterministic so
    the same retrieval produces the same prompt (helps A/B evals).
    """
    if not retrieved:
        return "## Retrieved alerts\n(none available)"
    lines = ["## Retrieved alerts (most relevant first)"]
    for i, hit in enumerate(retrieved, start=1):
        lines.append(f"{i}. [similarity {hit.similarity:.2f}] {hit.source_text}")
    return "\n".join(lines)


class ChatService:
    def __init__(self, session: AsyncSession, client: LLMClient | None = None) -> None:
        self._session = session
        self._client = client or LLMClient()

    async def stream(self, question: str) -> AsyncIterator[str]:
        """Yield the SSE-shaped chunks for one chat request.

        Emits, in order:

          data: {"type": "metadata", "retrieved": [...], "model": "..."}
          data: {"type": "content",  "delta": "First "}
          data: {"type": "content",  "delta": "word "}
          ...
          data: {"type": "done"}

        Each ``data:`` line ends with ``\\n\\n`` per the SSE spec.
        JSON-in-SSE (rather than raw text) so the frontend can
        distinguish metadata from content and render a "sources"
        panel independent of the streaming answer.
        """
        settings = get_settings()

        # 1. Embed the question. Wrap the config failure so the /chat
        # handler can translate to a 503 with a clear message.
        try:
            question_embedding = await self._client.embed(question)
        except LLMConfigurationError:
            yield _sse({
                "type": "error",
                "error": "LLM not configured (LLM_API_KEY missing).",
            })
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding failed: %s", exc)
            yield _sse({"type": "error", "error": f"embedding failed: {exc}"})
            return

        # 2. Retrieve top-K. Empty list on SQLite / retrieval outage
        # is a soft failure -- we still call the LLM, it just tells
        # the user we don't have grounded context.
        retrieved = await top_k_similar(
            session=self._session,
            query_embedding=question_embedding,
            k=settings.rag_top_k,
        )

        # 3. Emit metadata FIRST so the client can render "sources"
        # while the LLM composes the actual answer.
        yield _sse(
            {
                "type": "metadata",
                "retrieved": [
                    {
                        "alert_id": hit.alert_id,
                        "similarity": round(hit.similarity, 3),
                    }
                    for hit in retrieved
                ],
                "model": settings.llm_chat_model,
            }
        )

        # 4. Build prompt + stream the answer.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{_build_context_block(retrieved)}\n\n## Question\n{question}"},
        ]
        try:
            async for delta in self._client.stream_chat(messages):
                yield _sse({"type": "content", "delta": delta})
        except LLMConfigurationError:
            yield _sse({"type": "error", "error": "LLM not configured."})
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat stream failed: %s", exc)
            yield _sse({"type": "error", "error": f"chat stream failed: {exc}"})
            return

        yield _sse({"type": "done"})


def _sse(payload: dict) -> str:
    """Format one JSON payload as an SSE ``data:`` line ending in \\n\\n."""
    return f"data: {json.dumps(payload)}\n\n"


__all__ = ["SYSTEM_PROMPT", "ChatService"]
