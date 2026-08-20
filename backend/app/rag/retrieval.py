"""pgvector top-K retrieval + a SQLite fallback that returns nothing.

The interesting query is the pgvector one:

    SELECT alert_id, source_text, embedding <=> :query AS distance
    FROM alert_embeddings
    ORDER BY distance
    LIMIT :k

``<=>`` is the cosine-distance operator (available with
``vector_cosine_ops``); ``<->`` (L2) and ``<#>`` (inner product) are
the alternatives. Cosine is the right choice for the OpenAI /
SentenceTransformers style embeddings we're consuming here -- their
scale is not meaningful, only direction is.

The service degrades gracefully on SQLite (dev / test): the retrieval
call returns an empty list, and ChatService responds with a "no
retrieval available on this backend" note instead of crashing. This
keeps the earlier test suites SQLite-friendly while the RAG feature
runs only against Postgres in prod.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedAlert:
    alert_id: int
    source_text: str
    distance: float  # cosine distance in [0, 2]; 0 = identical direction

    @property
    def similarity(self) -> float:
        """Convenience for prompt building: 1 - distance/2, in [0, 1]."""
        return max(0.0, 1.0 - self.distance / 2.0)


def _is_postgres(session: AsyncSession) -> bool:
    """Dialect check -- pgvector operators only exist on Postgres."""
    return session.bind.dialect.name == "postgresql"  # type: ignore[union-attr]


async def top_k_similar(
    *,
    session: AsyncSession,
    query_embedding: Sequence[float],
    k: int,
) -> list[RetrievedAlert]:
    """Return the K most similar alerts by cosine distance, or an empty
    list on SQLite (no pgvector) or on any error.

    Empty list on error is a deliberate choice: retrieval failure
    should not fail the whole chat request. ChatService detects the
    empty list and prompts the LLM to say "I don't have relevant
    alert history" rather than fabricating.
    """
    if not _is_postgres(session):
        logger.info(
            "pgvector retrieval skipped: dialect is %s, not postgresql",
            session.bind.dialect.name if session.bind else "unknown",
        )
        return []

    # Pass the vector as its Postgres string literal (``[1.0,0.0,...]``)
    # then ``CAST AS vector``. This avoids having to register the
    # pgvector asyncpg codec per-connection, which is fiddly with
    # SQLAlchemy's async pool (needs a "connect" event listener that
    # reaches into ``raw_connection.driver_connection``). The string
    # literal path works with asyncpg, psycopg, and psycopg2 without
    # any codec setup and costs a few bytes of extra wire traffic --
    # negligible next to the vector payload itself.
    stmt = text(
        """
        SELECT alert_id, source_text, embedding <=> CAST(:query AS vector) AS distance
        FROM alert_embeddings
        ORDER BY distance
        LIMIT :k
        """
    )
    try:
        rows = (
            await session.execute(
                stmt,
                {"query": _to_vector_literal(query_embedding), "k": k},
            )
        ).mappings().all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("pgvector retrieval failed: %s", exc)
        return []

    return [
        RetrievedAlert(
            alert_id=int(row["alert_id"]),
            source_text=str(row["source_text"]),
            distance=float(row["distance"]),
        )
        for row in rows
    ]


def _to_vector_literal(embedding: Sequence[float]) -> str:
    """Encode a numeric sequence as pgvector's string literal form.

    pgvector accepts ``[1.0,2.0,3.0]``. ``str(list)`` produces
    ``[1.0, 2.0, 3.0]`` -- the spaces are legal but wasteful. Format
    tight for wire size.
    """
    return "[" + ",".join(f"{float(x):.7g}" for x in embedding) + "]"


__all__ = ["RetrievedAlert", "top_k_similar"]
