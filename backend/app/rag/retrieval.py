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

from app.schemas.auth import TenantScope

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
    scope: TenantScope,
) -> list[RetrievedAlert]:
    """Return the K most similar alerts by cosine distance, or an empty
    list on SQLite (no pgvector) or on any error.

    Scope-filtered: an operator's chat cannot retrieve context from
    another mine's alerts. Filter runs INSIDE the SQL WHERE so the
    pgvector search is bounded to visible rows -- filtering after the
    fact would let another mine's row rank #1 and get sent to the LLM
    even if we then dropped it, wasting tokens and risking a leak
    through prompt-injection edge cases.

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
    # Scope filter: an admin sees every alert's embedding; an operator
    # sees only their own mine's. Enforced at the JOIN so pgvector's
    # HNSW index still gets a bounded row set to search.
    stmt = text(
        """
        SELECT ae.alert_id,
               ae.source_text,
               ae.embedding <=> CAST(:query AS vector) AS distance
        FROM alert_embeddings ae
        JOIN alert_logs a ON a.id = ae.alert_id
        WHERE :is_admin OR a.mine_id = :mine_id
        ORDER BY distance
        LIMIT :k
        """
    )
    try:
        rows = (
            await session.execute(
                stmt,
                {
                    "query": _to_vector_literal(query_embedding),
                    "k": k,
                    "is_admin": scope.is_admin,
                    # 0 is a safe filler when is_admin=True (never
                    # compared) but must be an int for the SQL bind.
                    "mine_id": scope.mine_id if scope.mine_id is not None else 0,
                },
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
