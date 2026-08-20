"""ORM: ``alert_embeddings``.

Separate table from ``alert_logs`` (not a column) because:

* Not every alert needs embedding -- we may skip low-risk alerts to
  save token spend.
* Re-embedding after a model swap rewrites this table without
  touching the source-of-truth alert log.
* The vector column type has a fixed dimension; a nullable vector
  on ``alert_logs`` would still consume the dimension in the schema,
  and switching dim on model swap would need a real ALTER on the
  full alert table.

Vector dimension is bound to :attr:`Settings.llm_embedding_dim`;
mismatched writes fail with a ``DataError`` on INSERT, which is the
right failure mode (early + loud) rather than silent index corruption.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base, TimestampTz, _utcnow


def _embedding_column():
    """Deferred import: only builds a pgvector Vector column when the
    package is available (which it always is in this venv, but keeping
    the guard makes ``from app.rag.models import ...`` safe on a
    slim install where pgvector wasn't pip-installed)."""
    from pgvector.sqlalchemy import Vector

    dim = get_settings().llm_embedding_dim
    return mapped_column(Vector(dim), nullable=False)


class AlertEmbedding(Base):
    __tablename__ = "alert_embeddings"

    # PK is also the FK -- one embedding per alert row, no separate
    # surrogate id needed. ON DELETE CASCADE keeps embeddings from
    # outliving their source alert; if operators purge historical
    # alerts, the embeddings go with them (respecting the "audit log
    # is the source of truth" principle from earlier steps).
    alert_id: Mapped[int] = mapped_column(
        ForeignKey("alert_logs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # The exact text the model saw. Stored so re-ranking, prompt
    # inspection, and cache reasoning all use the same string. NOT the
    # user's question -- that's transient per /api/chat call.
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    # The name/version that produced this embedding. Necessary because
    # embedding models produce vectors that are meaningful only within
    # the model that generated them; a query embedded with model A can
    # never be searched against a corpus embedded with model B. When
    # this differs from Settings.llm_embedding_model we know the row
    # needs re-embedding.
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding = _embedding_column()
    created_at: Mapped[TimestampTz] = mapped_column(nullable=False, default=_utcnow)

    alert = relationship("AlertLog", backref="embedding_row")


__all__ = ["AlertEmbedding"]
