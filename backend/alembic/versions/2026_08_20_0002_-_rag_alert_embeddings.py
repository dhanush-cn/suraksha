"""rag: pgvector extension + alert_embeddings table

Revision ID: 0002_rag_embeddings
Revises: 0001_baseline
Create Date: 2026-08-20 00:00:00 UTC

Postgres-only: skips gracefully on SQLite (where the vector type
doesn't exist), so ``alembic upgrade head`` still works against the
dev SQLite database -- the RAG features degrade to "retrieval
returns nothing" there.

Creates:

* The ``vector`` extension if it doesn't already exist. Requires the
  connecting user to have ``CREATE`` privilege on the database; on
  managed Postgres this usually means running one initial ``CREATE
  EXTENSION`` as a superuser -- documented in ``DEPLOYMENT.md``.
* ``alert_embeddings`` table with the FK CASCADE to ``alert_logs``.
* An HNSW index on the embedding column with ``vector_cosine_ops``.
  HNSW (over IVFFlat) because HNSW doesn't need a training corpus,
  performs well up to a few million rows, and is now stable in
  pgvector 0.5+. Bumped to a real IVFFlat with lists tuned when the
  corpus outgrows HNSW's memory footprint.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_rag_embeddings"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Embedding dimension MUST match Settings.llm_embedding_dim on the
# app side. Hardcoded here (rather than read from Settings) because
# Alembic migrations should be replayable years later with the exact
# same DDL -- swapping to a bigger model requires a NEW migration
# (drop + recreate + re-embed), not a code-time value change.
_EMBEDDING_DIM = 1_536


def _is_postgres() -> bool:
    return op.get_context().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        # SQLite dev: skip. The app-side retrieval already checks the
        # dialect and returns empty on non-postgres, so the /api/chat
        # path still works (LLM will say "no relevant history").
        return

    # 1. Extension (idempotent). Some managed Postgres providers
    # (Supabase, Neon) enable this via a UI toggle; the IF NOT EXISTS
    # keeps this a no-op there.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Table.
    op.create_table(
        "alert_embeddings",
        sa.Column("alert_id", sa.Integer(), primary_key=True),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        # Vector column declared via raw SQL so the migration doesn't
        # depend on the ``pgvector`` Python package being importable
        # at Alembic-run time.
        sa.Column(
            "embedding",
            sa.dialects.postgresql.ARRAY(sa.Float()),  # placeholder for autogenerate diffs
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["alert_id"],
            ["alert_logs.id"],
            ondelete="CASCADE",
            name="fk_alert_embeddings_alert_id",
        ),
    )
    # 2a. Now ALTER the column to the actual vector type -- doing it
    # in two steps sidesteps SQLAlchemy's lack of a native vector type
    # (unless pgvector.sqlalchemy is imported, which we're avoiding
    # inside migrations for portability).
    op.execute(f"ALTER TABLE alert_embeddings ALTER COLUMN embedding TYPE vector({_EMBEDDING_DIM}) USING NULL")
    # ``USING NULL`` because the column can't be alter-cast from
    # ARRAY(Float) to vector automatically; the table is empty at
    # this point so NULLing and re-populating is a no-op.
    # Then re-add NOT NULL after the type change.
    op.execute("ALTER TABLE alert_embeddings ALTER COLUMN embedding SET NOT NULL")

    # 3. HNSW index. cosine_ops chosen to match the ``<=>`` operator
    # our retrieval query uses. m + ef_construction take pgvector's
    # defaults (16 / 64) -- good balance for a few hundred thousand
    # rows and small enough to fit in typical shared_buffers.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_alert_embeddings_hnsw "
        "ON alert_embeddings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute("DROP INDEX IF EXISTS ix_alert_embeddings_hnsw")
    op.drop_table("alert_embeddings")
    # Leave the vector extension in place; another migration or
    # feature may depend on it. Dropping an extension is disruptive
    # (invalidates all vector-typed columns anywhere in the DB) and
    # should be an explicit ops action, not a code-driven downgrade.
