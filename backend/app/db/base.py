"""Declarative base + shared column primitives."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from sqlalchemy import DateTime, Integer, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, mapped_column


class UtcDateTime(TypeDecorator):
    """DateTime that guarantees tz-aware UTC on both write AND read.

    Postgres+TIMESTAMPTZ preserves the timezone natively. SQLite stores
    the ISO string and returns a *naive* datetime -- which then compares
    unequal to any tz-aware value and blows up ``datetime - datetime``
    with ``TypeError: can't subtract offset-naive and offset-aware``.
    This decorator round-trips UTC explicitly:

    * On write: any naive value is assumed to already be UTC (we
      control every writer via _utcnow); any tz-aware value is
      converted to UTC.
    * On read: naive values are stamped with tzinfo=UTC so the
      returned object always compares cleanly against ``datetime.now
      (timezone.utc)``.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):  # noqa: ARG002
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):  # noqa: ARG002
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


def _utcnow() -> datetime:
    """Timezone-aware UTC now.

    We store timezone-aware datetimes end to end -- Postgres does the
    right thing with TIMESTAMPTZ, and sqlite's aiosqlite driver just
    stores the ISO string. Never call ``datetime.utcnow()``: it returns
    naive datetimes that silently compare unequal to tz-aware ones and
    are a common source of "off by hours" bugs when the app moves
    servers.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base.

    All ORM models inherit from this. Kept separate from any specific
    engine/session so Alembic's ``env.py`` can import ``Base.metadata``
    without pulling in the runtime engine (which needs env vars set).
    """


# Convenience column types -- Annotated aliases so per-column ``Mapped``
# declarations stay short. Pattern from the SQLAlchemy 2.0 docs.
IntPK = Annotated[int, mapped_column(Integer, primary_key=True, autoincrement=True)]
TimestampTz = Annotated[datetime, mapped_column(UtcDateTime())]


__all__ = ["Base", "IntPK", "TimestampTz", "UtcDateTime", "_utcnow"]
