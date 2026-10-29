"""ORM models -- ``mines``, ``alert_logs``, ``users``.

The column shapes are compatible with the raw-sqlite3 tables in
``backend/database.py`` so the two layers can coexist during the
migration. Alembic ``0001_baseline.py`` creates exactly this schema
against a fresh database.

Composite index rationale:

* ``ix_alert_logs_mine_triggered`` on ``(mine_id, triggered_at DESC)``
  matches ``get_recent_alerts``: filter by ``mine_id``, order by
  ``triggered_at`` desc, limit N. Without it, every alert-history read
  is a table scan.
* ``ux_users_username`` (unique) exists so login lookup is an index seek
  and duplicate signups can't slip past a race between two INSERTs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IntPK, TimestampTz, UtcDateTime, _utcnow


class Mine(Base):
    __tablename__ = "mines"

    id: Mapped[IntPK]
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    company: Mapped[str] = mapped_column(String(120), nullable=False)
    location_name: Mapped[str] = mapped_column(String(500), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    pit_depth_m: Mapped[float] = mapped_column(Float, nullable=False, default=150.0)
    slope_angle_deg: Mapped[float] = mapped_column(Float, nullable=False, default=45.0)
    contact_email: Mapped[Optional[str]] = mapped_column(String(320))
    contact_phone: Mapped[Optional[str]] = mapped_column(String(32))
    alert_threshold_pct: Mapped[float] = mapped_column(
        Float, nullable=False, default=70.0
    )
    created_at: Mapped[TimestampTz] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        # server_default keeps direct INSERTs (Alembic seed, psql, etc.)
        # honest even when SQLAlchemy isn't in the picture.
        server_default=text("CURRENT_TIMESTAMP"),
    )

    alerts: Mapped[list["AlertLog"]] = relationship(
        back_populates="mine",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    users: Mapped[list["User"]] = relationship(back_populates="mine")


class AlertLog(Base):
    __tablename__ = "alert_logs"

    id: Mapped[IntPK]
    mine_id: Mapped[int] = mapped_column(
        # ON DELETE CASCADE matches the ORM cascade above -- clean up
        # historical alerts when a mine record is removed, so we don't
        # accumulate orphaned rows over years of demo tenants coming
        # and going.
        ForeignKey("mines.id", ondelete="CASCADE"),
        nullable=False,
    )
    risk_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    rainfall_mm: Mapped[Optional[float]] = mapped_column(Float)
    pore_pressure_kpa: Mapped[Optional[float]] = mapped_column(Float)
    velocity_mm_h: Mapped[Optional[float]] = mapped_column(Float)
    seismic_rms_g: Mapped[Optional[float]] = mapped_column(Float)
    top_shap_reason: Mapped[Optional[str]] = mapped_column(Text)
    triggered_at: Mapped[TimestampTz] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    mine: Mapped["Mine"] = relationship(back_populates="alerts")

    __table_args__ = (
        Index(
            "ix_alert_logs_mine_triggered",
            "mine_id",
            # DESC because get_recent_alerts always sorts newest-first.
            # Postgres uses the sort direction as-is; SQLite ignores it
            # but Alembic still records the intent.
            text("triggered_at DESC"),
        ),
    )


class User(Base):
    """Application user for auth.

    Right now :mod:`backend.auth` seeds an in-memory admin + operator
    pair for the demo; this table is the migration target. A follow-up
    step swaps ``UserRepository`` in for the seed dict without needing
    another schema change.

    Note on ``password_hash``: this column holds a bcrypt hash produced
    by :func:`app.core.security.hash_password` -- never a plaintext
    password. bcrypt's own salt lives inside the hash string, so no
    separate salt column is needed.
    """

    __tablename__ = "users"

    id: Mapped[IntPK]
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # Store the security.Role enum's string value directly; a String
    # column is easier to migrate than an ENUM type across databases.
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    mine_id: Mapped[Optional[int]] = mapped_column(
        # ON DELETE SET NULL: if the mine is deleted, the user record
        # survives but loses its tenant scope. An admin can then reassign
        # them; without this the user row would either vanish (CASCADE)
        # or the DELETE would fail (RESTRICT). SET NULL is the least-
        # surprising option for a rare, admin-driven event.
        ForeignKey("mines.id", ondelete="SET NULL"),
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # UtcDateTime (not DateTime(timezone=True)) so login-time comparisons
    # stay tz-aware on SQLite dev boxes as well as production Postgres.
    last_login_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())
    created_at: Mapped[TimestampTz] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    mine: Mapped[Optional["Mine"]] = relationship(back_populates="users")

    __table_args__ = (
        # Case-insensitive lookups happen in the repository (LOWER(...))
        # so a UNIQUE on the raw column is enough for the storage layer;
        # the app-level normalization (username.lower()) prevents
        # "Admin" and "admin" from being registered as two accounts.
        UniqueConstraint("username", name="ux_users_username"),
    )


__all__ = ["AlertLog", "Mine", "User"]
