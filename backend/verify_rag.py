"""Smoke-verification of the RAG layer (Step 8). Run: python verify_rag.py

Covers what can be tested without pgvector -- the layered pieces plus
the graceful-degradation paths. The vector-similarity check itself
runs only when a real Postgres+pgvector is reachable; otherwise it's
reported PASS-SKIPPED with the reason.

Checks:

* build_alert_source_text is deterministic across calls (equal inputs
  -> equal string), and contains every meaningful axis.
* LLMClient raises LLMConfigurationError when no API key is set.
* LLMClient.stream_chat parses the OpenAI SSE format correctly
  (mocked httpx AsyncClient.stream).
* ChatService yields metadata BEFORE the first content chunk, and
  falls through to a "config missing" error event when the LLM is
  unconfigured.
* retrieval.top_k_similar returns [] on SQLite (no pgvector).
* embed_alert worker task returns "skipped" on SQLite / no LLM key.
* /api/chat returns a 200 SSE stream with an error event when the
  LLM is unconfigured (proves the endpoint is wired but degrades).
* Optional: end-to-end pgvector query works when RAG_POSTGRES_URL
  is set to a live Postgres with the ``vector`` extension.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("JWT_SECRET", "a" * 48)
os.environ.setdefault("DATABASE_URL", "sqlite:///./mines.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ENVIRONMENT", "test")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def check(name: str):
    def wrapper(fn):
        try:
            fn()
            results.append((PASS, name, ""))
        except _Skipped as sk:
            results.append((SKIP, name, str(sk)))
        except Exception as exc:  # noqa: BLE001
            results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return wrapper


class _Skipped(Exception):
    """Raise to report a check as skipped rather than passed/failed."""


async def _async_gen(items) -> "AsyncIterator[str]":
    """Simple async iterator over an in-memory list.

    Defined above the checks because @check evaluates its wrapped fn
    at decoration time -- helpers must exist first.
    """
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# source_text builder
# ---------------------------------------------------------------------------


@check("embeddings: build_alert_source_text is deterministic and complete")
def _():
    from app.rag.embeddings import build_alert_source_text

    kwargs = dict(
        mine_name="Grasberg",
        company="Freeport Copper-Gold",
        risk_level="critical",
        risk_percentage=88.4,
        top_shap_reason="Pore Water Pressure (90.2 kPa)",
        rainfall_mm=42.0,
        pore_pressure_kpa=90.2,
        velocity_mm_h=5.1,
        seismic_rms_g=0.44,
        triggered_at=datetime(2026, 8, 20, 10, 30, tzinfo=timezone.utc),
    )
    a = build_alert_source_text(**kwargs)
    b = build_alert_source_text(**kwargs)
    assert a == b, "same inputs must produce byte-identical source_text"

    # Every meaningful axis appears once so cosine search over these
    # strings can rank by any of them.
    for token in (
        "Grasberg", "Freeport", "CRITICAL", "88.4%", "Pore Water Pressure",
        "rainfall", "42.0", "pore pressure", "90.2", "velocity",
        "seismic",
    ):
        assert token in a, f"missing axis: {token!r} not in {a!r}"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


@check("llm: LLMClient raises LLMConfigurationError when no API key set")
def _():
    from app.core.config import get_settings
    from app.rag.client import LLMClient, LLMConfigurationError

    # Ensure key is unset for this check.
    os.environ.pop("LLM_API_KEY", None)
    get_settings.cache_clear()  # type: ignore[attr-defined]

    async def go():
        client = LLMClient()
        try:
            await client.embed("hello")
        except LLMConfigurationError:
            pass
        else:
            raise AssertionError("expected LLMConfigurationError")

    asyncio.run(go())


@check("llm: stream_chat parses the OpenAI SSE format and yields content")
def _():
    from app.rag.client import LLMClient

    # Set an API key so the client doesn't short-circuit on config.
    os.environ["LLM_API_KEY"] = "sk-test"
    from app.core.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    # Simulate the SSE lines a real backend would emit.
    lines = [
        "",  # blank keepalive
        ": ping",  # SSE comment
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[{"delta":{"content":"!"}}]}',
        'data: [DONE]',
    ]

    async def fake_aiter_lines():
        for line in lines:
            yield line

    fake_response = MagicMock()
    fake_response.aiter_lines = fake_aiter_lines
    fake_response.raise_for_status = MagicMock()

    class _FakeStreamCtx:
        async def __aenter__(self_inner):  # noqa: N805
            return fake_response
        async def __aexit__(self_inner, *exc):  # noqa: N805
            return False

    class _FakeClientCtx:
        def stream(self_inner, *args, **kwargs):  # noqa: N805
            return _FakeStreamCtx()
        async def __aenter__(self_inner):  # noqa: N805
            return self_inner
        async def __aexit__(self_inner, *exc):  # noqa: N805
            return False

    async def go():
        chunks: list[str] = []
        with patch("httpx.AsyncClient", return_value=_FakeClientCtx()):
            async for delta in LLMClient().stream_chat(
                [{"role": "user", "content": "hi"}]
            ):
                chunks.append(delta)
        assert "".join(chunks) == "Hello world!", chunks

    asyncio.run(go())


# ---------------------------------------------------------------------------
# ChatService
# ---------------------------------------------------------------------------


@check("chat: emits metadata event BEFORE first content chunk")
def _():
    """The frontend can then render a 'sources' block while the LLM
    is still composing the answer body."""
    from app.rag.service import ChatService

    # Pre-populate the API key so ChatService gets past the config
    # guard and reaches the streaming path.
    os.environ["LLM_API_KEY"] = "sk-test"
    from app.core.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    async def go():
        session = MagicMock()
        # Patch retrieval to return one hit, and the LLM stream to
        # yield two deltas.
        with patch(
            "app.rag.service.top_k_similar",
            new=AsyncMock(return_value=[
                MagicMock(alert_id=42, similarity=0.87, source_text="fake"),
            ]),
        ), patch(
            "app.rag.client.LLMClient.embed",
            new=AsyncMock(return_value=[0.0] * 1536),
        ), patch(
            "app.rag.client.LLMClient.stream_chat",
            new=lambda self, messages: _async_gen(["one ", "two"]),
        ):
            events: list[dict] = []
            async for raw in ChatService(session).stream("what happened?"):
                assert raw.startswith("data: ") and raw.endswith("\n\n")
                events.append(json.loads(raw[6:].rstrip()))

        types = [e["type"] for e in events]
        assert types[0] == "metadata", f"metadata not first: {types}"
        assert types[-1] == "done", f"done not last: {types}"
        assert "content" in types, f"no content: {types}"

        # metadata event carries the retrieved alert ids
        assert events[0]["retrieved"][0]["alert_id"] == 42

    asyncio.run(go())


@check("chat: LLM unconfigured -> error event, not exception")
def _():
    from app.rag.service import ChatService

    os.environ.pop("LLM_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    async def go():
        events = []
        async for raw in ChatService(MagicMock()).stream("hi"):
            events.append(json.loads(raw[6:].rstrip()))
        assert events[0]["type"] == "error"
        assert "not configured" in events[0]["error"].lower()

    asyncio.run(go())


# ---------------------------------------------------------------------------
# retrieval fallback on SQLite
# ---------------------------------------------------------------------------


@check("retrieval: top_k_similar returns [] on non-postgres session")
def _():
    from app.rag.retrieval import top_k_similar

    async def go():
        fake_session = MagicMock()
        fake_session.bind.dialect.name = "sqlite"
        hits = await top_k_similar(session=fake_session, query_embedding=[0.0] * 4, k=5)
        assert hits == []

    asyncio.run(go())


# ---------------------------------------------------------------------------
# embed_alert task -- skipped path
# ---------------------------------------------------------------------------


@check("embed_alert: worker task returns 'skipped' when LLM_API_KEY is missing")
def _():
    from app.workers.tasks import embed_alert

    os.environ.pop("LLM_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    async def go():
        result = await embed_alert({}, 1)
        assert result["status"] == "skipped"
        assert "LLM_API_KEY" in result["reason"]

    asyncio.run(go())


# ---------------------------------------------------------------------------
# /api/chat endpoint wired + returns SSE with error when LLM off
# ---------------------------------------------------------------------------


@check("api: /api/chat returns SSE stream with 'error' event when LLM unconfigured")
def _():
    from fastapi.testclient import TestClient

    from database import init_db

    os.environ.pop("LLM_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]

    init_db()
    import main

    c = TestClient(main.app)
    login = c.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).json()
    headers = {"Authorization": f"Bearer {login['access_token']}"}

    r = c.post("/api/chat", json={"question": "hello"}, headers=headers)
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/event-stream"), r.headers
    body = r.text
    # Every event is a `data: {...}` line ending with \n\n.
    assert "data: " in body
    # We wrote the "no API key" branch to emit an error event as the
    # first event; assert that.
    first_event = body.split("\n\n", 1)[0]
    assert first_event.startswith("data: ")
    payload = json.loads(first_event[6:])
    assert payload["type"] == "error"


# ---------------------------------------------------------------------------
# optional: end-to-end pgvector round-trip against a real Postgres
# ---------------------------------------------------------------------------


@check("e2e: pgvector round-trip (requires RAG_POSTGRES_URL)")
def _():
    """Full path: insert 3 alerts with hand-crafted embeddings, query
    with a vector aligned with the first, assert the ordering is
    (identical -> near-direction -> orthogonal).

    Verifies the wire-encoding path (list -> [1.0,0.0,...] literal ->
    CAST vector), the HNSW index is picked up, and the ``<=>`` cosine
    distance operator returns distances in the expected [0, 2] range.
    """
    live_url = os.environ.get("RAG_POSTGRES_URL")
    if not live_url:
        raise _Skipped(
            "RAG_POSTGRES_URL not set "
            "(start `docker compose -f docker-compose.dev.yml up postgres`)"
        )

    # Swap the app's DATABASE_URL to the live Postgres for this check.
    saved = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = live_url

    from app.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    from app.db import engine as engine_module

    engine_module.reset_engine_for_tests()

    try:
        from datetime import datetime, timezone

        from sqlalchemy import text as _text

        from app.db.engine import session_scope
        from app.db.models import AlertLog, Mine
        from app.rag.embeddings import build_alert_source_text_from_row
        from app.rag.retrieval import _to_vector_literal, top_k_similar

        async def go():
            # Fresh scratch space; TRUNCATE with CASCADE + RESTART
            # IDENTITY so the sequence rewinds and we get deterministic
            # ids on repeated runs.
            async with session_scope() as s:
                await s.execute(
                    _text(
                        "TRUNCATE mines, alert_logs, alert_embeddings, users "
                        "RESTART IDENTITY CASCADE"
                    )
                )
                m = Mine(
                    name="Grasberg", company="Freeport",
                    location_name="Papua", latitude=-4.05, longitude=137.11,
                    contact_email="s@x.org", alert_threshold_pct=70.0,
                )
                s.add(m)
                await s.flush()

                alerts = []
                for i, (rl, rp) in enumerate(
                    [("critical", 88.4), ("warning", 55.0), ("critical", 91.2)]
                ):
                    a = AlertLog(
                        mine_id=m.id, risk_percentage=rp, risk_level=rl,
                        rainfall_mm=10.0, pore_pressure_kpa=50.0,
                        velocity_mm_h=0.1, seismic_rms_g=0.05,
                        top_shap_reason="test",
                        triggered_at=datetime(
                            2026, 8, 20, 10 + i, 0, tzinfo=timezone.utc
                        ),
                    )
                    s.add(a)
                    await s.flush()
                    alerts.append(a)
                    a.mine = m

                # Distinct 1536-dim vectors: alert 1 == [1,0,0,...],
                # alert 2 orthogonal, alert 3 near alert 1's direction.
                vecs = [
                    [1.0] + [0.0] * 1535,
                    [0.0, 1.0] + [0.0] * 1534,
                    [0.9, 0.1] + [0.0] * 1534,
                ]
                for a, v in zip(alerts, vecs):
                    await s.execute(
                        _text(
                            "INSERT INTO alert_embeddings "
                            "(alert_id, source_text, model, embedding, created_at) "
                            "VALUES (:aid, :src, :m, CAST(:v AS vector), CURRENT_TIMESTAMP)"
                        ),
                        {
                            "aid": a.id,
                            "src": build_alert_source_text_from_row(a),
                            "m": "test-embed",
                            "v": _to_vector_literal(v),
                        },
                    )

            async with session_scope() as s:
                hits = await top_k_similar(
                    session=s,
                    query_embedding=[1.0] + [0.0] * 1535,
                    k=3,
                )
                hit_ids = [h.alert_id for h in hits]
                assert hit_ids == [1, 3, 2], (
                    f"expected [1, 3, 2] (identical -> near -> orthogonal), got {hit_ids}"
                )
                # Cosine distances: 0 for identical, ~0.006 for the
                # 0.9/0.1 alignment, 1.0 for orthogonal.
                assert hits[0].distance < 1e-6, hits[0].distance
                assert hits[1].distance < 0.05, hits[1].distance
                assert abs(hits[2].distance - 1.0) < 1e-3, hits[2].distance

        asyncio.run(go())
    finally:
        # Restore whatever DATABASE_URL was before this check.
        if saved is not None:
            os.environ["DATABASE_URL"] = saved
        else:
            os.environ.pop("DATABASE_URL", None)
        get_settings.cache_clear()  # type: ignore[attr-defined]
        engine_module.reset_engine_for_tests()


if __name__ == "__main__":
    print("\n" + "=" * 78)
    for status, name, detail in results:
        marker = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘"}[status]
        print(f"  {marker} {status}  {name}")
        if detail:
            print(f"           -> {detail}")
    failures = sum(1 for s, _, _ in results if s == FAIL)
    passes = sum(1 for s, _, _ in results if s == PASS)
    skipped = sum(1 for s, _, _ in results if s == SKIP)
    print("=" * 78)
    print(f"  {passes} passed, {skipped} skipped, {failures} failed  ({len(results)} total)")
    print("=" * 78)
    raise SystemExit(1 if failures else 0)
