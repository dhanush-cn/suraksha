# RockfallGuard — Code Audit & Refactoring Plan

**Scope:** `dhanush-cn/suraksha` @ `d224135` — cloned and read in full (1,091 LOC backend, 583 LOC frontend, CI, Dockerfile).
**Method:** static read-through + targeted greps. Findings marked **[V]** were verified by execution; the rest are read-confirmed against specific line numbers.

Severity: **S1** = broken or exploitable in production · **S2** = will break under load/failure · **S3** = maintainability and correctness debt.

---

## 1. Self-Audit & Error Log

### 1.1 Critical Bugs / Runtime Errors

| # | Sev | Location | Finding |
|---|-----|----------|---------|
| C1 | S1 | `main.py:133` | **`/api/telemetry/{mine_id}` raises `TypeError` on every call.** Passes `mine_info=target_mine`; `simulator.generate_telemetry_frame(self, scenario, weather_data)` accepts no such parameter and has no `**kwargs`. This is the product's core endpoint. |
| C2 | S1 | `main.py:203-204` | **`NameError: np is not defined`** in `/api/upload_csv`. `np.mean`/`np.max` are called but `numpy` is never imported in the module. Fires on the first non-empty CSV. |
| C3 | S1 | `.github/workflows/ci-cd.yml:41,50` | **CI cannot pass.** Runs `pip install -r requirements.txt` and `python ml/train_model.py`; neither `requirements.txt` nor `ml/` exists in the repo. The Dockerfile also `COPY`s `ml/` and `models/`, so the image build fails identically. |
| C4 | S1 | `ml_engine.py:26-36` | **Deferred `AttributeError` on missing model artifacts.** `load_ml_artifacts` catches the load failure, prints a warning, and returns. `_SCALER` stays `None`, so line 79 fails with `NoneType has no attribute 'transform'` — surfacing as an opaque 500 far from the real cause. |
| C5 | S2 | `app.js:389` vs `cv_engine.py:104` | **Frontend/backend contract mismatch.** UI reads `data.visual_status`; the API returns `anomaly_status`. The drone panel renders `undefined`. |
| C6 | S2 | `cv_engine.py:111-117` | **Failures return HTTP 200.** The handler catches everything and returns `{"status": "error"}` with a 200 status and `visual_risk_percentage: 0.0` — a failed analysis is indistinguishable from a genuinely safe wall. |
| C7 | S2 | `main.py:234-241` | **Exception double-wrapping.** The bare `except Exception` catches the handler's own `HTTPException(400, "Uploaded file is empty.")` and re-wraps it, producing `"Image processing error: 400: Uploaded file is empty."` |
| C8 | S2 | `main.py:125-127` | **Silent wrong-mine fallback.** An unknown `mine_id` falls back to `mines[0]`, then a hardcoded dummy. A request for mine 99 returns mine 1's telemetry, labelled as a success. Should be `404`. |
| C9 | S3 | `notification_service.py:84` | `os.popen("date /t")` is a Windows command. On Linux/Docker the ternary takes the other branch, so every dispatch log entry's timestamp is the literal string `"Now"`. |
| C10 | S3 | `database.py:69-83` | Connection leak: `register_mine` closes the connection on the success and `IntegrityError` paths only. Any other exception leaks it. No `try/finally`. |
| C11 | S3 | `main.py:100` | `req.dict()` is the Pydantic v1 API; deprecated in v2 (`.model_dump()`). |
| C12 | S3 | `main.py:32` | `@app.on_event("startup")` is deprecated in favour of the `lifespan` context manager. |

### 1.2 Architectural & Structural Flaws

| # | Sev | Finding |
|---|-----|---------|
| A1 | S1 | **`auth.py` is dead code.** `main.py` never imports it. `get_current_user`, `enforce_tenant_access` and `enforce_admin_only` are fully written and never called — every endpoint is unauthenticated. `backend/tests/test_app.py` asserts `403` on cross-tenant access, so the tests encode the intended behaviour the app does not implement. |
| A2 | S1 | **The alerting pipeline is disconnected.** `notification_service` (SMTP/Twilio) and `redis_service.publish_emergency_alert` are implemented but never imported by `main.py`, and nothing subscribes to the `rockfall_emergency_alerts` channel. On a critical reading, `log_alert()` writes a row and **nobody is notified**. For an emergency-alerting product this is the defining defect. |
| A3 | S1 | **No layer separation.** `main.py` is HTTP routing + business rules + persistence + ML orchestration + file I/O. Route handlers call `database.py` functions directly; there is no service or repository layer, so no business rule is unit-testable without an HTTP client. |
| A4 | S2 | **Duplicated business logic.** The alert-threshold rule exists twice with *different* values: `/api/predict_risk` hardcodes `risk_pct >= 60.0` (line 105) while `/api/telemetry` uses the mine's configured `alert_threshold_pct` (line 140). The same reading alerts or doesn't depending on which endpoint saw it. The `shap_explanations[0]` fallback string is likewise duplicated with two different defaults (lines 106, 145). |
| A5 | S2 | **Module-level singletons with hidden state.** `simulator_instance` (`simulator.py:73`) accumulates `cumulative_disp` in process memory, and `redis_service` (`redis_service.py:74`) binds a connection at import time. Both make the app non-reentrant and prevent horizontal scaling: each replica has a different displacement history. |
| A6 | S2 | **`DISPATCH_LOGS` is a module-level list** (`notification_service.py:19`). Per-process memory, unbounded growth, and with >1 worker `get_dispatch_logs()` returns a different answer per replica. |
| A7 | S2 | **No migration path.** Schema lives in `CREATE TABLE IF NOT EXISTS` inside `init_db()`. Any column change to a deployed database is a manual operation. No Alembic. |
| A8 | S3 | **Sibling imports** (`from database import ...`) require the CWD to be `backend/`; `tests/test_app.py` compensates with `sys.path.append`. Not an installable package. |
| A9 | S3 | **`SELECT *` to the client.** `get_all_mines()` returns raw rows, so every new column is automatically exposed. |
| A10 | S3 | **Static files mounted at `/`** (`main.py:246`) after the API routes. Works, but couples frontend delivery to the API process and blocks independent CDN deployment. |

### 1.3 Performance & Async Bottlenecks

| # | Sev | Finding |
|---|-----|---------|
| P1 | S1 | **Blocking I/O in `async def`.** `/api/upload_csv` is `async` but runs `pd.read_csv` plus a synchronous `df.iterrows()` loop calling `predict_rockfall_risk` per row. This blocks the **entire event loop** — not just this request — for the duration. A 10k-row CSV freezes the whole server. |
| P2 | S1 | **Synchronous ML inference in the request path.** `/api/analyze_drone_image` runs a PyTorch forward pass inline; `/api/telemetry` runs sklearn + SHAP inline. SHAP is the expensive part and is recomputed on every poll. |
| P3 | S1 | **Blocking network calls.** `weather_service` uses `requests` (sync) inside async handlers; `notification_service` performs a blocking SMTP handshake and a sync `requests.post` to Twilio. A slow SMTP server stalls the event loop. |
| P4 | S2 | **N+1 and no connection pooling.** `get_db_connection()` opens a fresh `sqlite3.connect()` per call. `/api/telemetry/{mine_id}` calls `get_all_mines()` (full table scan) and filters in Python to find one row. |
| P5 | S2 | **SQLite under concurrent writers.** Writer-level file locking; a sensor-ingestion workload is concurrent by definition → `database is locked`. |
| P6 | S2 | **Unbounded work from client input.** No row cap on CSV upload, no file-size cap, no image-dimension cap. A 500 MB CSV is a trivial DoS. |
| P7 | S2 | **No retry, no idempotency, no dead-letter.** A failed Twilio call is printed and dropped. Conversely, because `/api/telemetry` is polled on a timer, a slope above threshold writes a row **and** dispatches an alert on *every poll*. |
| P8 | S3 | **No caching of hot reads.** Redis caches weather only; `get_all_mines()` hits the DB on every telemetry request. |
| P9 | S3 | Missing index on `alert_logs(mine_id, triggered_at)` — the exact access pattern of `get_recent_alerts`. |

### 1.4 Security & Validation Issues

Mapped to **OWASP Top 10:2025** (current edition; SSRF is now folded into A01).

| # | Sev | OWASP | Finding |
|---|-----|-------|---------|
| S1 | **S1** | A01/A07 | **No token = admin.** `auth.py:70-79` returns `{"role": "admin"}` when the `Authorization` header is absent. Wiring auth in without removing this ships a bypass, not a defence. |
| S2 | **S1** | A02/A04 | **Hardcoded fallback JWT secret in a public repo** (`auth.py:11`). Any environment missing `JWT_SECRET` signs tokens with a value anyone can read on GitHub — trivial admin-token forgery. |
| S3 | **S1** | A01 | **Every endpoint is unauthenticated** (consequence of A1), including destructive routes. |
| S4 | S2 | A02 | **`allow_origins=["*"]` with `allow_credentials=True`** (`main.py:23-29`). Browsers reject this combination, so the config is both insecure in intent and non-functional in practice. |
| S5 | S2 | A05 | **DOM XSS in the frontend.** `app.js:302,312` interpolate `file.name` into `innerHTML` unescaped; lines 343-353 and 387-396 do the same with API response fields. |
| S6 | S2 | A05 | **HTML injection into emails.** `notification_service.py:30-58` builds the body by f-string interpolation of `mine_name`, `risk_level`, `top_reason` with no escaping. `mine_name` is attacker-controlled via mine registration. |
| S7 | S2 | A06 | **Unrestricted file upload.** `/api/upload_csv` validates by filename suffix only (`.endswith(".csv")`) — no content-type or content check — and writes every upload to the same fixed path `../data/custom_mine_upload.csv`, so concurrent uploads race and overwrite. `/api/analyze_drone_image` validates only `len(contents) < 10`. |
| S8 | S2 | A06 | **No rate limiting anywhere.** ML inference endpoints are free to call in a loop. |
| S9 | S2 | A10 | **Swallowed dispatch failures.** SMTP/Twilio exceptions are caught, printed, and the entry is logged as `"DISPATCHED (Simulated Server Log)"` — a *successful-looking* record for an alert that was never sent. |
| S10 | S2 | A09 | **No structured logging, no auth event logging.** `print()` throughout; no correlation IDs; nothing would reveal credential stuffing. |
| S11 | S3 | A02 | **No security headers, no TLS story.** No HSTS/CSP/`X-Frame-Options`/`nosniff`; Dockerfile exposes plain HTTP on `:8005` with no termination layer documented. |
| S12 | S3 | A03 | **Zero dependency pinning** — no `requirements.txt`, no lockfile, no `pip-audit`/`bandit` in CI. |
| S13 | S3 | A08 | **Deserialization surface.** `joblib.load` (×5) and `torch.load` execute arbitrary code on untrusted input. Safe today (fixed local paths), but must never be pointed at user-supplied artifacts. Confirm `torch.load(weights_only=True)`. |
| S14 | S3 | A01 | **Over-exposure of PII.** `GET /api/mines` returns `contact_email` and `contact_phone` for all mines, unauthenticated. |
| S15 | S3 | A07 | 7-day token TTL, no refresh flow, no `jti`, no revocation path. |
| S16 | S3 | A05 | Error responses return `str(e)` to the client (`main.py:86,226,241`), leaking paths, driver internals and query fragments. |

**Validation gaps (Pydantic):** no range constraints anywhere — `latitude=9999`, `humidity_pct=-40`, `slope_angle_deg=200` all validate today. `scenario` is an unvalidated `str` that silently falls through to a default branch on a typo. Models permit extra fields (mass-assignment risk). `float("nan")` passes every check and yields `risk_percentage: nan` — a critical reading that never triggers an alert. No response models, so the API contract is undocumented and unenforced.

**Credit where due — these are correct and worth keeping:**
- `database.py` parameterises **every** query with `?` placeholders. No SQL injection.
- `auth.py` uses `hmac.compare_digest` (timing-safe) and pins HMAC-SHA256 on the verify side, so it resists `alg:none`/algorithm-confusion **by construction**.
- `weather_service.py` degrades gracefully to sane defaults when Open-Meteo is unreachable.
- `redis_service.py` implements clean cache-aside with a null-client fallback.
- `tests/test_app.py` already encodes the correct tenant-isolation contract.

---

## 2. Refactoring Plan

**Sequencing principle:** stop the bleeding → build the foundation → migrate module by module behind it. No step leaves `main` unrunnable.

**Step 0 — Triage (½ day).** Fix C1, C2, C3, C9, C10. Add pinned `requirements.txt`. Get CI green. *Nothing else can be validated until the app boots and the pipeline runs.*

**Step 1 — Foundation (`core/` + `schemas/`) — ✅ implemented below.** Config, exceptions, logging, security, middleware, error handlers, and the full Pydantic v2 contract set. Additive: nothing imports it yet, so it cannot break the running app.

**Step 2 — Persistence.** SQLAlchemy 2.0 async models + `asyncpg`; Alembic baseline migration; add the `users` table auth needs and the `(mine_id, triggered_at)` index. Repository classes replace direct `database.py` calls. *Fixes A7, A9, P4, P5, P9.*

**Step 3 — Service layer.** `MineService`, `RiskService`, `AlertService`, `AuthService`. Business rules move out of route handlers; the duplicated threshold logic (A4) collapses into one `AlertService.should_trigger()`. Services raise `AppError`, never `HTTPException`. *Fixes A3, A4.*

**Step 4 — Auth wiring.** `get_current_principal` dependency using `core/security` + `schemas/auth.Principal`; **delete the guest fallback**. Apply `Depends()` per route; `authorize_mine()` on every mine-scoped path. Target: `tests/test_app.py` passes unmodified. *Fixes S1, S2, S3, A1.*

**Step 5 — Async & background workers.** Convert I/O to `httpx.AsyncClient`; move ML inference to `run_in_threadpool`; move notification dispatch and CSV batch scoring to arq (Redis-backed, matches the existing asyncio+Redis stack). Add exponential-backoff retry, the Redis `SET NX` idempotency lock keyed on `AlertCreate.idempotency_key`, and a dead-letter list. *Fixes P1, P2, P3, P7, A2, A6, S9.*

**Step 6 — Redis expansion.** Rate limiter, mine-metadata cache with explicit invalidation, JWT revocation blocklist, distributed alert lock. Replace fire-and-forget Pub/Sub with Streams + consumer groups (Pub/Sub drops messages with no subscriber — unacceptable for emergency alerts). *Fixes S8, P8, S15, A2.*

**Step 7 — Hardening & observability.** Prometheus metrics; real readiness probe; escape all frontend `innerHTML` (or use `textContent`); Jinja2 autoescape for emails; upload content-type + size + row caps; `pip-audit` + `bandit` in CI; multi-stage Dockerfile; TLS termination. *Fixes S5, S6, S7, S11, S12, S10.*

---

## 3. Code Implementation — Foundational Layer

Delivered under `backend/app/`. **22/22 verification checks pass** (`python verify_core.py`).

```
backend/
├── .env.example
├── verify_core.py              # executable proof of the properties below
└── app/
    ├── core/
    │   ├── config.py           # pydantic-settings; no secret defaults; prod invariants
    │   ├── exceptions.py       # AppError hierarchy + stable ErrorCode enum
    │   ├── error_handlers.py   # one error envelope; no stack-trace leakage
    │   ├── logging.py          # JSON logs, correlation IDs, secret redaction
    │   ├── middleware.py       # TrustedHost→CORS→RequestContext→GZip→Headers→BodyLimit
    │   └── security.py         # PyJWT (pinned alg) + bcrypt; token types; jti
    └── schemas/
        ├── base.py             # RequestModel(extra=forbid) / ResponseModel / StrictFloatModel
        ├── common.py           # ErrorResponse, HealthResponse, Page
        ├── auth.py             # Principal — no anonymous-admin construction path
        ├── mine.py             # Create/Update/Summary/Detail split
        ├── telemetry.py        # physical sensor bounds; Scenario enum
        └── alert.py            # idempotency key; dispatch state machine
```

### Verified security properties

| Property | Check |
|---|---|
| `alg=none` forgery rejected | `security: alg=none forgery is rejected` |
| Refresh token can't be replayed as access token | `security: refresh token rejected where access expected` |
| Short/missing `JWT_SECRET` fails at boot | `config: rejects short jwt_secret` |
| Production rejects `debug=True` / empty CORS | 2 checks |
| Cross-tenant access raises | `schemas: Principal blocks cross-tenant access` |
| Unscoped non-admin principal impossible | `schemas: non-admin without mine_id scope is rejected` |
| Mass assignment blocked | `schemas: extra fields forbidden on requests` |
| NaN sensor readings rejected | `schemas: NaN rejected in sensor reading` |
| Passwords redacted from logs | `logging: JSON output redacts secrets` |
| No stack traces in 500 responses | `app: middleware + error handlers produce uniform envelope` |
| Oversized bodies → 413 | `app: body size limit returns 413` |

### One defect found *in the new code* by the harness

`pydantic-settings` JSON-decodes complex-typed fields **before** `mode="before"` validators run, so `CORS_ORIGINS=a,b` raised `SettingsError` instead of reaching `_split_csv`. Fixed with the `NoDecode` annotation. Worth noting as an interview anecdote: the bug was invisible on read-through and only appeared on execution — which is the argument for the harness existing at all.

### Interview-facing design decisions

1. **Two model bases, opposite defaults.** Requests `extra="forbid"` (a typo'd field is a 422, not a silently ignored parameter); responses `extra="ignore"` + `from_attributes=True`.
2. **Auth is a dependency, not middleware.** Middleware would run on `/health` and `/docs`, and wouldn't appear in the OpenAPI schema. `Depends()` is per-route and self-documenting.
3. **`authorize_mine()` raises; `can_access_mine()` returns bool.** Prefer the raising form at call sites — a bare boolean is easy to call and forget to branch on. Fails closed.
4. **Idempotency key buckets to the minute.** Polling every 5s on a critical slope currently means an SMS every 5s. Bucketing collapses the burst; the Redis lock makes dispatch exactly-once across retries and replicas.
5. **Out-of-distribution readings are flagged, not rejected.** A genuine slope failure produces extreme numbers — refusing them would silence the one alert that matters. Accept, score, mark `low_confidence`.
6. **Liveness ≠ readiness.** A liveness probe that fails on a Redis outage triggers a restart that cannot fix Redis, converting a degraded dependency into a restart loop.

---

## Next step

Say the word and I'll take **Step 2 + Step 4**: SQLAlchemy async models, Alembic baseline, the repository layer, and the `get_current_principal` dependency wired into `main.py` — with `backend/tests/test_app.py` passing unmodified as the acceptance criterion.