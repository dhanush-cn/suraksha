# Background workers (arq)

Three request-path operations were previously blocking the FastAPI event
loop:

| Endpoint                       | Old shape                                                       | Symptom                                                                 |
| ------------------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `POST /api/upload_csv`         | `async def` that ran `df.iterrows()` -> `predict_rockfall_risk` | 10k-row CSV froze every other request until the loop finished           |
| `POST /api/analyze_drone_image`| `async def` that ran a torch forward pass                        | Similar — a single image locked the event loop for the full inference   |
| Emergency dispatch (SMS/email) | Wasn't wired at all; `notification_service.py` used blocking SMTP + `requests` and caught-and-dropped failures | Alerts were never sent; when they had been, a Twilio outage was invisible |

They now run on an arq worker over the Redis that's already deployed for
the weather cache. The public API is:

```
POST  /api/upload_csv         -> 202 {"status":"queued","job_id":"..."}
POST  /api/analyze_drone_image -> 202 {"status":"queued","job_id":"..."}
GET   /api/telemetry/{mine_id}  # enqueues dispatch_alert when risk >= threshold
GET   /api/jobs/{job_id}        # poll for job status + result
GET   /api/dispatch/dead_letter # admin: inspect failed dispatches
```

## Running the worker

From the `backend/` directory:

```bash
arq app.workers.settings.WorkerSettings
```

The worker reads the same `.env` / environment as the API (`REDIS_URL`,
`SMTP_*`, `TWILIO_*`). Multiple workers can run against the same Redis;
each job is delivered to exactly one.

## Dispatch semantics

`dispatch_alert` implements three properties the old dead code did not:

1. **Idempotency via `SET NX`** — keyed on
   `AlertCreate.idempotency_key` (SHA-256 of `mine_id:risk_level:YYYYMMDDHHMM`).
   Two identical telemetry frames in the same minute enqueue two jobs;
   only the first delivers, the second's task returns `{"status": "skipped"}`.
   Lock TTL is 1 h — long enough to outlive the retry chain, short enough
   to garbage-collect naturally.

2. **Retry with exponential backoff** — on any exception, the task raises
   `arq.Retry(defer=2**attempt)`. Retries fire at 2s, 4s, 8s. Configured
   via `WorkerSettings.max_tries = DISPATCH_MAX_TRIES = 3`. `score_csv`
   and `analyze_image` override this to `max_tries=1` via `arq.func(...)`
   in `WorkerSettings.functions` — deterministic CPU work that gains
   nothing from being retried on failure.

3. **Dead-letter queue** — after the last retry, `_dead_letter` LPUSHes a
   failure payload into `rockfallguard:dispatch:dead_letter` (LTRIMmed to
   1000 entries so memory can't grow unbounded during a sustained outage).
   Ops inspect via `GET /api/dispatch/dead_letter` or
   `redis-cli LRANGE rockfallguard:dispatch:dead_letter 0 -1`.

## Why arq (and not Celery)

**arq wins for this project because:**

- **Native asyncio.** The API is FastAPI + httpx + redis.asyncio. arq
  tasks are `async def(ctx, ...)` and can share the same async libraries
  the API uses — Twilio dispatch uses `httpx.AsyncClient`, the Redis
  operations use the same `redis.asyncio` client. Celery is fundamentally
  sync; its async support is experimental and requires a separate
  event-loop worker, which means task code can't share code with the API.
- **Zero new infrastructure.** arq is Redis-only; Redis is already a
  dependency. Celery+Redis works but Celery's production sweet spot is
  RabbitMQ, which would mean provisioning a second message broker.
- **Small surface.** arq is ~2000 LOC vs Celery's ~40k. For three task
  types with no chords/chains/complex routing, Celery's flexibility is
  overhead. `WorkerSettings` fits in one screen.
- **Built-in cron** (`cron_jobs=[...]` on WorkerSettings) covers periodic
  jobs like DLQ health checks without a separate `celery beat`.

**Where Celery would win, so I know when to reach for it:**

- **Priority queues.** If emergency dispatch had to jump ahead of a
  backlog of CSV scoring jobs, Celery's stable priority queues and
  named-queue routing handle it out of the box. arq has one queue.
- **Canvas primitives.** For a fan-out/gather workflow (shard a 500k-row
  CSV across N workers, collect, summarize), Celery's `chord` and `group`
  are first-class. arq requires `asyncio.gather` inside a task.
- **Multi-broker.** Celery abstracts RabbitMQ / Redis / SQS. arq only
  speaks Redis. A move to SQS-backed workers would mean a rewrite.
- **Ecosystem tooling.** Flower, django-celery-beat, and long-tail bug
  fixes from a much larger user base. arq is well-maintained but a
  single-maintainer project — production readiness cuts differently.

**Known trade-offs of the current implementation:**

- No jitter on retry. If 100 alerts fail simultaneously (Twilio incident),
  all 100 retry at exactly 2s. Real production would add
  `random.uniform(0, 1)` to `defer=` to spread the retry burst.
- Redis is both broker and result backend. A Redis outage takes down the
  queue and the FastAPI enqueue path (though the DB alert log survives,
  so the alert can be re-driven from `get_recent_alerts`). Celery+RabbitMQ
  separates these; if that separation matters for a given deployment,
  that's the exit ramp.
- DLQ is an application-managed list, not a broker-level queue. arq
  (and Celery-with-Redis) have no built-in DLQ; only Celery-with-
  RabbitMQ gets DLX for free. The upside is that a JSON payload in a
  Redis list is trivial to inspect and reprocess without extra tooling.
