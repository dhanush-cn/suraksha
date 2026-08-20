"""arq ``WorkerSettings`` -- the process launched by the ``arq`` CLI.

Run the worker from the ``backend/`` directory with::

    arq app.workers.settings.WorkerSettings

Or, with the venv Python already on PATH::

    python -m arq app.workers.settings.WorkerSettings

The worker process is separate from the FastAPI process on purpose:

* API pods are horizontally-scaled by inbound HTTP load; worker pods are
  scaled by queue depth. Coupling them wastes one axis of scaling.
* A CPU spike from a 50k-row CSV or a torch inference cannot page out
  incoming health-check requests.
* Rolling out an ML-engine change becomes a worker-only deploy, so the
  API stays up while workers restart.
"""

from __future__ import annotations

import os

# Match the dev defaults in main.py so the worker boots without a full
# environment set (e.g. under a local `arq` CLI invocation). Real
# deployments MUST set these -- os.environ.setdefault leaves anything
# already exported untouched. Must run BEFORE importing anything that
# reaches Settings() via app.core.config's lru_cache.
os.environ.setdefault(
    "JWT_SECRET",
    "rockfallguard-dev-only-insecure-jwt-secret-do-not-use-in-prod",
)
os.environ.setdefault("DATABASE_URL", "sqlite:///./mines.db")

from arq import func  # noqa: E402

from app.workers.queue import build_redis_settings  # noqa: E402
from app.workers.tasks import (  # noqa: E402
    DISPATCH_MAX_TRIES,
    analyze_image,
    dispatch_alert,
    score_csv,
)


class WorkerSettings:
    """arq reads this class attribute-by-attribute. Do not add ``__init__``."""

    # Per-function overrides:
    #   score_csv / analyze_image use max_tries=1 -- deterministic CPU
    #     work; retrying a corrupt input just wastes worker capacity.
    #   dispatch_alert uses the worker-wide default (DISPATCH_MAX_TRIES),
    #     which its retry-with-Retry(defer=...) logic depends on to
    #     trigger the dead-letter branch on the final attempt.
    # Names must match the .__name__ the enqueue side calls with.
    functions = [
        func(score_csv, name="score_csv", max_tries=1),
        func(analyze_image, name="analyze_image", max_tries=1),
        dispatch_alert,
    ]

    max_tries = DISPATCH_MAX_TRIES

    # 5 minutes: the largest supported CSV upload (50k rows per the
    # ``max_upload_rows`` setting) scores well inside this budget on the
    # sklearn model, plus headroom for image analysis on CPU.
    job_timeout = 300

    # Shared Redis with the enqueue path -- one source of truth for the
    # connection settings.
    redis_settings = build_redis_settings()


__all__ = ["WorkerSettings"]
