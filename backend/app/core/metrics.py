"""Prometheus metric definitions.

One module owns every metric, so the label cardinality budget is
reviewable in one place. Metrics are recorded via helper methods
(``record_ml_inference``, ``record_dispatch_outcome``, etc.) rather
than callers touching the Counter/Histogram objects directly -- this
keeps label naming consistent and makes the "what fields does this
metric have" question one grep away.

Cardinality discipline:

* Every label domain is bounded. ``channel`` is (email, sms, webhook);
  ``outcome`` is (sent, simulated, failed, dead_lettered, skipped);
  ``scope`` (rate-limit / cache) matches the fixed set in
  :mod:`app.core.rate_limit` and :mod:`app.core.cache`. No free-form
  strings (mine name, user id) go into labels -- those would blow up
  the time-series count.
* Latency histograms use Prometheus's default bucket layout; if we
  later find they don't match the observed distribution we tune per
  metric here, not per call site.

Exposition: :mod:`main` mounts ``/metrics`` on the app; Prometheus
scrapes it. In multi-worker deployments the standard
``prometheus_client`` in-process registry is per-worker, which means
scraping a single URL returns whichever worker answered. Real
production would use the multiprocess mode (``PROMETHEUS_MULTIPROC_DIR``);
documented as a follow-up rather than done here because it needs an
init hook in whatever process manager we adopt.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# ML / CV inference latency
# ---------------------------------------------------------------------------

ml_inference_seconds = Histogram(
    "rockfallguard_ml_inference_seconds",
    "Time spent running a risk-scoring inference call, end to end.",
    labelnames=("model",),
)

cv_inference_seconds = Histogram(
    "rockfallguard_cv_inference_seconds",
    "Time spent running a drone-image CV inference call.",
    labelnames=("model",),
)


@contextmanager
def observe_ml_inference(model: str = "rockfall_ensemble"):
    """Time-and-record wrapper. Usage: ``with observe_ml_inference(): ...``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ml_inference_seconds.labels(model=model).observe(time.perf_counter() - start)


@contextmanager
def observe_cv_inference(model: str = "pit_wall_cnn"):
    start = time.perf_counter()
    try:
        yield
    finally:
        cv_inference_seconds.labels(model=model).observe(time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Alert dispatch outcomes
# ---------------------------------------------------------------------------
#
# Success rate by channel is what the on-call rotation actually needs
# on their dashboard. ``outcome`` deliberately has a small closed set
# so PromQL like `sum(rate(rockfallguard_dispatch_total{outcome="sent"}[5m]))
# / sum(rate(rockfallguard_dispatch_total[5m]))` gives a real ratio.

dispatch_total = Counter(
    "rockfallguard_dispatch_total",
    "Alert dispatch attempts, labelled by channel and terminal outcome.",
    labelnames=("channel", "outcome"),
)


def record_dispatch_outcome(*, channel: str, outcome: str) -> None:
    """channel: email | sms | webhook.
    outcome: sent | simulated | failed | dead_lettered | skipped."""
    dispatch_total.labels(channel=channel, outcome=outcome).inc()


# ---------------------------------------------------------------------------
# Cache hit/miss ratio
# ---------------------------------------------------------------------------

cache_hit_total = Counter(
    "rockfallguard_cache_hit_total",
    "Cache lookups that returned a value.",
    labelnames=("scope",),
)
cache_miss_total = Counter(
    "rockfallguard_cache_miss_total",
    "Cache lookups that returned nothing (miss OR outage).",
    labelnames=("scope",),
)


def record_cache_hit(scope: str) -> None:
    cache_hit_total.labels(scope=scope).inc()


def record_cache_miss(scope: str) -> None:
    cache_miss_total.labels(scope=scope).inc()


# ---------------------------------------------------------------------------
# HTTP request counter (populated by RequestContextMiddleware)
# ---------------------------------------------------------------------------
#
# Path label is bounded to routes present in the app, not raw
# request.url.path (which would include /api/mines/1, /api/mines/2, ...
# and explode cardinality). Middleware uses request.scope["route"].path
# when available.

http_requests_total = Counter(
    "rockfallguard_http_requests_total",
    "HTTP requests processed, by method + templated path + status.",
    labelnames=("method", "path", "status"),
)

http_request_seconds = Histogram(
    "rockfallguard_http_request_seconds",
    "HTTP request latency, by method + templated path.",
    labelnames=("method", "path"),
)


def record_http_request(*, method: str, path: str, status: int, elapsed_s: float) -> None:
    http_requests_total.labels(method=method, path=path, status=str(status)).inc()
    http_request_seconds.labels(method=method, path=path).observe(elapsed_s)


__all__ = [
    "cache_hit_total",
    "cache_miss_total",
    "cv_inference_seconds",
    "dispatch_total",
    "http_request_seconds",
    "http_requests_total",
    "ml_inference_seconds",
    "observe_cv_inference",
    "observe_ml_inference",
    "record_cache_hit",
    "record_cache_miss",
    "record_dispatch_outcome",
    "record_http_request",
]
