"""Service layer.

Services own business rules and orchestrate repositories. They never
know about HTTP -- no ``FastAPI``, no ``Request``, no ``HTTPException``.
Domain-signalling errors are raised as :class:`app.core.exceptions.AppError`
subclasses; the API layer translates them into responses.

The four services correspond to the four resources the API exposes:

* :class:`MineService`  -- mines table CRUD + business rules.
* :class:`RiskService`  -- ML inference + risk-driver explanation.
* :class:`AlertService` -- when to trigger, how to record, how to dispatch.
                           Owns the ONE definition of ``should_trigger``.
* :class:`AuthService`  -- login flow (credential check + token mint).

Repositories are injected via constructor. This lets a test pass in
fakes without spinning up a real DB, and lets a future step swap the
sqlite path for a fully-async postgres one without changing service
signatures.
"""

from app.services.alert import AlertService
from app.services.auth import AuthService
from app.services.mine import MineService
from app.services.risk import RiskService

__all__ = ["AlertService", "AuthService", "MineService", "RiskService"]
