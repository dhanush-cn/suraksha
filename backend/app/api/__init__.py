"""API-layer helpers: FastAPI dependencies, session/service factories.

Everything HTTP-facing that doesn't belong in a service (auth guards,
DB session per request, service construction) lives here.
"""
