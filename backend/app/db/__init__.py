"""SQLAlchemy 2.0 async persistence layer.

Public surface:

* :mod:`app.db.base` -- ``Base`` DeclarativeBase and common column mixins.
* :mod:`app.db.models` -- ORM classes (Mine, AlertLog, User).
* :mod:`app.db.engine` -- Async engine, session factory, session context.

Route handlers should not import from this module directly; they go
through :mod:`app.repositories`.
"""
