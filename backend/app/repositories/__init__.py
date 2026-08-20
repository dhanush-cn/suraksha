"""Repository layer -- the ONLY place raw SQLAlchemy queries live.

Route handlers and services take a :class:`sqlalchemy.ext.asyncio.AsyncSession`
and instantiate a repository. Doing the SQL in one place keeps three
concerns from bleeding into main.py:

* Query performance -- index-friendly ORDER BY / LIMIT / WHERE stays
  reviewable in isolation.
* Access-pattern semantics -- eager vs lazy loading, projection vs full
  entity, list-of-dict vs ORM object -- are consistent per resource.
* Test surface -- unit tests hit the repository against SQLite; route
  tests can stub the repository without spinning up a database.

Repositories never raise ``HTTPException``. If a caller asks for a
missing resource, the repo returns ``None`` (or an empty list) and the
service/handler decides whether that's a 404 or a domain-appropriate
:class:`app.core.exceptions.NotFoundError`.
"""

from app.repositories.alert import AlertRepository
from app.repositories.mine import MineRepository
from app.repositories.user import UserRepository

__all__ = ["AlertRepository", "MineRepository", "UserRepository"]
