"""Background worker package (arq + Redis).

Public surface lives in :mod:`app.workers.queue` (enqueue helpers) and
:mod:`app.workers.tasks` (the task functions). The arq worker itself is
launched via :mod:`app.workers.settings`.
"""
