# RockfallGuard

Slope-stability monitoring for open-pit mines. FastAPI service that fuses sensor
telemetry with live weather, scores rockfall risk with a gradient-boosted model,
explains each score with SHAP, and dispatches alerts through a background worker.

The headline engineering decision in this repository is a negative result: an
earlier version of this project claimed **99.83% accuracy**. That claim is
withdrawn. [`MODEL_CARD.md`](MODEL_CARD.md) documents the four mechanisms that
produced it and reports what the models actually do. If you read one file here,
read that one.

---

## Running it

Requires Python 3.12 and, for the queue-backed features, Docker.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements.txt   # Windows
python ml/train_model.py        # generates the five risk-model artifacts
python ml/train_cnn.py          # trains the aerial CNN (~3 min, CPU)
```

On Windows, `run-demo.bat` does all of the above plus Redis, the worker and the
API in one step. Otherwise:

```bash
docker compose -f docker-compose.dev.yml up -d redis
cd backend
alembic upgrade head
arq app.workers.settings.WorkerSettings &     # second shell
python -m uvicorn main:app --port 8005
```

Dashboard at `http://localhost:8005`, API docs at `/docs`.
Demo accounts: `admin` / `admin123` (all mines), `grasberg_user` / `user123`
(scoped to one tenant).

`backend/.env` needs `JWT_SECRET` and `DATABASE_URL`; there are deliberately no
defaults for either, so a misconfigured deploy fails loudly at startup rather
than quietly running on a known secret.

---

## Architecture

```
frontend/          static dashboard, role-aware, polls telemetry once a second
backend/
  main.py          HTTP surface only -- thin shell over the service layer
  app/core/        config, JWT, rate limiting, logging, metrics, middleware
  app/services/    risk, alerting, mines, auth
  app/repositories/ data access, tenant-scoped
  app/workers/     arq tasks: CSV scoring, image analysis, alert dispatch
  app/rag/         alert Q&A over pgvector
  ml_engine.py     XGBoost risk model + SHAP
  cv_engine.py     aerial CNN
ml/                training pipelines for both models
```

Long-running work does not block the request. `/api/upload_csv` and
`/api/analyze_drone_image` persist the upload, enqueue an arq job and return
`202` with a `poll_url`; the frontend polls it to completion.

---

## The two models

**Rockfall risk** (`ml/train_model.py`) — XGBoost on synthetic telemetry, split
temporally per site. Critical-class recall 0.79 at precision 0.38. The low
precision is deliberate: missing a slope failure costs more than investigating a
false alarm. Do not quote the 0.91 accuracy figure on its own; see the model
card for why.

**Aerial terrain CNN** (`ml/train_cnn.py`) — binary classifier separating
annotated open-pit regions from surrounding background, trained on 1,012
labelled scenes from the Open-Pit Mine Object Detection Dataset. Held-out
ROC-AUC 0.956, with the split grouped by source scene so no scene appears in
both train and test.

It is **not** a crack detector. The only labels available were single-class
boxes around pit workings, so that is the only thing it can honestly report.
Inference scans the frame in overlapping windows at the scale the model was
trained on, because the model sees crops and a whole downscaled scene is out of
distribution.

---

## Verification

Each refactoring phase ships a script that re-checks its own invariants:

```bash
cd backend
python verify_core.py          # config, JWT, logging, error envelope
python verify_persistence.py   # repositories, tenant scoping
python verify_services.py      # service layer
python verify_redis.py         # cache, rate limiter, blocklist
python verify_workers.py       # queue, retries, dead-letter
python verify_cv.py            # CNN weights, threshold, tiled scan, response
python -m pytest tests/        # integration
```

---

## Known limitations

- The risk model trains on **synthetic** telemetry. No public dataset of real
  instrumented open-pit slope measurements exists. Metrics measure recovery of a
  generating function, not real-world prediction.
- The CNN's negatives are drawn from the same scenes as its positives, so it
  separates pit regions from their surroundings *within* mining imagery. It has
  never been shown a scene with no mine in it.
- SMTP and Twilio credentials are optional; without them alert dispatch runs in
  simulated mode and logs instead of sending.
- Without Redis the API still serves the dashboard, telemetry, risk scoring and
  SHAP; only the queue-backed endpoints degrade, and they say so.
