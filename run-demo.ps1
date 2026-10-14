# RockfallGuard - local demo launcher
#
# Starts the API + dashboard on http://localhost:8005 using the existing
# .venv (Python 3.12). Core demo path (login, mine list, live telemetry,
# risk prediction, SHAP, alert history) needs nothing but this script.
#
# Not covered here, because both need Redis + the arq worker:
#   * CSV upload   -> /api/upload_csv enqueues a background job
#   * /health/ready -> reports 503 while Redis is unreachable (expected)
# To demo those, run `docker compose -f docker-compose.dev.yml up -d`
# and `.\.venv\Scripts\arq.exe app.workers.settings.WorkerSettings`
# from the backend folder in a second window first.

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
$py   = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    throw "No virtualenv at $py. Create one with: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt"
}

Set-Location (Join-Path $repo 'backend')

# Config. Settings has no defaults for JWT_SECRET / DATABASE_URL by design,
# so write a dev .env once if it isn't there. ASCII on purpose: PowerShell's
# `>` writes UTF-16, which pydantic-settings cannot read.
if (-not (Test-Path '.env')) {
    $secret = & $py -c "import secrets; print(secrets.token_hex(32))"
    Set-Content -Path '.env' -Encoding ascii -Value @(
        "JWT_SECRET=$secret",
        "DATABASE_URL=sqlite:///./mines.db",
        "LOG_JSON=false"
    )
    Write-Host "[setup] wrote backend\.env with a fresh dev JWT secret" -ForegroundColor Yellow
}

Write-Host "[1/3] applying migrations" -ForegroundColor Cyan
& $py -m alembic upgrade head

Write-Host "[2/3] checking ML artifacts" -ForegroundColor Cyan
$needed = @(
    'rockfall_classifier.joblib',
    'rockfall_regressor.joblib',
    'scaler.joblib',
    'shap_explainer.joblib',
    'feature_names.joblib'
)
$missing = $needed | Where-Object { -not (Test-Path (Join-Path $repo "models\$_")) }
if ($missing) {
    Write-Host "      missing: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "      regenerating (one-off, ~1 min)..." -ForegroundColor Yellow
    Push-Location $repo
    & $py ml\train_model.py
    Pop-Location
} else {
    Write-Host "      all five present" -ForegroundColor Green
}

Write-Host "[3/3] starting API on http://localhost:8005" -ForegroundColor Cyan
Write-Host "      dashboard login: admin / admin123" -ForegroundColor Green
Write-Host "      Ctrl+C to stop" -ForegroundColor DarkGray

# Open the dashboard once uvicorn is actually accepting connections.
# A fixed sleep is not enough: torch + xgboost + shap import slowly on a
# cold filesystem cache, so a 6s wait landed on ERR_CONNECTION_REFUSED.
Start-Job -ScriptBlock {
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        try {
            $c = New-Object Net.Sockets.TcpClient
            $c.Connect('127.0.0.1', 8005)
            $c.Close()
            Start-Process 'http://localhost:8005'
            return
        } catch { }
    }
} | Out-Null

# No --reload: the reloader double-logs startup and watchfiles chatter
# clutters a screen recording.
& $py -m uvicorn main:app --host 127.0.0.1 --port 8005
