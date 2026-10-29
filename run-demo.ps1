# RockfallGuard - local demo launcher
#
# Brings up the whole stack:
#   Redis (docker compose)  -> rate limiter, JWT blocklist, arq queue
#   arq worker (own window) -> CSV scoring + drone CNN jobs
#   uvicorn (this window)   -> API + dashboard on http://localhost:8005
#
# Without Docker the API still runs and the dashboard, telemetry, risk
# scoring and SHAP all work; only /api/upload_csv and
# /api/analyze_drone_image need the queue, and they report that plainly.

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
$py   = Join-Path $repo '.venv\Scripts\python.exe'
$arq  = Join-Path $repo '.venv\Scripts\arq.exe'

if (-not (Test-Path $py)) {
    throw "No virtualenv at $py. Create one with: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt"
}

Set-Location (Join-Path $repo 'backend')

# --- config ---------------------------------------------------------------
# Settings has no defaults for JWT_SECRET / DATABASE_URL by design. ASCII on
# purpose: PowerShell's `>` writes UTF-16, which pydantic-settings can't read.
if (-not (Test-Path '.env')) {
    $secret = & $py -c "import secrets; print(secrets.token_hex(32))"
    Set-Content -Path '.env' -Encoding ascii -Value @(
        "JWT_SECRET=$secret",
        "DATABASE_URL=sqlite:///./mines.db",
        "REDIS_URL=redis://127.0.0.1:6379/0",
        "LOG_JSON=false"
    )
    Write-Host "[setup] wrote backend\.env with a fresh dev JWT secret" -ForegroundColor Yellow
}

# 127.0.0.1, never "localhost": on Windows localhost resolves to ::1 first,
# Docker publishes Redis on IPv4 only, and the v6 attempt burns the full
# redis_socket_timeout_seconds (2s) on EVERY request before falling back.
if (-not (Select-String -Path '.env' -Pattern '^REDIS_URL=' -Quiet)) {
    Add-Content -Path '.env' -Encoding ascii -Value "REDIS_URL=redis://127.0.0.1:6379/0"
    Write-Host "[setup] pinned REDIS_URL to 127.0.0.1 in backend\.env" -ForegroundColor Yellow
}

# --- redis ----------------------------------------------------------------
function Test-Port([int]$port) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', $port); $c.Close(); return $true
    } catch { return $false }
}

$redisUp = Test-Port 6379
if (-not $redisUp) {
    Write-Host "[1/4] starting Redis via docker compose" -ForegroundColor Cyan
    $dockerOk = $false
    try { docker info *> $null; $dockerOk = ($LASTEXITCODE -eq 0) } catch { $dockerOk = $false }

    if ($dockerOk) {
        Push-Location $repo
        docker compose -f docker-compose.dev.yml up -d redis
        Pop-Location
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            if (Test-Port 6379) { $redisUp = $true; break }
        }
    } else {
        Write-Host "      Docker is not responding - start Docker Desktop and re-run" -ForegroundColor Yellow
    }
} else {
    Write-Host "[1/4] Redis already listening on 6379" -ForegroundColor Cyan
}

if ($redisUp) {
    Write-Host "      Redis up" -ForegroundColor Green
} else {
    Write-Host "      no Redis: CSV upload and drone analysis will return 503" -ForegroundColor Yellow
}

# --- migrations -----------------------------------------------------------
Write-Host "[2/4] applying migrations" -ForegroundColor Cyan
& $py -m alembic upgrade head

# --- ml artifacts ---------------------------------------------------------
Write-Host "[3/4] checking ML artifacts" -ForegroundColor Cyan
$needed = @(
    'rockfall_classifier.joblib',
    'rockfall_regressor.joblib',
    'scaler.joblib',
    'shap_explainer.joblib',
    'feature_names.joblib'
)
$missing = $needed | Where-Object { -not (Test-Path (Join-Path $repo "models\$_")) }
if ($missing) {
    Write-Host "      missing: $($missing -join ', ') - regenerating (~1 min)" -ForegroundColor Yellow
    Push-Location $repo
    & $py ml\train_model.py
    Pop-Location
} else {
    Write-Host "      all five present" -ForegroundColor Green
}
if (-not (Test-Path (Join-Path $repo 'models\drone_cnn_model.pth'))) {
    Write-Host "      note: no drone_cnn_model.pth - the CNN reports itself UNTRAINED" -ForegroundColor DarkYellow
}

# --- worker ---------------------------------------------------------------
if ($redisUp -and (Test-Path $arq)) {
    # Stop any worker from a previous run first. Two workers on one queue is
    # not an error, but the old one holds the previous cv_engine/ml_engine in
    # memory, so after retraining a model the stale process keeps answering
    # with the weights it loaded at startup.
    $stale = Get-Process -Name arq -ErrorAction SilentlyContinue
    if ($stale) {
        Write-Host "[4/4] stopping $($stale.Count) worker(s) from a previous run" -ForegroundColor Yellow
        $stale | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    Write-Host "[4/4] starting arq worker in its own window" -ForegroundColor Cyan
    Start-Process -FilePath $arq `
                  -ArgumentList 'app.workers.settings.WorkerSettings' `
                  -WorkingDirectory (Join-Path $repo 'backend')
    Start-Sleep -Seconds 3
}

# Refuse to start a second API on a port that already has one: uvicorn would
# fail to bind and the browser would silently keep talking to the old process.
if (Test-Port 8005) {
    Write-Host ""
    Write-Host "Port 8005 is already in use -- close the previous server window first." -ForegroundColor Red
    return
}

Write-Host ""
Write-Host "API on http://localhost:8005   login: admin / admin123" -ForegroundColor Green
Write-Host "Ctrl+C to stop (close the worker window separately)" -ForegroundColor DarkGray

# Open the dashboard only once uvicorn is actually accepting connections -- a
# fixed sleep raced the import of torch/xgboost/shap on a cold cache.
Start-Job -ScriptBlock {
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        try {
            $c = New-Object Net.Sockets.TcpClient
            $c.Connect('127.0.0.1', 8005); $c.Close()
            Start-Process 'http://localhost:8005'
            return
        } catch { }
    }
} | Out-Null

# No --reload: the reloader double-logs startup and watchfiles chatter
# clutters a screen recording.
& $py -m uvicorn main:app --host 127.0.0.1 --port 8005
