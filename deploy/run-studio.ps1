$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $projectPython)) {
    throw 'Project Python environment missing. See docs/next-phase/STUDIO_S1.md.'
}
Push-Location -LiteralPath $projectRoot
try {
    Write-Host 'NCA Studio: http://127.0.0.1:8001 — press Ctrl+C to stop.'
    & $projectPython -m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001
} finally {
    Pop-Location
}
