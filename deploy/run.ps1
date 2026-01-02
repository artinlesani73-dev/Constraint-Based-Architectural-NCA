# Startup Script for NCA Deployment
Write-Host "Starting Architectural NCA Deployment Server..." -ForegroundColor Cyan

# Check for python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Please install Python and add it to your PATH."
    exit
}

# Optional: Install dependencies
# pip install -r deploy/requirements.txt

# Start the server
# Using -m to ensure relative imports work if necessary, but here we'll just run server.py
# Set PYTHONPATH to root so 'deploy.' imports work if needed
$env:PYTHONPATH = (Get-Location).Path
python deploy/server.py
