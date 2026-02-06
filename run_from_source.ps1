# Run Meetings Transcriber from source on Windows (PowerShell).
# Uses venv's python.exe directly so no script execution policy is needed.
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv", "Scripts", "python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv found. Create one with: python -m venv .venv"
    Write-Host "Then install deps (no activate needed): .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
& $py main.py
if ($LASTEXITCODE -ne 0) { Read-Host "Press Enter to close" }
