# Rapid Scribe dev launcher — refreshes PATH so npm works without restarting the terminal.
$nodeDir = "C:\Program Files\nodejs"
if (-not (Test-Path "$nodeDir\npm.cmd")) {
    Write-Error "Node.js not found at $nodeDir. Install from https://nodejs.org or run: winget install OpenJS.NodeJS.LTS"
    exit 1
}
$env:Path = "$nodeDir;" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
Set-Location $PSScriptRoot
if (-not (Test-Path "node_modules")) {
    npm install
}
# Free port 5173 / stale Electron if a previous dev session is stuck
Get-Process electron -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
npm run dev
