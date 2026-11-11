# Smart Hive - install and try on Windows (testing, or an old laptop next to the hives).
#   powershell -ExecutionPolicy Bypass -File install\install_windows.ps1
# Creates .venv, installs the package (CPU PyTorch), downloads the models, copies the
# example config and runs the demo on the bundled samples.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$py = (Get-Command py -ErrorAction SilentlyContinue)
if ($null -eq $py) { $python = "python" } else { $python = "py -3" }

Write-Host "==> Python environment" -ForegroundColor Yellow
if (-not (Test-Path ".venv")) { Invoke-Expression "$python -m venv .venv" }
& .venv\Scripts\python.exe -m pip install -q --upgrade pip
& .venv\Scripts\python.exe -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .venv\Scripts\python.exe -m pip install -q -e ".[video]"

Write-Host "==> Models" -ForegroundColor Yellow
& .venv\Scripts\python.exe tools\fetch_weights.py

if (-not (Test-Path "config.yaml")) { Copy-Item config.example.yaml config.yaml }

Write-Host "==> Demo on the bundled samples" -ForegroundColor Yellow
& .venv\Scripts\smart-hive.exe demo
