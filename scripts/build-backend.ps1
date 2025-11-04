# PowerShell script for building on Windows
$ErrorActionPreference = "Stop"

Write-Host "Building whisper-subtitler API backend..." -ForegroundColor Green

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
uv pip install pyinstaller

Write-Host "Ensuring project dependencies are installed..." -ForegroundColor Yellow
uv sync

Write-Host "Running PyInstaller..." -ForegroundColor Yellow
uv run pyinstaller api.spec

Write-Host "Build complete!" -ForegroundColor Green
Write-Host "Binary location: dist\" -ForegroundColor Green

Get-ChildItem dist\ | Format-Table Name, Length

$BinaryName = (Get-ChildItem dist\)[0].Name
Write-Host ""
Write-Host "To test the binary:" -ForegroundColor Cyan
Write-Host "  .\dist\$BinaryName" -ForegroundColor Cyan
Write-Host ""
Write-Host "To copy to Tauri:" -ForegroundColor Cyan
Write-Host "  New-Item -ItemType Directory -Force -Path whisper-subtitler-app\src-tauri\bin" -ForegroundColor Cyan
Write-Host "  Copy-Item dist\$BinaryName whisper-subtitler-app\src-tauri\bin\" -ForegroundColor Cyan
