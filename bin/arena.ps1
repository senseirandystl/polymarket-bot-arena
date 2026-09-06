# bin/arena.ps1 — Windows-native helper for Polymarket Bot Arena
#
# Mirrors bin/arena (bash) for PowerShell/cmd without requiring Git Bash/WSL.
# Uses .venv\Scripts\python.exe and defaults ARENA_NO_DASHBOARD=1 so the arena
# process does not auto-spawn the dashboard (manage it separately or use -WithDashboard).
#
# Usage:
#   .\bin\arena.ps1                      # paper arena only (no dashboard child)
#   .\bin\arena.ps1 -WithDashboard       # start dashboard in a new window, then arena
#   .\bin\arena.ps1 -DashboardOnly       # dashboard server only
#   .\bin\arena.ps1 --mode live          # extra args forwarded to arena.py
#
# Unix / macOS / WSL: keep using ./bin/arena (this file does not replace it).

[CmdletBinding()]
param(
    [switch]$WithDashboard,
    [switch]$DashboardOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArenaArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error @"
bin/arena.ps1: cannot find the venv interpreter at:
    $Python

Create the venv with:
    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
"@
}

$env:PYTHONUNBUFFERED = "1"

function Start-Dashboard {
    $dash = Join-Path $RepoRoot "dashboard\server.py"
    Write-Host "Starting dashboard: $Python $dash"
    Start-Process -FilePath $Python -ArgumentList @($dash) -WorkingDirectory $RepoRoot
}

if ($DashboardOnly) {
    & $Python (Join-Path $RepoRoot "dashboard\server.py")
    exit $LASTEXITCODE
}

if ($WithDashboard) {
    Start-Dashboard
    Start-Sleep -Seconds 2
}

# Default: do not auto-spawn dashboard from the arena process.
if (-not $env:ARENA_NO_DASHBOARD) {
    $env:ARENA_NO_DASHBOARD = "1"
}

$arenaPy = Join-Path $RepoRoot "arena.py"
Write-Host "Starting arena: $Python $arenaPy $($ArenaArgs -join ' ')  (ARENA_NO_DASHBOARD=$($env:ARENA_NO_DASHBOARD))"
& $Python $arenaPy @ArenaArgs
exit $LASTEXITCODE
