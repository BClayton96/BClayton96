# Runs one paper-trading decision via Alpaca and logs the result.
# Meant to be called by Windows Task Scheduler (see README.md
# "Running automatically").
#
# Reads credentials from .env in the project root - copy .env.example to
# .env and fill in your own Alpaca keys before using this. Never commit
# .env (it's gitignored).
#
# By default this stays on Alpaca's PAPER endpoint (no --live flag). Do
# not add --live to this script casually - going live should stay a
# deliberate, one-off decision, not something a scheduled task does
# silently.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSScriptRoot
Set-Location $ScriptDir

$envFile = Join-Path $ScriptDir ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

$Ticker = if ($env:TRADING_BOT_TICKER) { $env:TRADING_BOT_TICKER } else { "AAPL" }
$WeeklyBudget = if ($env:TRADING_BOT_WEEKLY_BUDGET) { $env:TRADING_BOT_WEEKLY_BUDGET } else { "50" }

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$LogFile = Join-Path $ScriptDir "logs\live-trade.log"
"=== $(Get-Date -AsUTC -Format 'yyyy-MM-ddTHH:mm:ssZ') ===" | Out-File -Append -FilePath $LogFile
& $Python main.py live-trade --ticker $Ticker --weekly-budget $WeeklyBudget *>> $LogFile
"" | Out-File -Append -FilePath $LogFile
