$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-run\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv-run. Follow README.md to install the shared environment.' }
$cliArgs = $args
if ($cliArgs.Count -eq 0) { $cliArgs = @('--help') }
Push-Location $projectRoot
try {
    & $python -m ghosthands.cli @cliArgs
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
