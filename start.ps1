param(
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

try {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null
} catch {
}

function Get-PythonLauncher {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ exe = "python"; args = @() }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ exe = "py"; args = @("-3") }
    }
    return $null
}

$py = Get-PythonLauncher
if (-not $py) {
    Write-Error "Python not found. Install from python.org and check 'Add Python to PATH'."
    exit 1
}

if (-not (Test-Path .venv)) {
    & $py.exe @($py.args) -m venv .venv
}

$activatePath = Join-Path ".venv" "Scripts\\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Error "Virtual environment is missing. Delete .venv and run start.ps1 again."
    exit 1
}

. $activatePath

python -m pip install -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}

function Get-PortFromEnv {
    param([string]$Path)
    $defaultPort = 8000
    if (-not (Test-Path $Path)) {
        return $defaultPort
    }
    $line = Get-Content $Path | Where-Object { $_ -match "^\s*PORT\s*=" } | Select-Object -First 1
    if (-not $line) {
        return $defaultPort
    }
    $value = $line.Split("=", 2)[1].Trim()
    if ($value -match "^\d+$") {
        return [int]$value
    }
    return $defaultPort
}

function Get-ListeningPids {
    param([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        return @()
    }
    return $conns | Select-Object -ExpandProperty OwningProcess -Unique
}

function Stop-LocalProOnPort {
    param([int]$Port)
    $pids = Get-ListeningPids -Port $Port
    foreach ($procId in $pids) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
        if (-not $proc) {
            continue
        }
        $cmd = $proc.CommandLine
        $exe = $proc.ExecutablePath
        $isLocalProPath = ($cmd -match "LocalPro") -or ($exe -match "LocalPro")
        $isAppMain = $cmd -match "app\.main"
        if (($isLocalProPath -and $isAppMain) -or ($cmd -match "uvicorn" -and $isAppMain)) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
}

$port = $null
if ($env:PORT -and $env:PORT -match "^\d+$") {
    $port = [int]$env:PORT
} else {
    $port = Get-PortFromEnv -Path ".env"
}
Stop-LocalProOnPort -Port $port

if (Get-ListeningPids -Port $port) {
    $originalPort = $port
    $found = $false
    for ($p = $port + 1; $p -le $port + 20; $p++) {
        if (-not (Get-ListeningPids -Port $p)) {
            $port = $p
            $found = $true
            break
        }
    }
    if (-not $found) {
        Write-Error "No free port found near $originalPort."
        exit 1
    }
    Write-Host "Port $originalPort is busy, using $port."
}

$env:PORT = "$port"
if ($Reload) {
    $env:LOCALPRO_RELOAD = "1"
} else {
    Remove-Item Env:LOCALPRO_RELOAD -ErrorAction SilentlyContinue
}

Write-Host "Starting LocalPro on http://127.0.0.1:$port"
python -m app.main
