param(
    [switch]$Reload,
    [switch]$ForcePort
)

$ErrorActionPreference = "Stop"

try {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null
} catch {
}

$repoRoot = (Resolve-Path $PSScriptRoot).Path
$repoPattern = [regex]::Escape($repoRoot)

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

function Find-AvailablePort {
    param(
        [int]$StartPort,
        [int]$Attempts = 50
    )
    $maxPort = [Math]::Min($StartPort + $Attempts, 65535)
    for ($candidate = $StartPort; $candidate -le $maxPort; $candidate++) {
        if (-not (Get-ListeningPids -Port $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Show-PortOwners {
    param([int]$Port)
    $pids = Get-ListeningPids -Port $Port
    if (-not $pids) {
        return
    }
    Write-Host "Port $Port is in use by PID(s): $($pids -join ', ')"
    foreach ($procId in $pids) {
        $proc = Get-ProcessInfo -ProcessId $procId
        if ($proc) {
            if ($proc.Name) { Write-Host "  Name: $($proc.Name)" }
            if ($proc.ExecutablePath) { Write-Host "  Path: $($proc.ExecutablePath)" }
            if ($proc.CommandLine) { Write-Host "  Cmd : $($proc.CommandLine)" }
        }
    }
}

function Stop-PortOwners {
    param([int]$Port)
    $pids = Get-ListeningPids -Port $Port
    foreach ($procId in $pids) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        } catch {
        }
        try {
            taskkill /PID $procId /F | Out-Null
        } catch {
        }
    }
}

function Get-ProcessInfo {
    param([int]$ProcessId)
    return Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
}

function Test-LocalProProcess {
    param([object]$ProcessInfo)
    if (-not $ProcessInfo) {
        return $false
    }
    $cmd = $ProcessInfo.CommandLine
    $exe = $ProcessInfo.ExecutablePath
    if (-not $cmd) {
        $cmd = ""
    }
    if (-not $exe) {
        $exe = ""
    }
    $isRepoProcess = ($cmd -match $repoPattern) -or ($exe -match $repoPattern)
    if (-not $isRepoProcess) {
        return $false
    }
    $isAppMain = $cmd -match "app\.main"
    $isUvicorn = $cmd -match "uvicorn"
    $isPythonHost = ($cmd -match '\bpython(\.exe)?\b') -or ($cmd -match '\bpy(\.exe)?\b') -or ($exe -match 'python(\.exe)?$')
    return $isPythonHost -or $isUvicorn -or $isAppMain
}

function Get-LocalProRootPid {
    param([int]$ProcessId)
    $current = Get-ProcessInfo -ProcessId $ProcessId
    if (-not (Test-LocalProProcess -ProcessInfo $current)) {
        return $null
    }
    $root = $current
    while ($true) {
        $parentId = $root.ParentProcessId
        if (-not $parentId) {
            break
        }
        $parent = Get-ProcessInfo -ProcessId $parentId
        if (-not (Test-LocalProProcess -ProcessInfo $parent)) {
            break
        }
        $root = $parent
    }
    return [int]$root.ProcessId
}

function Stop-ProcessTree {
    param([int]$RootPid)
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$RootPid" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -RootPid $child.ProcessId
    }
    Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
}

function Stop-LocalProOnPort {
    param([int]$Port)
    $pids = Get-ListeningPids -Port $Port
    $stopped = @{}
    foreach ($procId in $pids) {
        $rootPid = Get-LocalProRootPid -ProcessId $procId
        if ($null -ne $rootPid -and -not $stopped.ContainsKey($rootPid)) {
            $stopped[$rootPid] = $true
            Stop-ProcessTree -RootPid $rootPid
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
    if ($ForcePort) {
        Write-Host "Port $port is busy. Attempting to stop the process using it..."
        Stop-PortOwners -Port $port
        Start-Sleep -Milliseconds 300
        if (Get-ListeningPids -Port $port) {
            Show-PortOwners -Port $port
            Write-Error "Port $port is still busy. Re-run in an elevated PowerShell or stop it manually."
            exit 1
        }
    } else {
        $fallbackPort = Find-AvailablePort -StartPort ($port + 1)
        if ($null -ne $fallbackPort) {
            Show-PortOwners -Port $port
            Write-Host "Port $port is busy. Using $fallbackPort instead."
            $port = $fallbackPort
        } else {
            Show-PortOwners -Port $port
            Write-Error "Port $port is busy. Stop the process using it and re-run start.ps1 (or use -ForcePort)."
            exit 1
        }
    }
}

$env:PORT = "$port"
if ($Reload) {
    $env:LOCALPRO_RELOAD = "1"
} else {
    Remove-Item Env:LOCALPRO_RELOAD -ErrorAction SilentlyContinue
}

Write-Host "Starting LocalPro on http://127.0.0.1:$port"
try {
    python -m app.main
} finally {
    Stop-LocalProOnPort -Port $port
}
