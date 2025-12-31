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

$requirementsPath = "requirements.txt"
$depsOk = $false
$checkScript = @'
import importlib.metadata as md
import re
import sys
from pathlib import Path

req_path = Path(sys.argv[1])
missing = []
mismatch = []

for raw in req_path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    line = line.split(";", 1)[0].strip()
    if not line:
        continue
    if "==" in line:
        name, ver = line.split("==", 1)
        name = name.strip()
        ver = ver.strip()
        base = name.split("[", 1)[0]
        if not base:
            continue
        try:
            inst = md.version(base)
        except md.PackageNotFoundError:
            missing.append(name)
            continue
        if inst != ver:
            mismatch.append(f"{base}=={ver} (have {inst})")
    else:
        name = re.split(r"[<>=]", line, 1)[0].strip()
        base = name.split("[", 1)[0]
        if not base:
            continue
        try:
            md.version(base)
        except md.PackageNotFoundError:
            missing.append(base)

if missing or mismatch:
    if missing:
        print("missing=" + ",".join(missing))
    if mismatch:
        print("mismatch=" + ",".join(mismatch))
    sys.exit(1)
sys.exit(0)
'@
try {
    $checkScript | python - $requirementsPath
    $depsOk = $LASTEXITCODE -eq 0
} catch {
    $depsOk = $false
}
if (-not $depsOk) {
    Write-Host "Installing dependencies from $requirementsPath..."
    python -m pip install -r $requirementsPath --disable-pip-version-check --no-input --default-timeout 15 --retries 1
} else {
    Write-Host "Dependencies already installed."
}

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}

function Get-EnvValueFromFile {
    param(
        [string]$Path,
        [string]$Key
    )
    if (-not (Test-Path $Path)) {
        return $null
    }
    $pattern = "^\s*$([regex]::Escape($Key))\s*="
    $line = Get-Content $Path | Where-Object { $_ -match $pattern } | Select-Object -First 1
    if (-not $line) {
        return $null
    }
    $value = $line.Split("=", 2)[1].Trim()
    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    return $value
}

function Get-ConfigValue {
    param(
        [string]$Path,
        [string]$Key
    )
    if (-not (Test-Path $Path)) {
        return $null
    }
    try {
        $json = Get-Content -Path $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
    if (-not $json) {
        return $null
    }
    return $json.$Key
}

function Convert-ToIntOrNull {
    param([object]$Value)
    if ($null -eq $Value) {
        return $null
    }
    $text = "$Value".Trim()
    if ($text -match "^\d+$") {
        return [int]$text
    }
    return $null
}

function Test-EnvOverridesEnabled {
    param([string]$EnvPath)
    $raw = $env:LOCALPRO_ENV_OVERRIDES_CONFIG
    if (-not $raw) {
        $raw = Get-EnvValueFromFile -Path $EnvPath -Key "LOCALPRO_ENV_OVERRIDES_CONFIG"
    }
    if (-not $raw) {
        return $false
    }
    $val = "$raw".Trim().ToLower()
    return $val -in @("1", "true", "yes", "on")
}

function Resolve-ListenerAddress {
    param([string]$BindHost)
    if (-not $BindHost -or $BindHost -eq "0.0.0.0") {
        return [System.Net.IPAddress]::Any
    }
    if ($BindHost -eq "localhost") {
        return [System.Net.IPAddress]::Loopback
    }
    try {
        return [System.Net.IPAddress]::Parse($BindHost)
    } catch {
        try {
            $addr = [System.Net.Dns]::GetHostAddresses($BindHost) | Where-Object { $_.AddressFamily -eq "InterNetwork" } | Select-Object -First 1
            if ($addr) {
                return $addr
            }
        } catch {
        }
    }
    return [System.Net.IPAddress]::Any
}

function Test-PortAvailable {
    param(
        [int]$Port,
        [string]$BindHost = "0.0.0.0"
    )
    try {
        $addr = Resolve-ListenerAddress -BindHost $BindHost
        $listener = [System.Net.Sockets.TcpListener]::new($addr, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Get-ListeningPids {
    param([int]$Port)
    # Get-NetTCPConnection can hang on some Windows setups, so prefer netstat for a quick check.
    $pids = @()
    $netstatCmd = Get-Command netstat -ErrorAction SilentlyContinue
    if ($netstatCmd) {
        try {
            $lines = netstat -ano -p TCP 2>$null | Select-String -Pattern "LISTENING" -SimpleMatch
            foreach ($line in $lines) {
                $parts = $line.Line.Trim() -split "\s+"
                if ($parts.Length -lt 5) {
                    continue
                }
                if ($parts[3] -ne "LISTENING") {
                    continue
                }
                $local = $parts[1]
                if ($local -match ":(\d+)$") {
                    $localPort = [int]$matches[1]
                    if ($localPort -eq $Port) {
                        $pid = $parts[4]
                        if ($pid -match "^\d+$") {
                            $pids += [int]$pid
                        }
                    }
                }
            }
        } catch {
            $pids = @()
        }
        return $pids | Select-Object -Unique
    }
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        return @()
    }
    return $conns | Select-Object -ExpandProperty OwningProcess -Unique
}

function Find-AvailablePort {
    param(
        [int]$StartPort,
        [string]$BindHost = "0.0.0.0",
        [int]$Attempts = 50
    )
    $maxPort = [Math]::Min($StartPort + $Attempts, 65535)
    for ($candidate = $StartPort; $candidate -le $maxPort; $candidate++) {
        if (Test-PortAvailable -Port $candidate -BindHost $BindHost) {
            return $candidate
        }
    }
    return $null
}

function Get-LocalIPv4Addresses {
    $ips = @()
    try {
        $lines = ipconfig | Select-String -Pattern "IPv4 Address" -SimpleMatch
        foreach ($line in $lines) {
            $match = [regex]::Match($line.Line, "(\\d{1,3}(?:\\.\\d{1,3}){3})")
            if ($match.Success) {
                $ip = $match.Value
                if ($ip -and $ip -notlike "127.*" -and $ip -notlike "169.254*") {
                    $ips += $ip
                }
            }
        }
    } catch {
    }
    if (-not $ips) {
        try {
            $name = $env:COMPUTERNAME
            if ($name) {
                $dnsIps = [System.Net.Dns]::GetHostAddresses($name) |
                    Where-Object { $_.AddressFamily -eq "InterNetwork" } |
                    ForEach-Object { $_.ToString() }
                foreach ($ip in $dnsIps) {
                    if ($ip -and $ip -notlike "127.*" -and $ip -notlike "169.254*") {
                        $ips += $ip
                    }
                }
            }
        } catch {
        }
    }
    return $ips | Select-Object -Unique
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

$envPath = ".env"
$configPath = "config.json"
$envOverrides = Test-EnvOverridesEnabled -EnvPath $envPath

$envPort = Convert-ToIntOrNull $env:PORT
if (-not $envPort) {
    $envPort = Convert-ToIntOrNull (Get-EnvValueFromFile -Path $envPath -Key "PORT")
}
$configPort = Convert-ToIntOrNull (Get-ConfigValue -Path $configPath -Key "port")
$defaultPort = 8000
if ($envOverrides) {
    $port = $envPort
    if (-not $port) {
        $port = $configPort
    }
} else {
    $port = $configPort
    if (-not $port) {
        $port = $envPort
    }
}
if (-not $port) {
    $port = $defaultPort
}

$envHost = $env:HOST
if (-not $envHost) {
    $envHost = Get-EnvValueFromFile -Path $envPath -Key "HOST"
}
$configHost = Get-ConfigValue -Path $configPath -Key "host"
if ($envOverrides) {
    $bindHost = if ($envHost) { $envHost } else { $configHost }
} else {
    $bindHost = if ($configHost) { $configHost } else { $envHost }
}
if ($bindHost) {
    $bindHost = "$bindHost".Trim()
}
if (-not $bindHost) {
    $bindHost = "0.0.0.0"
}
Stop-LocalProOnPort -Port $port

if (-not (Test-PortAvailable -Port $port -BindHost $bindHost)) {
    if ($ForcePort) {
        Write-Host "Port $port is busy. Attempting to stop the process using it..."
        Stop-PortOwners -Port $port
        Start-Sleep -Milliseconds 300
        if (-not (Test-PortAvailable -Port $port -BindHost $bindHost)) {
            Show-PortOwners -Port $port
            Write-Error "Port $port is still busy. Re-run in an elevated PowerShell or stop it manually."
            exit 1
        }
    } else {
        $fallbackPort = Find-AvailablePort -StartPort ($port + 1) -BindHost $bindHost
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

$displayHost = $bindHost
if (-not $displayHost -or $displayHost -eq "0.0.0.0") {
    $displayHost = "127.0.0.1"
}
Write-Host "Starting LocalPro on http://${displayHost}:$port"
$lanIps = Get-LocalIPv4Addresses
if ($lanIps) {
    if ($bindHost -eq "0.0.0.0") {
        foreach ($ip in $lanIps) {
            Write-Host "LAN: http://${ip}:$port"
        }
    } elseif ($lanIps -contains $bindHost) {
        Write-Host "LAN: http://${bindHost}:$port"
    } else {
        Write-Host "LAN IPs: $($lanIps -join ', ') (set HOST=0.0.0.0 for LAN access)"
    }
}
try {
    $uvicornArgs = @("app.main:app", "--host", $bindHost, "--port", "$port")
    if ($Reload) {
        $uvicornArgs += "--reload"
    }
    python -m uvicorn @uvicornArgs
} finally {
    Stop-LocalProOnPort -Port $port
}
