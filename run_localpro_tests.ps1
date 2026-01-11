$ErrorActionPreference = "Stop"

function Resolve-ApiBase {
    $envPort = $env:PORT
    $port = $null
    if ($envPort -and $envPort -match "^\d+$") {
        $port = [int]$envPort
    }
    if (-not $port -and (Test-Path ".env")) {
        $line = Get-Content -Path ".env" | Where-Object { $_ -match "^\s*PORT\s*=" } | Select-Object -First 1
        if ($line -and $line -match "=(\d+)") {
            $port = [int]$matches[1]
        }
    }
    if (-not $port -and (Test-Path "config.json")) {
        try {
            $cfg = Get-Content -Path "config.json" -Raw | ConvertFrom-Json
            if ($cfg.port) { $port = [int]$cfg.port }
        } catch {
        }
    }
    if (-not $port -and (Test-Path "server-8000.err.log")) {
        $line = Get-Content -Path "server-8000.err.log" -Tail 50 | Select-String -Pattern "Uvicorn running on" | Select-Object -Last 1
        if ($line -and $line.Line -match ":(\d+)") {
            $port = [int]$matches[1]
        }
    }
    if (-not $port) { $port = 8000 }
    return "http://127.0.0.1:$port"
}

$apiBase = Resolve-ApiBase

$tests = @(
    @{
        name = "real_estate_exhaustive"
        prompt = "What golf club neighborhoods in the Naples and Bonita Springs area have homes built between 2010 and now, single family homes or detached villas that can be purchased for less than 1.0M, and aren't more than 15 miles from the nearest beach?"
        payload = @{
            reasoning_mode = "manual"
            manual_level = "ULTRA"
            planning_mode = "extensive"
            plan_reasoning_mode = "extensive"
            model_tier = "pro"
            search_depth_mode = "auto"
        }
    }
    @{
        name = "ir_runbook_auto"
        prompt = "Create an incident response runbook for a mid-size SaaS company dealing with credential stuffing. Include detection signals, containment steps in the first 2 hours, communications plan, 24-hour checklist, and long-term fixes."
        payload = @{
            reasoning_mode = "auto"
            planning_mode = "auto"
            plan_reasoning_mode = "auto"
            model_tier = "auto"
            search_depth_mode = "auto"
        }
    }
    @{
        name = "oncall_handoff_fast"
        prompt = "Create a detailed on-call handoff SOP for a 24/7 SaaS team. Include a pre-handoff checklist, status update template, escalation paths, and a 7-item 'things to verify' section."
        payload = @{
            reasoning_mode = "auto"
            planning_mode = "normal"
            plan_reasoning_mode = "normal"
            model_tier = "fast"
            search_depth_mode = "auto"
        }
    }
    @{
        name = "data_retention_deep"
        prompt = "Draft a data retention and deletion policy for a SaaS serving EU and US customers. Include a retention schedule by data type, legal holds, deletion workflow, audit logging, and roles/responsibilities."
        payload = @{
            reasoning_mode = "manual"
            manual_level = "HIGH"
            planning_mode = "normal"
            plan_reasoning_mode = "normal"
            model_tier = "deep"
            search_depth_mode = "auto"
        }
    }
    @{
        name = "edr_vendor_auto"
        prompt = "Compare three mid-market EDR tools (CrowdStrike Falcon, SentinelOne, Microsoft Defender for Endpoint) with pricing tiers, key features, and best fit for a 500-employee SaaS. Provide a table and pricing caveats."
        payload = @{
            reasoning_mode = "auto"
            planning_mode = "auto"
            plan_reasoning_mode = "auto"
            model_tier = "auto"
            search_depth_mode = "auto"
        }
    }
)

function Save-Results($items) {
    $items | ConvertTo-Json -Depth 8 | Set-Content -Path "test-run-results.json"
}

$all = @()
if (Test-Path "test-run-results.json") {
    $raw = Get-Content -Path "test-run-results.json" -Raw
    if ($raw) {
        $parsed = $raw | ConvertFrom-Json
        $all += @($parsed)
    }
}

foreach ($test in $tests) {
    $attempt = 0
    $success = $false
    while (-not $success -and $attempt -lt 3) {
        $attempt++
        $payload = @{}
        foreach ($k in $test.payload.Keys) {
            $payload[$k] = $test.payload[$k]
        }
        $payload["question"] = $test.prompt
        $jsonBody = $payload | ConvertTo-Json -Depth 6
        $resp = Invoke-RestMethod -Method Post -Body $jsonBody -ContentType "application/json" "$apiBase/api/run"
        $runId = $resp.run_id
        $run = $null
        while ($true) {
            Start-Sleep -Seconds 6
            $run = Invoke-RestMethod "$apiBase/api/run/$runId"
            if ($run.status -in @("completed", "archived", "done")) {
                break
            }
        }
        $final = $run.final_answer
        $success = $run.status -eq "completed" -and $final -and ($final -notmatch "Unable to finalize") -and ($final -notmatch "couldn't load an instance") -and ($final -notmatch "could not load an instance") -and ($final -notmatch "I found local models but couldn't load")
        $entry = [ordered]@{
            name = $test.name
            run_id = $run.run_id
            status = $run.status
            confidence = $run.confidence
            final_answer = $final
            created_at = $run.created_at
            mode = $test.payload
            prompt = $test.prompt
            attempt = $attempt
            success = $success
        }
        $all += $entry
        Save-Results $all
    }
    if (-not $success) {
        Write-Host "Test $($test.name) failed after $attempt attempts."
    }
}
