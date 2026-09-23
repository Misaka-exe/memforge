# MemForge - Experiment Watchdog (ASCII only to avoid PS5.1 encoding issues)
# Monitors Stage 5B full run; sends email every 100 questions + on completion/error.
# Required env vars: MEMFORGE_SMTP_FROM, MEMFORGE_SMTP_PASSWORD

$ErrorActionPreference = "Stop"
$projectRoot = "D:\agentmemory\memforge"
$resultDir = "$projectRoot\results\longmemeval\stage5_real"
$logFile = "$projectRoot\results\longmemeval\stage5_real_run.log"
$systems = @("naive_vector", "hybrid", "full")
$expected = 500
$checkInterval = 60
$milestones = @(100, 200, 300, 400)

function Send-Notification {
    param([string]$Title, [string]$Body)
    $smtpServer = if ($env:MEMFORGE_SMTP_SERVER) { $env:MEMFORGE_SMTP_SERVER } else { "smtp.qq.com" }
    $smtpPort   = if ($env:MEMFORGE_SMTP_PORT)   { [int]$env:MEMFORGE_SMTP_PORT } else { 587 }
    $from = $env:MEMFORGE_SMTP_FROM
    $pwd  = $env:MEMFORGE_SMTP_PASSWORD
    $to   = if ($env:MEMFORGE_SMTP_TO) { $env:MEMFORGE_SMTP_TO } else { $from }
    if (-not $from -or -not $pwd) { Write-Error "SMTP env vars missing"; return }
    $secure = ConvertTo-SecureString $pwd -AsPlainText -Force
    $cred = New-Object System.Management.Automation.PSCredential($from, $secure)
    $Body += "`n`n---`nSent by MemForge Watchdog at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    try {
        Send-MailMessage -From $from -To $to -Subject $Title -Body $Body `
            -SmtpServer $smtpServer -Port $smtpPort -UseSsl -Credential $cred -Encoding UTF8
        Write-Output "[watchdog] Notification sent: $Title"
    } catch {
        Write-Error "[watchdog] Failed to notify: $_"
    }
}

function Get-Progress {
    $counts = @{}
    foreach ($s in $systems) {
        $p = "$resultDir\$s\raw_qa.jsonl"
        if (Test-Path $p) { $counts[$s] = (Get-Content $p).Count } else { $counts[$s] = 0 }
    }
    return $counts
}

function Get-TokenUsage {
    $totalPrompt = 0
    $totalCompletion = 0
    foreach ($s in $systems) {
        $p = "$resultDir\$s\raw_qa.jsonl"
        if (Test-Path $p) {
            $records = Get-Content $p | ConvertFrom-Json
            foreach ($r in $records) {
                $totalPrompt += $r.prompt_tokens
                $totalCompletion += $r.completion_tokens
            }
        }
    }
    return @{ prompt = $totalPrompt; completion = $totalCompletion; total = $totalPrompt + $totalCompletion }
}

Write-Output "[watchdog] Started at $(Get-Date -Format 'HH:mm:ss')"
$notified = $false
$lastMilestone = @{ "naive_vector" = 0; "hybrid" = 0; "full" = 0 }
$startTime = Get-Date

while (-not $notified) {
    $counts = Get-Progress
    $total = ($counts.Values | Measure-Object -Sum).Sum
    $allDone = $true
    foreach ($s in $systems) { if ($counts[$s] -lt $expected) { $allDone = $false } }

    $proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 100 }
    $procAlive = ($null -ne $proc)

    Write-Output ("[watchdog {0}] naive={1} hybrid={2} full={3} total={4}/1500 proc={5}" -f `
        (Get-Date -Format 'HH:mm:ss'), $counts["naive_vector"], $counts["hybrid"], $counts["full"], $total, $procAlive)

    # Milestone notifications (every 100 questions per system)
    foreach ($s in $systems) {
        $cur = $counts[$s]
        foreach ($m in $milestones) {
            if ($cur -ge $m -and $lastMilestone[$s] -lt $m) {
                $lastMilestone[$s] = $m
                $tokens = Get-TokenUsage
                $elapsed = ((Get-Date) - $startTime).TotalMinutes
                $rate = if ($elapsed -gt 0) { [math]::Round($total / $elapsed, 1) } else { 0 }
                $etaMin = if ($rate -gt 0) { [math]::Round((1500 - $total) / $rate, 0) } else { "?" }
                $body = "Progress update: $s reached $m/$expected`n`n"
                $body += "naive_vector: $($counts['naive_vector'])/500`n"
                $body += "hybrid:       $($counts['hybrid'])/500`n"
                $body += "full:         $($counts['full'])/500`n"
                $body += "Total:        $total/1500`n`n"
                $body += "Tokens used:  $($tokens.total) (prompt $($tokens.prompt) + completion $($tokens.completion))`n"
                $body += "Elapsed:      $([math]::Round($elapsed,1)) min`n"
                $body += "Rate:         $rate q/min`n"
                $body += "ETA:          ~$etaMin min remaining`n"
                $body += "Process:      $(if ($procAlive) {'running'} else {'NOT RUNNING'})`n"
                Send-Notification -Title "MemForge progress: $s $m/500" -Body $body
            }
        }
    }

    # Error detection (Python tracebacks only, not PowerShell noise)
    if (Test-Path $logFile) {
        $pyErrors = Select-String -Path $logFile -Pattern "Traceback \(most recent call last\)","Exception:","Error:" -SimpleMatch -ErrorAction SilentlyContinue | Where-Object { $_.Line -notmatch "NativeCommandError|FullyQualifiedErrorId|CategoryInfo" }
        if ($pyErrors -and -not $notified) {
            $lastErr = ($pyErrors | Select-Object -Last 1).Line
            Send-Notification -Title "MemForge: possible error in run log" -Body "Detected Python error in log:`n$lastErr`n`nCheck: $logFile"
            $notified = $true
            break
        }
    }

    # Completion
    if ($allDone) {
        $body = "MemForge Stage 5B full run (500x3) completed!`n`n"
        foreach ($s in $systems) {
            $metricsFile = "$resultDir\$s\metrics.json"
            if (Test-Path $metricsFile) {
                $m = Get-Content $metricsFile | ConvertFrom-Json
                $body += "[$s]`n  QA Accuracy: $($m.overall_accuracy)`n  Answerable: $($m.answerable_accuracy) ($($m.answerable_n) q)`n  Abstention: $($m.abstention_accuracy) ($($m.abstention_n) q)`n  Tokens: $($m.total_tokens)`n  Avg Latency: $($m.avg_latency_ms)ms`n`n"
            }
        }
        $body += "Result dir: $resultDir`nComparison: $resultDir\qa_comparison.csv"
        Send-Notification -Title "MemForge: FULL RUN COMPLETED" -Body $body
        $notified = $true
        break
    }

    # Process died unexpectedly
    if (-not $procAlive -and $total -lt 1500 -and $total -gt 0) {
        Send-Notification -Title "MemForge: process exited unexpectedly" -Body "Python process exited but run incomplete.`nProgress: naive=$($counts['naive_vector']) hybrid=$($counts['hybrid']) full=$($counts['full'])`nCheck log: $logFile"
        $notified = $true
        break
    }

    Start-Sleep -Seconds $checkInterval
}

Write-Output "[watchdog] Exited at $(Get-Date -Format 'HH:mm:ss')"
