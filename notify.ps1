# MemForge - Email Notification Script
# Usage: .\notify.ps1 -Title "Subject" -Body "Email body content"
#
# Required env vars:
#   MEMFORGE_SMTP_FROM     - sender email (e.g. yourname@qq.com)
#   MEMFORGE_SMTP_PASSWORD - SMTP authorization code (NOT your email password)
# Optional env vars:
#   MEMFORGE_SMTP_TO       - recipient email (defaults to sender)
#   MEMFORGE_SMTP_SERVER   - default smtp.qq.com
#   MEMFORGE_SMTP_PORT     - default 465

param(
    [Parameter(Mandatory = $true)]
    [string]$Title,
    [string]$Body = ""
)

$smtpServer = if ($env:MEMFORGE_SMTP_SERVER) { $env:MEMFORGE_SMTP_SERVER } else { "smtp.qq.com" }
$smtpPort   = if ($env:MEMFORGE_SMTP_PORT)   { [int]$env:MEMFORGE_SMTP_PORT } else { 465 }
$from       = $env:MEMFORGE_SMTP_FROM
$password   = $env:MEMFORGE_SMTP_PASSWORD
$to         = if ($env:MEMFORGE_SMTP_TO) { $env:MEMFORGE_SMTP_TO } else { $from }

if (-not $from -or -not $password) {
    Write-Error "ERROR: Set MEMFORGE_SMTP_FROM and MEMFORGE_SMTP_PASSWORD (authorization code, not login password) env vars first."
    exit 1
}

$securePwd = ConvertTo-SecureString $password -AsPlainText -Force
$credential = New-Object System.Management.Automation.PSCredential($from, $securePwd)

$fullBody = if ($Body) { $Body } else { "(no body)" }
$fullBody += "`n`n---`nSent from MemForge at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

try {
    Send-MailMessage -From $from -To $to -Subject $Title -Body $fullBody `
        -SmtpServer $smtpServer -Port $smtpPort -UseSsl -Credential $credential -Encoding UTF8
    Write-Output "[notify] Email sent: $Title"
} catch {
    Write-Error "[notify] Failed to send email: $_"
    exit 1
}
