param(
    [Parameter()]
    [int]$TaskId = 0,

    [Parameter()]
    [string]$CodePath,

    [Parameter()]
    [string]$Code,

    [Parameter()]
    [string]$BaseUrl = "http://127.0.0.1:8000",

    [Parameter()]
    [switch]$RawJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("-" * $Title.Length) -ForegroundColor DarkCyan
}

function Get-SubmissionText {
    param(
        [string]$InlineCode,
        [string]$FilePath
    )

    if (-not [string]::IsNullOrWhiteSpace($InlineCode)) {
        return $InlineCode
    }

    if (-not [string]::IsNullOrWhiteSpace($FilePath)) {
        if (-not (Test-Path -LiteralPath $FilePath)) {
            throw "CodePath not found: $FilePath"
        }
        return Get-Content -LiteralPath $FilePath -Raw
    }

    throw "Provide either -Code or -CodePath."
}

try {
    $submission = Get-SubmissionText -InlineCode $Code -FilePath $CodePath
    $endpoint = "$($BaseUrl.TrimEnd('/'))/grader"
    $payload = @{
        task_id = $TaskId
        answer  = $submission
    } | ConvertTo-Json -Depth 8 -Compress

    $response = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType "application/json" -Body $payload

    if ($RawJson) {
        $response | ConvertTo-Json -Depth 20
        exit 0
    }

    Write-Section "Validator Summary"
    [pscustomobject]@{
        task_id = $response.task_id
        score   = [math]::Round([double]$response.score, 6)
    } | Format-List

    Write-Section "Score Breakdown"
    [pscustomobject]@{
        correctness = [math]::Round([double]$response.breakdown.correctness, 6)
        performance = [math]::Round([double]$response.breakdown.performance, 6)
        quality     = [math]::Round([double]$response.breakdown.quality, 6)
    } | Format-Table -AutoSize

    Write-Section "Feedback"
    foreach ($line in ($response.feedback -split "`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line.TrimStart().StartsWith("- ")) {
            Write-Host ("  " + $line.Trim())
        }
        else {
            Write-Host $line
        }
    }
}
catch {
    Write-Host "Validator request failed." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Tips:"
    Write-Host "- Make sure the API is running (`uvicorn server.app:app --host 127.0.0.1 --port 8000`)."
    Write-Host "- Or pass your HF Space URL with -BaseUrl."
    exit 1
}
