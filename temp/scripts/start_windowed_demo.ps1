# PowerShell script to start all consortium components in organized windows

Write-Host "🚀 Starting Consortium Fraud Detection Demo" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Function to start a process in a new PowerShell window with title
function Start-ProcessWindow {
    param(
        [string]$Title,
        [string]$Command,
        [string]$Color = "White"
    )
    
    $scriptBlock = @"
`$Host.UI.RawUI.WindowTitle = '$Title'
Write-Host '$Title' -ForegroundColor $Color
Write-Host ('=' * $Title.Length) -ForegroundColor $Color
cd 'c:\Users\ramkr\hello-world\temp'
$Command
"@
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $scriptBlock
    Start-Sleep -Seconds 2
}

# Kill any existing processes
Write-Host "🧹 Cleaning up existing processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force -ErrorAction SilentlyContinue

# Wait for cleanup
Start-Sleep -Seconds 3

# Start Consortium Hub
Write-Host "🎯 Starting Consortium Hub..." -ForegroundColor Green
Start-ProcessWindow -Title "CONSORTIUM HUB (FRAUD DETECTION)" -Command "python consortium_hub.py" -Color "Green"

# Wait for hub to start
Start-Sleep -Seconds 5

# Start Banks
Write-Host "🏦 Starting Bank A (Wire Transfer Specialist)..." -ForegroundColor Blue
Start-ProcessWindow -Title "BANK A - Wire Transfer Specialist" -Command "python generic_bank_process.py --bank-id bank_A" -Color "Blue"

Write-Host "🏦 Starting Bank B (Identity Expert)..." -ForegroundColor Yellow
Start-ProcessWindow -Title "BANK B - Identity Expert" -Command "python generic_bank_process.py --bank-id bank_B" -Color "Yellow"

Write-Host "🏦 Starting Bank C (Network Analyst)..." -ForegroundColor Cyan
Start-ProcessWindow -Title "BANK C - Network Analyst" -Command "python generic_bank_process.py --bank-id bank_C" -Color "Cyan"

# Wait for banks to register
Start-Sleep -Seconds 8

# Check health
Write-Host "🔍 Checking system health..." -ForegroundColor Magenta
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing | ConvertFrom-Json
    Write-Host "✅ Consortium Hub: $($health.participants) participants registered" -ForegroundColor Green
} catch {
    Write-Host "❌ Consortium Hub not responding" -ForegroundColor Red
}

# Start UI
Write-Host "🖥️ Starting Streamlit UI..." -ForegroundColor Magenta
Start-ProcessWindow -Title "STREAMLIT UI - Demo Interface" -Command "streamlit run distributed_consortium_ui.py --server.port 8502 --server.address localhost" -Color "Magenta"

# Wait for UI
Start-Sleep -Seconds 10

# Test fraud detection
Write-Host "🧪 Testing BEC Fraud Detection..." -ForegroundColor Red
$testBody = @{
    "features" = @(0.35, 0.45, 0.75, 0.40, 0.85, 0.35, 0.40, 0.70, 0.80, 0.90, 0.25, 0.35, 0.15, 0.30, 0.10, 0.70, 0.85, 0.90, 0.40, 0.35, 0.75, 0.35, 0.65, 0.55, 0.85, 0.75, 0.70, 0.75, 0.45, 0.40)
    "use_case" = "fraud_detection"
} | ConvertTo-Json -Depth 3

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/inference" -Method POST -Body $testBody -ContentType "application/json" -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    Write-Host "📊 Test session created: $($result.session_id)" -ForegroundColor Green
    
    Start-Sleep -Seconds 6
    
    $testResults = Invoke-WebRequest -Uri "http://localhost:8080/results/$($result.session_id)" -UseBasicParsing | ConvertFrom-Json
    Write-Host "🎯 Test Results:" -ForegroundColor Cyan
    Write-Host "   Bank A: $($testResults.individual_scores.bank_A.ToString('F3'))" -ForegroundColor Blue
    Write-Host "   Bank B: $($testResults.individual_scores.bank_B.ToString('F3'))" -ForegroundColor Yellow  
    Write-Host "   Bank C: $($testResults.individual_scores.bank_C.ToString('F3'))" -ForegroundColor Cyan
    Write-Host "   Consensus: $($testResults.consensus_score.ToString('F3'))" -ForegroundColor White
    Write-Host "   Recommendation: $($testResults.recommendation)" -ForegroundColor $(if($testResults.recommendation -eq "approve") {"Red"} else {"Green"})
} catch {
    Write-Host "❌ Fraud detection test failed" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 DEMO READY!" -ForegroundColor Green
Write-Host "===============" -ForegroundColor Green
Write-Host "📱 Web UI: http://localhost:8502" -ForegroundColor Cyan
Write-Host "🎯 Demo: Select 'CEO Email Fraud' in the UI" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 You now have 5 organized windows:" -ForegroundColor White
Write-Host "   🟢 Consortium Hub (Green)" -ForegroundColor Green
Write-Host "   🔵 Bank A - Wire Specialist (Blue)" -ForegroundColor Blue  
Write-Host "   🟡 Bank B - Identity Expert (Yellow)" -ForegroundColor Yellow
Write-Host "   🔵 Bank C - Network Analyst (Cyan)" -ForegroundColor Cyan
Write-Host "   🟣 Streamlit UI (Magenta)" -ForegroundColor Magenta

# Open browser
Start-Process "http://localhost:8502"

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
