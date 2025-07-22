# Start all services in separate terminals for debuggability
Write-Host "🌟 STARTING DISTRIBUTED FRAUD DETECTION CONSORTIUM" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Function to start a service in a new terminal
function Start-ServiceInTerminal {
    param(
        [string]$Title,
        [string]$ScriptPath,
        [string]$Description
    )
    
    Write-Host "🚀 Starting $Description..." -ForegroundColor Yellow
    
    # Start in new PowerShell window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {python '$ScriptPath'}" -WindowStyle Normal
    
    Write-Host "✅ $Title started in new terminal" -ForegroundColor Green
}

# Start Consortium Hub first
Start-ServiceInTerminal -Title "Consortium Hub" -ScriptPath "start_consortium_hub.py" -Description "Consortium Hub (Port 8080)"

# Wait for hub to start
Write-Host "⏳ Waiting for Consortium Hub to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Bank Nodes
Start-ServiceInTerminal -Title "Bank Nodes" -ScriptPath "start_bank_nodes.py" -Description "Bank Participant Nodes"

# Wait for banks to register
Write-Host "⏳ Waiting for Banks to register..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

# Start Flask UI
Start-ServiceInTerminal -Title "Flask UI" -ScriptPath "start_flask_ui.py" -Description "Flask Web Interface (Port 5000)"

Write-Host ""
Write-Host "✅ ALL SERVICES STARTED IN SEPARATE TERMINALS!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Service URLs:" -ForegroundColor Cyan
Write-Host "   • Consortium Hub: http://localhost:8080" -ForegroundColor White
Write-Host "   • Flask UI:       http://localhost:5000" -ForegroundColor White
Write-Host ""
Write-Host "🔍 Health Checks:" -ForegroundColor Cyan
Write-Host "   • Hub Health:     http://localhost:8080/health" -ForegroundColor White
Write-Host "   • Participants:   http://localhost:8080/participants" -ForegroundColor White
Write-Host ""
Write-Host "💡 Each service runs in its own terminal for easy debugging." -ForegroundColor Yellow
Write-Host "   Close individual terminal windows to stop specific services." -ForegroundColor Yellow
Write-Host ""

# Test connectivity
Write-Host "🧪 Testing service connectivity..." -ForegroundColor Cyan

Start-Sleep -Seconds 3

try {
    $hubResponse = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -UseBasicParsing
    if ($hubResponse.StatusCode -eq 200) {
        Write-Host "✅ Consortium Hub is responding" -ForegroundColor Green
        
        # Check participants
        $participantsResponse = Invoke-WebRequest -Uri "http://localhost:8080/participants" -TimeoutSec 5 -UseBasicParsing
        $participants = $participantsResponse.Content | ConvertFrom-Json
        $bankCount = $participants.participants.Count
        Write-Host "✅ $bankCount bank(s) registered" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Consortium Hub not yet ready (normal during startup)" -ForegroundColor Yellow
}

try {
    $uiResponse = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 5 -UseBasicParsing
    if ($uiResponse.StatusCode -eq 200) {
        Write-Host "✅ Flask UI is responding" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Flask UI not yet ready (normal during startup)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 DISTRIBUTED CONSORTIUM IS READY!" -ForegroundColor Green
Write-Host "🌐 Open http://localhost:5000 to test fraud scenarios" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit this launcher (services will continue running)"
