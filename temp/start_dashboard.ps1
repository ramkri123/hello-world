# Consortium Fraud Detection UI Launcher (Windows PowerShell)
# Starts the Streamlit app with localhost binding

Write-Host "Starting Consortium Fraud Detection Dashboard..." -ForegroundColor Green
Write-Host "Network: Localhost (127.0.0.1:8501)" -ForegroundColor Cyan
Write-Host "Security: Local access only" -ForegroundColor Yellow
Write-Host ""

# Define paths for agent context files
$ProjectContextPath = "agent_context.json"
$DocumentationPath = "fraud_scenarios_explained.md"
$ExternalConfigPath = "C:\path\to\external\config.json"
$ExternalDataPath = "C:\path\to\external\data"
$ExternalDocsPath = "C:\path\to\external\documentation"

# Load project context for AI agents
if (Test-Path $ProjectContextPath) {
    Write-Host "Loading project context from: $ProjectContextPath" -ForegroundColor Cyan
    $ProjectContext = Get-Content $ProjectContextPath | ConvertFrom-Json
    Write-Host "Project: $($ProjectContext.project.name)" -ForegroundColor Green
}

# Load external configuration if available
if (Test-Path $ExternalConfigPath) {
    Write-Host "Loading external configuration from: $ExternalConfigPath" -ForegroundColor Cyan
    $ExternalConfig = Get-Content $ExternalConfigPath | ConvertFrom-Json
    # Use $ExternalConfig.property_name to access values
}

# Activate virtual environment (optional - we'll use full paths)
# & "C:/Users/ramkr/hello-world/temp/.venv/Scripts/Activate.ps1"

# Check if models exist
if (!(Test-Path "models")) {
    Write-Host "No trained models found. Running training first..." -ForegroundColor Yellow
    & "C:/Users/ramkr/hello-world/temp/.venv/Scripts/python.exe" consortium_comparison_score_prototype.py train
    Write-Host "Training complete!" -ForegroundColor Green
    Write-Host ""
}

# Start Streamlit with localhost binding
Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
& "C:/Users/ramkr/hello-world/temp/.venv/Scripts/streamlit.exe" run consortium_fraud_ui.py --server.address localhost --server.port 8501

Write-Host "Dashboard stopped." -ForegroundColor Red
