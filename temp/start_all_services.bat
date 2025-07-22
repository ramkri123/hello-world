@echo off
REM Start Consortium Hub in dedicated terminal
echo 🌟 Starting Consortium Hub in new terminal...
start "Consortium Hub" cmd /k "python start_consortium_hub.py"

REM Wait a moment for hub to start
timeout /t 3 /nobreak >nul

echo 🏦 Starting Bank Nodes in new terminal...
start "Bank Nodes" cmd /k "python start_bank_nodes.py"

REM Wait a moment for banks to register
timeout /t 5 /nobreak >nul

echo 🌐 Starting Flask UI in new terminal...
start "Flask UI" cmd /k "python start_flask_ui.py"

echo.
echo ✅ All services started in separate terminals!
echo.
echo 📊 Service URLs:
echo    • Consortium Hub: http://localhost:8080
echo    • Flask UI:       http://localhost:5000
echo.
echo 🔍 Health Checks:
echo    • Hub Health:     http://localhost:8080/health
echo    • Participants:   http://localhost:8080/participants
echo.
echo 💡 Each service is running in its own terminal for easy debugging.
echo    Close individual terminal windows to stop specific services.
echo.
pause
