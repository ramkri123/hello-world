#!/usr/bin/env python3
"""
Quick Launcher - Organized Project Structure
Simple commands for the reorganized consortium fraud detection system
"""

import subprocess
import sys
import os
import time

def show_help():
    """Show available commands"""
    print("🚀 PRIVACY-PRESERVING CONSORTIUM - QUICK LAUNCHER")
    print("=" * 55)
    print()
    print("📋 Available Commands:")
    print()
    print("🌐 SYSTEM MANAGEMENT:")
    print("  start-all         Start complete distributed system")
    print("  start-all-windowed Start system in separate windows (recommended)")
    print("  start-hub         Start consortium hub only")
    print("  start-banks       Start all banks")
    print("  start-ui          Start live dashboard (no simulator needed)")
    print("  start-ui-demo     Start demo UI with simulator")
    print()
    print("🏦 INDIVIDUAL BANKS:")
    print("  start-bank-a      Start Bank A (Wire Transfer Specialist)")
    print("  start-bank-b      Start Bank B (Identity Verification)")
    print("  start-bank-c      Start Bank C (Network Pattern Analyst)")
    print("  list-banks        List available bank configurations")
    print()
    print("🧪 TESTING:")
    print("  test-privacy      Test privacy-preserving system")
    print("  test-distributed  Test distributed system")
    print("  test-client       Run simple client test")
    print()
    print("📚 UTILITIES:")
    print("  help              Show this help")
    print("  structure         Show project structure")
    print()
    print("💡 Examples:")
    print("  python launcher.py start-all-windowed  # Recommended for troubleshooting")
    print("  python launcher.py start-all")
    print("  python launcher.py start-bank-a")
    print("  python launcher.py test-privacy")

def run_command(script_path, *args):
    """Run a script with arguments"""
    try:
        cmd = [sys.executable, script_path] + list(args)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ File not found: {script_path}")
        sys.exit(1)

def run_in_new_window(script_path, window_title, *args):
    """Run a script in a new Windows Command Prompt window"""
    try:
        import uuid
        
        # Create a unique batch file name in current directory
        safe_title = window_title.replace(" ", "_").replace("-", "_")
        bat_filename = f"start_{safe_title}_{uuid.uuid4().hex[:8]}.bat"
        bat_path = os.path.join(os.getcwd(), bat_filename)
        
        # Create the batch file
        with open(bat_path, 'w') as bat_file:
            # Write batch commands to the file
            bat_file.write(f'@echo off\n')
            bat_file.write(f'title {window_title}\n')
            bat_file.write(f'cd /d "{os.getcwd()}"\n')
            bat_file.write(f'echo Starting {window_title}...\n')
            bat_file.write(f'echo Working directory: %cd%\n')
            bat_file.write(f'echo.\n')
            
            # Build the command
            cmd_parts = [f'"{sys.executable}"', f'"{script_path}"'] + [f'"{arg}"' for arg in args]
            bat_file.write(' '.join(cmd_parts) + '\n')
            
            bat_file.write(f'echo.\n')
            bat_file.write(f'echo Process finished. Press any key to close window...\n')
            bat_file.write(f'pause >nul\n')
            bat_file.write(f'del "%~f0"\n')  # Delete the batch file when done
        
        # Start the batch file in a new window
        subprocess.Popen(['cmd', '/c', 'start', bat_filename])
        print(f"✅ Started {window_title} in new window")
        
    except Exception as e:
        print(f"❌ Failed to start {window_title}: {e}")

def start_system_windowed():
    """Start the complete system in separate windows for easy troubleshooting"""
    print("🚀 STARTING DISTRIBUTED CONSORTIUM SYSTEM IN SEPARATE WINDOWS")
    print("=" * 65)
    print()
    
    # Start consortium hub
    print("🏢 Starting Consortium Hub...")
    run_in_new_window("src/consortium/consortium_hub.py", "Consortium Hub", "--port", "8080")
    
    # Wait a moment for hub to start
    print("   Waiting for consortium hub to initialize...")
    time.sleep(4)
    
    # Start all banks
    print("🏦 Starting Banks...")
    run_in_new_window("src/banks/universal_bank_launcher.py", "Bank A - Wire Transfer", "bank_A")
    time.sleep(2)
    run_in_new_window("src/banks/universal_bank_launcher.py", "Bank B - Identity Verification", "bank_B")
    time.sleep(2)
    run_in_new_window("src/banks/universal_bank_launcher.py", "Bank C - Network Analysis", "bank_C")
    time.sleep(2)
    
    # Start UI
    print("🖥️ Starting Web Interface...")
    run_in_new_window_streamlit("src/ui/live_consortium_ui.py", "Live Consortium Dashboard", "8504")
    
    print()
    print("✅ SYSTEM STARTUP COMPLETE")
    print("=" * 40)
    print("🔗 Access Points:")
    print("   Consortium API: http://localhost:8080")
    print("   Live Dashboard: http://localhost:8504")
    print()
    print("💡 Each component is running in its own window for easy monitoring")
    print("   Close individual windows to stop specific components")

def run_in_new_window_streamlit(script_path, window_title, port="8504"):
    """Run a Streamlit app in a new Windows Command Prompt window"""
    try:
        import uuid
        
        # Create a unique batch file name in current directory
        safe_title = window_title.replace(" ", "_").replace("-", "_")
        bat_filename = f"start_streamlit_{safe_title}_{uuid.uuid4().hex[:8]}.bat"
        bat_path = os.path.join(os.getcwd(), bat_filename)
        
        # Create the batch file for Streamlit
        with open(bat_path, 'w') as bat_file:
            # Write batch commands to the file
            bat_file.write(f'@echo off\n')
            bat_file.write(f'title {window_title}\n')
            bat_file.write(f'cd /d "{os.getcwd()}"\n')
            bat_file.write(f'echo Starting {window_title} on port {port}...\n')
            bat_file.write(f'echo Working directory: %cd%\n')
            bat_file.write(f'echo.\n')
            
            # Build the streamlit command
            bat_file.write(f'"{sys.executable}" -m streamlit run "{script_path}" --server.port {port} --server.address localhost\n')
            
            bat_file.write(f'echo.\n')
            bat_file.write(f'echo Streamlit finished. Press any key to close window...\n')
            bat_file.write(f'pause >nul\n')
            bat_file.write(f'del "%~f0"\n')  # Delete the batch file when done
        
        # Start the batch file in a new window
        subprocess.Popen(['cmd', '/c', 'start', bat_filename])
        print(f"✅ Started {window_title} in new window (port {port})")
        
    except Exception as e:
        print(f"❌ Failed to start {window_title}: {e}")

def main():
    """Main launcher"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    # System management commands
    if command == "start-all":
        run_command("scripts/start_unified_consortium.py")
    elif command == "start-all-windowed":
        start_system_windowed()
    elif command == "start-hub":
        run_command("src/consortium/consortium_hub.py")
    elif command == "start-banks":
        run_command("scripts/start_banks_separately.py")
    elif command == "start-ui":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/live_consortium_ui.py", "--server.port", "8504", "--server.address", "localhost"])
    elif command == "start-ui-demo":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/consortium_fraud_ui.py", "--server.port", "8505", "--server.address", "localhost"])
    
    # Individual bank commands
    elif command == "start-bank-a":
        run_command("src/banks/universal_bank_launcher.py", "bank_A")
    elif command == "start-bank-b":
        run_command("src/banks/universal_bank_launcher.py", "bank_B")
    elif command == "start-bank-c":
        run_command("src/banks/universal_bank_launcher.py", "bank_C")
    elif command == "list-banks":
        run_command("src/banks/universal_bank_launcher.py", "--list-banks")
    
    # Testing commands
    elif command == "test-privacy":
        run_command("tests/test_privacy_consortium.py")
    elif command == "test-distributed":
        run_command("tests/test_distributed_consortium.py")
    elif command == "test-client":
        run_command("src/consortium/consortium_client.py")
    
    # Utility commands
    elif command == "help":
        show_help()
    elif command == "structure":
        with open("docs/PROJECT_STRUCTURE.md", "r") as f:
            print(f.read())
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python launcher.py help' for available commands")
        sys.exit(1)

if __name__ == "__main__":
    main()
