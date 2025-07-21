#!/usr/bin/env python3
"""
Simple Cross-Platform Fraud Detection System Launcher
Works without external dependencies
"""

import os
import sys
import time
import subprocess
import signal
import platform

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🛡️  CONSORTIUM FRAUD DETECTION SYSTEM LAUNCHER")
    print("=" * 60)
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍  Python: {sys.version.split()[0]}")
    print("=" * 60)

def kill_existing_processes():
    """Kill existing Python processes using platform-specific commands"""
    print("🔄 Stopping existing fraud detection processes...")
    
    try:
        if platform.system() == "Windows":
            # Windows: Kill python processes running our scripts
            subprocess.run([
                "taskkill", "/f", "/im", "python.exe"
            ], capture_output=True, text=True)
            subprocess.run([
                "taskkill", "/f", "/im", "pythonw.exe"
            ], capture_output=True, text=True)
        else:
            # Linux/Unix: Kill python processes
            subprocess.run([
                "pkill", "-f", "src/ui/consortium_ui.py"
            ], capture_output=True)
            subprocess.run([
                "pkill", "-f", "simple_fraud_ui.py"
            ], capture_output=True)
        
        print("   ✅ Killed existing processes")
        time.sleep(2)
        
    except Exception as e:
        print(f"   ⚠️  Process cleanup error (continuing anyway): {e}")

def get_python_executable():
    """Get the correct Python executable path"""
    # Check if we're in a virtual environment
    venv_python = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe')
    if os.path.exists(venv_python):
        return venv_python
    
    # Check Linux/Mac virtual environment
    venv_python_unix = os.path.join(os.getcwd(), '.venv', 'bin', 'python')
    if os.path.exists(venv_python_unix):
        return venv_python_unix
    
    # Fall back to system Python
    return sys.executable

def start_service(script_name, description, port=None):
    """Start a service and return the process"""
    print(f"🚀 Starting {description}...")
    
    try:
        script_path = os.path.join(os.getcwd(), script_name)
        
        if not os.path.exists(script_path):
            print(f"   ❌ Script not found: {script_path}")
            return None
        
        python_exe = get_python_executable()
        print(f"   🐍 Using Python: {python_exe}")
        
        # Start the process in background
        if platform.system() == "Windows":
            process = subprocess.Popen(
                [python_exe, script_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        
        print(f"   ✅ Started PID {process.pid}")
        
        if port:
            print(f"   ⏳ Waiting for service on port {port}...")
            time.sleep(3)  # Give service time to start
            
            # Simple port check
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    print(f"   ✅ Service responding on port {port}")
                else:
                    print(f"   ⚠️  Port {port} not yet accessible (may need more time)")
            except Exception as e:
                print(f"   ⚠️  Could not check port {port}: {e}")
        
        return process
        
    except Exception as e:
        print(f"   ❌ Failed to start {description}: {e}")
        return None

def wait_for_exit():
    """Wait for user to press Ctrl+C"""
    try:
        print("\n💡 Press Ctrl+C to stop all services")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested...")

def cleanup_processes(processes):
    """Clean up all processes"""
    print("🔄 Stopping all services...")
    
    for process, name in processes:
        try:
            if process and process.poll() is None:
                print(f"   🔴 Stopping {name}...")
                
                if platform.system() == "Windows":
                    subprocess.run([
                        "taskkill", "/f", "/pid", str(process.pid)
                    ], capture_output=True)
                else:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except:
                        process.terminate()
                
                print(f"   ✅ {name} stopped")
        except Exception as e:
            print(f"   ⚠️  Error stopping {name}: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Kill existing processes
    kill_existing_processes()
    
    print("\n🚀 Starting Fraud Detection System...")
    
    processes = []
    
    # Start consortium hub first
    print("\n1️⃣ Starting Consortium Hub...")
    consortium_process = start_service(
        'src/ui/consortium_ui.py',
        'Consortium Hub',
        port=8080
    )
    
    if consortium_process:
        processes.append((consortium_process, "Consortium Hub"))
        
        # Wait a bit more for consortium to fully initialize
        print("   ⏳ Waiting for consortium to initialize...")
        time.sleep(5)
        
        # Start web UI
        print("\n2️⃣ Starting Web UI...")
        ui_process = start_service(
            'simple_fraud_ui.py',
            'Web UI',
            port=5001
        )
        
        if ui_process:
            processes.append((ui_process, "Web UI"))
    
    if not processes:
        print("❌ No services started successfully")
        return
    
    # Success message
    print("\n" + "=" * 60)
    print("✅ FRAUD DETECTION SYSTEM READY!")
    print("=" * 60)
    print("🌐 Web Interface: http://localhost:5001")
    print("🔗 Consortium API: http://localhost:8080")
    print("📊 Test Page: http://localhost:5001/test")
    print("=" * 60)
    
    # Wait for exit signal
    wait_for_exit()
    
    # Cleanup
    cleanup_processes(processes)
    
    print("✅ All services stopped")
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()
