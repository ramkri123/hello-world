#!/usr/bin/env python3
"""
Cross-Platform Fraud Detection System Launcher
Kills existing processes and starts everything fresh
"""

import os
import sys
import time
import subprocess
import signal
import platform
import psutil
from threading import Thread

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🛡️  CONSORTIUM FRAUD DETECTION SYSTEM LAUNCHER")
    print("=" * 60)
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍  Python: {sys.version.split()[0]}")
    print("=" * 60)

def kill_python_processes():
    """Kill all Python processes running fraud detection components"""
    print("🔄 Stopping existing fraud detection processes...")
    
    target_scripts = [
        'src/ui/consortium_ui.py',
        'simple_fraud_ui.py', 
        'consortium_comparison_score_prototype.py'
    ]
    
    killed_count = 0
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if it's a Python process running our scripts
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any(script in ' '.join(cmdline) for script in target_scripts):
                        print(f"   🔴 Killing PID {proc.info['pid']}: {' '.join(cmdline)}")
                        proc.kill()
                        killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        print(f"   ⚠️  Error during process cleanup: {e}")
    
    if killed_count > 0:
        print(f"   ✅ Killed {killed_count} existing processes")
        time.sleep(2)  # Give processes time to clean up
    else:
        print("   ✅ No existing processes found")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_modules = ['flask', 'flask_cors', 'requests', 'psutil', 'numpy', 'sklearn']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"   ❌ {module}")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + ' '.join(missing))
        return False
    
    return True

def start_process(script_name, description, port=None, wait_time=3):
    """Start a Python process and return the Popen object"""
    print(f"🚀 Starting {description}...")
    
    try:
        # Use absolute path to ensure script is found
        script_path = os.path.join(os.getcwd(), script_name)
        
        if not os.path.exists(script_path):
            print(f"   ❌ Script not found: {script_path}")
            return None
        
        # Start the process
        if platform.system() == "Windows":
            # On Windows, use CREATE_NEW_PROCESS_GROUP to allow clean termination
            process = subprocess.Popen(
                [sys.executable, script_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            # On Unix-like systems
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
        
        print(f"   ✅ Started PID {process.pid}")
        
        if port:
            print(f"   🌐 Waiting for service on port {port}...")
            # Give the service time to start
            time.sleep(wait_time)
            
            # Check if port is accessible (basic check)
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    print(f"   ✅ Service responding on port {port}")
                else:
                    print(f"   ⚠️  Port {port} not yet accessible")
            except Exception as e:
                print(f"   ⚠️  Could not check port {port}: {e}")
        
        return process
        
    except Exception as e:
        print(f"   ❌ Failed to start {description}: {e}")
        return None

def monitor_process(process, name):
    """Monitor a process and report if it exits unexpectedly"""
    try:
        returncode = process.wait()
        if returncode != 0:
            print(f"\n⚠️  {name} exited with code {returncode}")
            # Print any error output
            stderr = process.stderr.read().decode('utf-8', errors='ignore')
            if stderr:
                print(f"Error output:\n{stderr}")
    except Exception as e:
        print(f"\n⚠️  Error monitoring {name}: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        sys.exit(1)
    
    # Kill existing processes
    kill_python_processes()
    
    print("\n🚀 Starting Fraud Detection System Components...")
    
    processes = []
    
    # Start consortium hub first
    consortium_process = start_process(
        'src/consortium/consortium_hub.py',
        'Consortium Hub (Port 8080)',
        port=8080,
        wait_time=5
    )
    
    if consortium_process:
        processes.append((consortium_process, "Consortium Hub"))
        
        # Start web UI after consortium is ready
        ui_process = start_process(
            'simple_fraud_ui.py',
            'Web UI (Port 5001)',
            port=5001,
            wait_time=3
        )
        
        if ui_process:
            processes.append((ui_process, "Web UI"))
        
    else:
        print("❌ Failed to start consortium hub, aborting...")
        sys.exit(1)
    
    if not processes:
        print("❌ No processes started successfully")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ FRAUD DETECTION SYSTEM READY!")
    print("=" * 60)
    print("🌐 Web Interface: http://localhost:5001")
    print("🔗 Consortium API: http://localhost:8080")
    print("📊 Test Page: http://localhost:5001/test")
    print("=" * 60)
    print("💡 Press Ctrl+C to stop all services")
    print("=" * 60)
    
    # Start monitoring threads
    monitor_threads = []
    for process, name in processes:
        thread = Thread(target=monitor_process, args=(process, name), daemon=True)
        thread.start()
        monitor_threads.append(thread)
    
    try:
        # Keep the launcher running and monitor processes
        while True:
            time.sleep(1)
            
            # Check if any process has died
            dead_processes = []
            for i, (process, name) in enumerate(processes):
                if process.poll() is not None:
                    dead_processes.append((i, name))
            
            if dead_processes:
                print(f"\n⚠️  Detected dead processes: {[name for _, name in dead_processes]}")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested...")
    
    # Cleanup
    print("🔄 Stopping all services...")
    for process, name in processes:
        try:
            if process.poll() is None:  # Process is still running
                print(f"   🔴 Stopping {name}...")
                
                if platform.system() == "Windows":
                    # On Windows, send CTRL_BREAK_EVENT
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    # On Unix, send SIGTERM to process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=3)
                    print(f"   ✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    process.kill()
                    print(f"   🔴 {name} force killed")
        except Exception as e:
            print(f"   ⚠️  Error stopping {name}: {e}")
    
    print("✅ All services stopped")
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()
