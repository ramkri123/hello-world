#!/usr/bin/env python3
"""
Clean Fraud Detection System Launcher
NO STREAMLIT - Only Flask-based components
Supports both Windows and Linux with proper process management
"""

import os
import sys
import time
import signal
import subprocess
import platform
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not available - using basic process management")

class CleanSystemLauncher:
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.processes = []
        
    def kill_existing_processes(self):
        """Kill any existing Python processes that might interfere"""
        print("🧹 Cleaning up existing processes...")
        
        if HAS_PSUTIL:
            # Advanced cleanup with psutil
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        # Kill processes related to our fraud detection system
                        if any(keyword in cmdline.lower() for keyword in [
                            'consortium', 'fraud', 'simple_fraud_ui', 
                            'bank_simulation', 'launcher'
                        ]):
                            print(f"  🔥 Killing process: {proc.info['name']} (PID: {proc.info['pid']})")
                            proc.terminate()
                            killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Also kill any Streamlit processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'streamlit' in proc.info['name'].lower():
                        print(f"  🔥 Killing Streamlit process: {proc.info['name']} (PID: {proc.info['pid']})")
                        proc.terminate()
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if killed_count > 0:
                print(f"  ✅ Terminated {killed_count} processes")
                time.sleep(2)  # Wait for processes to terminate
        else:
            # Basic cleanup without psutil
            if self.is_windows:
                try:
                    subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                                 capture_output=True, check=False)
                    subprocess.run(['taskkill', '/f', '/im', 'streamlit.exe'], 
                                 capture_output=True, check=False)
                    print("  ✅ Killed Python and Streamlit processes")
                except:
                    pass
            else:
                try:
                    subprocess.run(['pkill', '-f', 'python.*consortium'], check=False)
                    subprocess.run(['pkill', '-f', 'python.*fraud'], check=False)
                    subprocess.run(['pkill', '-f', 'streamlit'], check=False)
                    print("  ✅ Killed Python and Streamlit processes")
                except:
                    pass
    
    def start_process(self, command, description, wait_time=2):
        """Start a process and track it"""
        print(f"🚀 Starting {description}...")
        try:
            if self.is_windows:
                # Windows: Start process in new window to keep it running
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Linux: Start process in background
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            self.processes.append(proc)
            print(f"  ✅ {description} started (PID: {proc.pid})")
            time.sleep(wait_time)
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to start {description}: {e}")
            return False
    
    def launch_system(self):
        """Launch the complete fraud detection system"""
        print("🛡️  Clean Fraud Detection System Launcher")
        print("=" * 50)
        print("📌 Flask-only components (NO Streamlit)")
        
        # Step 1: Clean up
        self.kill_existing_processes()
        
        # Step 2: Start Consortium Hub (Production version)
        if not self.start_process(
            "python src/consortium/consortium_hub.py",
            "Production Consortium Hub (Port 8080)",
            wait_time=3
        ):
            return False
        
        # Step 3: Start Bank Processes
        if not self.start_process(
            "python src/banks/bank_A_process.py",
            "Bank A - Wire Transfer Specialist",
            wait_time=2
        ):
            return False
            
        if not self.start_process(
            "python src/banks/bank_B_process.py", 
            "Bank B - Identity Verification",
            wait_time=2
        ):
            return False
            
        if not self.start_process(
            "python src/banks/bank_C_process.py",
            "Bank C - Network Analysis", 
            wait_time=2
        ):
            return False
        
        # Step 4: Start Web UI
        if not self.start_process(
            "python simple_fraud_ui.py",
            "Web UI (Port 5001)",
            wait_time=2
        ):
            return False
        
        print("\n🎉 Production System Launch Complete!")
        print("=" * 50)
        print("📍 Web UI: http://localhost:5001")
        print("🔗 Consortium Hub: http://localhost:8080")
        print("🏦 Bank A: Wire Transfer Specialist")
        print("🏦 Bank B: Identity Verification Expert") 
        print("🏦 Bank C: Network Pattern Analyst")
        print("📊 Status: Distributed production system running")
        print("❌ Streamlit: Disabled (clean Flask-only system)")
        print("\n💡 Open your browser to http://localhost:5001 to start testing!")
        
        return True
    
    def cleanup(self):
        """Clean shutdown of all processes"""
        print("\n🛑 Shutting down system...")
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
        print("✅ System shutdown complete")

def main():
    launcher = CleanSystemLauncher()
    
    try:
        success = launcher.launch_system()
        if success:
            print("\n⌨️  Press Ctrl+C to shutdown the system")
            while True:
                time.sleep(1)
        else:
            print("❌ System launch failed")
            
    except KeyboardInterrupt:
        launcher.cleanup()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        launcher.cleanup()

if __name__ == "__main__":
    main()
