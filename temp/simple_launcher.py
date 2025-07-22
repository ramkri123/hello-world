#!/usr/bin/env python3
"""
Simple, Reliable Launcher for Fraud Detection System
No complex cleanup - just start the components
"""

import subprocess
import time
import sys
import os

def start_consortium_hub():
    """Start the consortium hub"""
    print("🏛️  Starting Consortium Hub...")
    
    cmd = [sys.executable, "src/consortium/consortium_hub.py"]
    
    # Start the hub
    process = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    
    print(f"   ✅ Consortium Hub started (PID: {process.pid})")
    print(f"   🌐 Available at: http://localhost:8080")
    return process

def start_banks():
    """Start all three banks"""
    print("\n🏦 Starting Specialist Banks...")
    
    banks = ['A', 'B', 'C']
    processes = []
    
    for bank in banks:
        print(f"   Starting Bank {bank}...")
        
        cmd = [sys.executable, "src/banks/universal_bank_launcher.py", f"bank_{bank}"]
        
        process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        
        processes.append(process)
        print(f"   ✅ Bank {bank} started (PID: {process.pid})")
        time.sleep(2)  # Stagger startup
    
    return processes

def start_web_ui():
    """Start the web UI"""
    print("\n🌐 Starting Web UI...")
    
    cmd = [sys.executable, "simple_fraud_ui.py"]
    
    process = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    
    print(f"   ✅ Web UI started (PID: {process.pid})")
    print(f"   🌐 Available at: http://localhost:5001")
    return process

def main():
    print("🚀 SIMPLE FRAUD DETECTION SYSTEM LAUNCHER")
    print("=" * 50)
    
    try:
        # Start components in order
        hub_process = start_consortium_hub()
        time.sleep(3)  # Let hub start up
        
        bank_processes = start_banks()
        time.sleep(3)  # Let banks register
        
        ui_process = start_web_ui()
        
        print("\n" + "=" * 50)
        print("🎯 SYSTEM STARTUP COMPLETE!")
        print("\n📋 Running Components:")
        print(f"   🏛️  Consortium Hub: http://localhost:8080 (PID: {hub_process.pid})")
        print(f"   🏦 Bank A, B, C: Running in separate consoles")
        print(f"   🌐 Web UI: http://localhost:5001 (PID: {ui_process.pid})")
        
        print("\n🔧 Management:")
        print("   • Each component runs in its own console window")
        print("   • Close console windows to stop individual components")
        print("   • Or use taskkill commands to stop by PID")
        
        print("\n✅ System ready for fraud detection testing!")
        
        # Keep script alive
        print("\nPress Ctrl+C to exit this launcher (components will keep running)")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n👋 Launcher exiting (components still running)")
            
    except Exception as e:
        print(f"\n❌ Error during startup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
