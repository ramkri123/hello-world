#!/usr/bin/env python3
"""
Simple Clean Launcher (No external dependencies)
Flask-only fraud detection system
"""

import subprocess
import time
import platform
import os

def kill_processes():
    """Kill existing processes"""
    print("🧹 Cleaning up processes...")
    
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        try:
            # Kill Python processes
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         capture_output=True, check=False)
            print("  ✅ Killed Python processes")
        except:
            pass
    else:
        try:
            # Kill Python processes related to our system
            subprocess.run(['pkill', '-f', 'python.*consortium'], check=False)
            subprocess.run(['pkill', '-f', 'python.*fraud'], check=False)
            print("  ✅ Killed Python processes")
        except:
            pass

def start_system():
    """Start the fraud detection system"""
    print("🛡️  Simple Clean Fraud Detection Launcher")
    print("=" * 50)
    
    # Clean up first
    kill_processes()
    time.sleep(1)
    
    # Start consortium hub
    print("🚀 Starting Consortium Hub...")
    try:
        if platform.system() == "Windows":
            subprocess.Popen(
                "python consortium_comparison_score_prototype.py",
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                "python consortium_comparison_score_prototype.py",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        print("  ✅ Consortium Hub started")
        time.sleep(3)
    except Exception as e:
        print(f"  ❌ Failed to start consortium: {e}")
        return
    
    # Start web UI
    print("🚀 Starting Web UI...")
    try:
        if platform.system() == "Windows":
            subprocess.Popen(
                "python simple_fraud_ui.py",
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                "python simple_fraud_ui.py",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        print("  ✅ Web UI started")
        time.sleep(2)
    except Exception as e:
        print(f"  ❌ Failed to start web UI: {e}")
        return
    
    print("\n🎉 System Started!")
    print("=" * 50)
    print("📍 Web UI: http://localhost:5001")
    print("🔗 Consortium: http://localhost:8080")
    print("📊 Status: Flask services running")
    print("❌ Streamlit: Disabled")
    print("\n💡 Open http://localhost:5001 in your browser!")

if __name__ == "__main__":
    start_system()
