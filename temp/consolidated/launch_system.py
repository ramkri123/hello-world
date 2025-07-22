#!/usr/bin/env python3
"""
Consortium Fraud Detection System - Main Launcher
Consolidated launcher for the complete CEO fraud detection system
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

class ConsortiumLauncher:
    def __init__(self):
        self.processes = []
        self.base_path = Path(__file__).parent
        
    def start_consortium_hub(self):
        """Start the consortium hub"""
        print("🏛️ Starting Consortium Hub...")
        process = subprocess.Popen([
            sys.executable, 
            str(self.base_path / "src" / "consortium" / "consortium_hub.py")
        ])
        self.processes.append(("Consortium Hub", process))
        time.sleep(3)  # Give hub time to start
        
    def start_bank_nodes(self):
        """Start all bank nodes"""
        banks = ["bank_A_process.py", "bank_B_process.py", "bank_C_process.py"]
        
        for bank_file in banks:
            bank_name = bank_file.replace("_process.py", "").replace("_", " ").title()
            print(f"🏦 Starting {bank_name}...")
            process = subprocess.Popen([
                sys.executable,
                str(self.base_path / "src" / "consortium" / bank_file)
            ])
            self.processes.append((bank_name, process))
            time.sleep(2)  # Stagger bank startups
            
    def start_ui(self):
        """Start the CEO fraud detection UI"""
        print("🎭 Starting CEO Fraud Detection UI...")
        process = subprocess.Popen([
            sys.executable,
            str(self.base_path / "ceo_fraud_ui.py")
        ])
        self.processes.append(("CEO Fraud UI", process))
        
    def wait_for_services(self):
        """Wait for all services to be ready"""
        print("\n⏳ Waiting for all services to initialize...")
        time.sleep(5)
        
        print("\n✅ All services should now be running!")
        print("\n🌐 Access the CEO Fraud Demo at: http://localhost:5000")
        print("🎯 Focus: CEO fraud of different levels vs legitimate CEO")
        print("🏦 Demonstrates: Role of Bank vs Role of Consortium")
        print("\n💡 System Architecture:")
        print("   🏛️ Consortium Hub: Pattern recognition & coordination (port 8080)")
        print("   🏦 Bank A: Retail banking specialist (port 8001)")
        print("   🏦 Bank B: Corporate banking specialist (port 8002)")
        print("   🏦 Bank C: Investment banking specialist (port 8003)")
        print("   🎭 CEO Fraud UI: Interactive demo interface (port 5000)")
        
    def cleanup(self, signum=None, frame=None):
        """Clean shutdown of all processes"""
        print("\n🛑 Shutting down Consortium Fraud Detection System...")
        
        for name, process in self.processes:
            try:
                print(f"   Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"   Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"   Error stopping {name}: {e}")
                
        print("✅ All services stopped.")
        sys.exit(0)
        
    def run(self):
        """Main launcher method"""
        print("🚀 Launching Consortium Fraud Detection System")
        print("=" * 60)
        
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        try:
            # Start services in order
            self.start_consortium_hub()
            self.start_bank_nodes()
            self.start_ui()
            self.wait_for_services()
            
            # Keep running until interrupted
            print("\n📊 System is running. Press Ctrl+C to stop all services.")
            while True:
                time.sleep(1)
                
                # Check if any process has died
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"\n⚠️ {name} has stopped unexpectedly!")
                        
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.cleanup()

if __name__ == "__main__":
    launcher = ConsortiumLauncher()
    launcher.run()
