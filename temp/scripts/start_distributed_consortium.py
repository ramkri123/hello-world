#!/usr/bin/env python3
"""
Distributed Consortium Startup Script
Launches all components of the distributed system
"""

import subprocess
import time
import sys
import os
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistributedConsortiumLauncher:
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
    
    def start_consortium_hub(self):
        """Start the consortium hub server"""
        logger.info("🚀 Starting Consortium Hub...")
        
        cmd = [sys.executable, "consortium_hub.py", "--port", "8080"]
        process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        self.processes.append(("Consortium Hub", process))
        return process
    
    def start_participant_nodes(self):
        """Start all participant nodes as separate processes"""
        logger.info("🏦 Starting Participant Nodes as separate processes...")
        
        bank_processes = []
        
        # Start Bank A
        logger.info("🏦 Starting Bank A - Wire Transfer Specialist...")
        cmd_a = [sys.executable, "generic_bank_process.py", "--bank-id", "bank_A", "--consortium-url", "http://localhost:8080"]
        process_a = subprocess.Popen(
            cmd_a,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        self.processes.append(("Bank A", process_a))
        bank_processes.append(process_a)
        
        # Start Bank B
        logger.info("🔍 Starting Bank B - Identity Verification Expert...")
        cmd_b = [sys.executable, "generic_bank_process.py", "--bank-id", "bank_B", "--consortium-url", "http://localhost:8080"]
        process_b = subprocess.Popen(
            cmd_b,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        self.processes.append(("Bank B", process_b))
        bank_processes.append(process_b)
        
        # Start Bank C
        logger.info("🌐 Starting Bank C - Network Pattern Analyst...")
        cmd_c = [sys.executable, "generic_bank_process.py", "--bank-id", "bank_C", "--consortium-url", "http://localhost:8080"]
        process_c = subprocess.Popen(
            cmd_c,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        self.processes.append(("Bank C", process_c))
        bank_processes.append(process_c)
        
        return bank_processes
    
    def start_ui(self):
        """Start the Streamlit UI"""
        logger.info("🖥️ Starting Distributed UI...")
        
        cmd = [sys.executable, "-m", "streamlit", "run", "distributed_consortium_ui.py", "--server.port", "8501"]
        process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        self.processes.append(("Streamlit UI", process))
        return process
    
    def wait_for_hub_ready(self, timeout=30):
        """Wait for consortium hub to be ready"""
        import requests
        
        logger.info("⏳ Waiting for consortium hub to be ready...")
        
        for i in range(timeout):
            try:
                response = requests.get("http://localhost:8080/health", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Consortium hub is ready!")
                    return True
            except:
                pass
            time.sleep(1)
        
        logger.error("❌ Consortium hub failed to start within timeout")
        return False
    
    def run_test(self):
        """Run the distributed system test"""
        logger.info("🧪 Running distributed system test...")
        
        cmd = [sys.executable, "test_distributed_consortium.py"]
        result = subprocess.run(
            cmd,
            cwd=self.base_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Distributed system test PASSED!")
            return True
        else:
            logger.error("❌ Distributed system test FAILED!")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
    
    def stop_all(self):
        """Stop all running processes"""
        logger.info("🛑 Stopping all processes...")
        
        for name, process in self.processes:
            try:
                process.terminate()
                logger.info(f"🛑 Stopped {name}")
            except:
                pass
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Force kill if necessary
        for name, process in self.processes:
            try:
                if process.poll() is None:
                    process.kill()
                    logger.info(f"🔴 Force killed {name}")
            except:
                pass
    
    def start_full_system(self):
        """Start the complete distributed system"""
        try:
            # 1. Start consortium hub
            hub_process = self.start_consortium_hub()
            
            # 2. Wait for hub to be ready
            if not self.wait_for_hub_ready():
                self.stop_all()
                return False
            
            # 3. Start participant nodes (separate processes)
            bank_processes = self.start_participant_nodes()
            
            # 4. Wait a bit for nodes to register
            time.sleep(8)  # Give more time for separate processes to start
            
            # 5. Run test
            test_passed = self.run_test()
            
            if test_passed:
                # 6. Start UI
                ui_process = self.start_ui()
                
                logger.info("🎉 Distributed Consortium System Started Successfully!")
                logger.info("🌐 Access the UI at: http://localhost:8501")
                logger.info("🛠️ API Hub available at: http://localhost:8080")
                logger.info("📊 Health check: http://localhost:8080/health")
                logger.info("👥 Participants: http://localhost:8080/participants")
                logger.info("")
                logger.info("Press Ctrl+C to stop all services")
                
                return True
            else:
                logger.error("❌ Test failed, not starting UI")
                self.stop_all()
                return False
                
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")
            self.stop_all()
            return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        try:
            while True:
                time.sleep(5)
                
                # Check if any process has died
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.warning(f"⚠️ Process {name} has stopped")
                
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
            self.stop_all()

def main():
    """Main launcher function"""
    launcher = DistributedConsortiumLauncher()
    
    def signal_handler(sig, frame):
        logger.info("🛑 Received interrupt signal")
        launcher.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if launcher.start_full_system():
            launcher.monitor_processes()
        else:
            logger.error("❌ Failed to start distributed system")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        launcher.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
