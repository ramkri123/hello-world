#!/usr/bin/env python3
"""
Start All Banks - Unified Launcher
Uses the universal bank launcher to start all banks with maximum code reuse
"""

import subprocess
import sys
import time
import signal
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedConsortiumManager:
    """Manager for the distributed consortium system"""
    
    def __init__(self, consortium_url: str = "http://localhost:8080"):
        self.consortium_url = consortium_url
        self.processes = {}
        self.running = False
        
    def start_consortium_hub(self):
        """Start the consortium hub"""
        logger.info("🚀 Starting Consortium Hub...")
        
        try:
            process = subprocess.Popen([
                sys.executable, "src/consortium/consortium_hub.py", 
                "--port", "8080"
            ])
            self.processes["consortium_hub"] = process
            logger.info(f"✅ Consortium Hub started (PID: {process.pid})")
            
            # Give it time to start
            time.sleep(3)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start consortium hub: {e}")
            return False
    
    def start_bank(self, bank_id: str):
        """Start a specific bank using the universal launcher"""
        logger.info(f"🏦 Starting {bank_id.upper()}...")
        
        try:
            process = subprocess.Popen([
                sys.executable, "src/banks/universal_bank_launcher.py",
                bank_id,
                "--consortium-url", self.consortium_url
            ])
            self.processes[bank_id] = process
            logger.info(f"✅ {bank_id.upper()} started (PID: {process.pid})")
            
            # Give it time to register
            time.sleep(2)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {bank_id}: {e}")
            return False
    
    def start_all_banks(self):
        """Start all banks using the universal launcher"""
        banks = ["bank_A", "bank_B", "bank_C"]
        
        for bank_id in banks:
            if not self.start_bank(bank_id):
                return False
                
        return True
    
    def start_ui(self):
        """Start the distributed UI"""
        logger.info("🖥️ Starting Distributed UI...")
        
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "src/ui/distributed_consortium_ui.py",
                "--server.port", "8501"
            ])
            self.processes["ui"] = process
            logger.info(f"✅ UI started (PID: {process.pid})")
            logger.info("   🌐 Access at: http://localhost:8501")
            
            time.sleep(2)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start UI: {e}")
            return False
    
    def start_full_system(self, include_ui: bool = True):
        """Start the complete distributed system"""
        logger.info("🚀 STARTING DISTRIBUTED CONSORTIUM SYSTEM")
        logger.info("=" * 50)
        
        # Start consortium hub
        if not self.start_consortium_hub():
            logger.error("❌ Failed to start consortium hub")
            return False
        
        # Start all banks
        if not self.start_all_banks():
            logger.error("❌ Failed to start all banks")
            self.stop_all()
            return False
        
        # Start UI if requested
        if include_ui:
            if not self.start_ui():
                logger.warning("⚠️ Failed to start UI, but continuing...")
        
        logger.info("✅ DISTRIBUTED SYSTEM STARTUP COMPLETE")
        logger.info("=" * 50)
        logger.info("🏦 Active Components:")
        for name, process in self.processes.items():
            logger.info(f"   {name}: PID {process.pid}")
        logger.info("")
        logger.info("🔗 System URLs:")
        logger.info("   Consortium API: http://localhost:8080")
        if include_ui:
            logger.info("   Web Interface: http://localhost:8501")
        logger.info("")
        logger.info("Press Ctrl+C to stop all processes")
        
        self.running = True
        return True
    
    def stop_all(self):
        """Stop all processes"""
        logger.info("🛑 STOPPING DISTRIBUTED SYSTEM...")
        
        for name, process in self.processes.items():
            try:
                logger.info(f"   Stopping {name} (PID: {process.pid})")
                process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"   Force killing {name}")
                    process.kill()
                    
            except Exception as e:
                logger.error(f"   Error stopping {name}: {e}")
        
        self.processes.clear()
        self.running = False
        logger.info("✅ All processes stopped")
    
    def wait_for_interrupt(self):
        """Wait for user interrupt"""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
            self.stop_all()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Distributed Consortium Manager')
    parser.add_argument('--no-ui', action='store_true', help='Skip starting the UI')
    parser.add_argument('--consortium-url', default='http://localhost:8080', 
                       help='Consortium hub URL')
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    manager = DistributedConsortiumManager(args.consortium_url)
    
    def signal_handler(sig, frame):
        logger.info("🛑 Received interrupt signal")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the full system
        if manager.start_full_system(include_ui=not args.no_ui):
            # Wait for interrupt
            manager.wait_for_interrupt()
        else:
            logger.error("❌ Failed to start distributed system")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        manager.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
