#!/usr/bin/env python3
"""
Manual Bank Process Launcher
Start each bank as a separate Python process
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_bank_process(bank_id, bank_name):
    """Start a single bank process using the generic bank process"""
    logger.info(f"🚀 Starting {bank_name}...")
    
    cmd = [sys.executable, "generic_bank_process.py", "--bank-id", bank_id, "--consortium-url", "http://localhost:8080"]
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        logger.info(f"✅ {bank_name} started with PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"❌ Failed to start {bank_name}: {e}")
        return None

def main():
    """Main launcher function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual Bank Process Launcher')
    parser.add_argument('--bank', choices=['A', 'B', 'C', 'all'], default='all', 
                       help='Which bank to start (A, B, C, or all)')
    
    args = parser.parse_args()
    
    bank_configs = {
        'A': ('bank_A', 'Bank A - Wire Transfer Specialist'),
        'B': ('bank_B', 'Bank B - Identity Verification Expert'),
        'C': ('bank_C', 'Bank C - Network Pattern Analyst')
    }
    
    processes = []
    
    if args.bank == 'all':
        logger.info("🏦 Starting all bank processes...")
        for bank_letter, (bank_id, name) in bank_configs.items():
            process = start_bank_process(bank_id, name)
            if process:
                processes.append((name, process))
            time.sleep(2)  # Stagger startup
    else:
        bank_id, name = bank_configs[args.bank]
        process = start_bank_process(bank_id, name)
        if process:
            processes.append((name, process))
    
    if processes:
        logger.info(f"✅ Started {len(processes)} bank process(es)")
        logger.info("📊 Each bank is running in its own Python process")
        logger.info("🔍 Check individual console windows for bank-specific logs")
        logger.info("🛑 Close individual windows or press Ctrl+C here to stop monitoring")
        
        try:
            # Monitor processes
            while True:
                time.sleep(5)
                
                # Check if any process has died
                for name, process in processes:
                    if process.poll() is not None:
                        logger.warning(f"⚠️ {name} process has stopped")
                        
        except KeyboardInterrupt:
            logger.info("🛑 Stopping process monitoring")
    else:
        logger.error("❌ No bank processes started")

if __name__ == "__main__":
    main()
