#!/usr/bin/env python3
"""
Start Bank Nodes in dedicated terminal
"""
import sys
import os
import subprocess
import time
import requests
import threading

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.banks.participant_node import ParticipantNode, NodeConfig, create_bank_nodes

def wait_for_hub():
    """Wait for consortium hub to be available"""
    print("⏳ Waiting for consortium hub to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code == 200:
                print("✅ Consortium hub is ready!")
                return True
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}/30)")
    
    print("❌ Consortium hub not available")
    return False

def start_bank(bank_id, specialty, location="US-East"):
    """Start a single bank node"""
    try:
        print(f"🏦 Starting {bank_id}...")
        
        # Create all bank nodes and find the one we want
        nodes = create_bank_nodes("http://localhost:8080")
        node = next((n for n in nodes if n.config.node_id == bank_id), None)
        
        if not node:
            print(f"❌ Bank {bank_id} not found in configuration")
            return
        
        # Register and start polling
        if node.register_with_consortium():
            node.start_polling()
            print(f"✅ {bank_id} registered and polling")
            
            # Keep the node running
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                print(f"🛑 Stopping {bank_id}...")
                node.stop()
        else:
            print(f"❌ Failed to register {bank_id}")
        
    except Exception as e:
        print(f"❌ Error starting {bank_id}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🏦 BANK NODES - DEDICATED TERMINAL")
    print("=" * 50)
    print("🚀 Starting Bank Participant Nodes...")
    print("📍 Banks: A (Wire Transfer), B (Identity), C (Network)")
    print("🌐 Consortium URL: http://localhost:8080")
    print("-" * 50)
    
    # Wait for consortium hub
    if not wait_for_hub():
        print("❌ Cannot start banks without consortium hub")
        return
    
    print("🏦 Starting Bank Nodes...")
    
    # Bank configurations
    banks = [
        ("bank_A", "wire_transfer_specialist"),
        ("bank_B", "identity_verification_expert"), 
        ("bank_C", "network_pattern_analyst")
    ]
    
    # Start each bank in a separate thread
    threads = []
    for bank_id, specialty in banks:
        thread = threading.Thread(
            target=start_bank,
            args=(bank_id, specialty),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        time.sleep(1)  # Stagger startup
    
    print("✅ All bank nodes started!")
    print("🔄 Banks are now polling for inference requests...")
    print("💡 Press Ctrl+C to stop all banks")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(10)
            # Check if banks are still responding
            try:
                response = requests.get("http://localhost:8080/participants", timeout=5)
                if response.status_code == 200:
                    participants = response.json().get('participants', [])
                    active_banks = len([p for p in participants if p['status'] == 'active'])
                    print(f"📊 Status check: {active_banks}/3 banks active", end='\r')
            except:
                pass
                
    except KeyboardInterrupt:
        print("\n🛑 Bank nodes stopped by user")

if __name__ == "__main__":
    main()
