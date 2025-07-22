#!/usr/bin/env python3
"""
Start Bank A Participant Node
"""

import sys
import os
sys.path.append('src')

from banks.participant_node import ParticipantNode, NodeConfig

def main():
    # Bank A configuration
    config = NodeConfig(
        node_id="bank_A",
        specialty="fraud_detection",
        consortium_url="http://localhost:8080",
        model_path="models/bank_A_model.pkl",
        bank_accounts=[
            "ACCA12345", "ACCA67890", "ACCA11111", 
            "ACCA99999", "ACCA55555"
        ],
        geolocation="US-East"
    )
    
    # Create and start node
    node = ParticipantNode(config)
    
    print("🏦 STARTING BANK A PARTICIPANT NODE")
    print("==================================")
    print(f"📍 Node ID: {config.node_id}")
    print(f"🎯 Specialty: {config.specialty}")
    print(f"🌐 Consortium Hub: {config.consortium_url}")
    print(f"🤖 Model Path: {config.model_path}")
    print(f"📋 Accounts: {len(config.bank_accounts)} accounts")
    print("🔐 One-way hash anonymization enabled")
    print("🎭 Scenario-aware confidence weighting")
    
    try:
        # Register with consortium
        if node.register():
            print("✅ Successfully registered with consortium hub")
            # Start heartbeat and wait for inference requests
            node.start_heartbeat()
            print("💓 Heartbeat started - node is ready for inference requests")
            print("🔄 Press Ctrl+C to stop the node")
            
            # Keep running
            while True:
                time.sleep(1)
        else:
            print("❌ Failed to register with consortium hub")
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down Bank A node...")
        node.is_running = False
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    import time
    main()
