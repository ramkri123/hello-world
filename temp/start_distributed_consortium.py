#!/usr/bin/env python3
"""
Start the Complete Distributed Consortium System
Starts hub and all bank nodes with the new anonymization system
"""

import subprocess
import time
import sys
import os
import signal
import requests

def start_consortium_hub():
    """Start the consortium hub"""
    print("🚀 Starting Consortium Hub...")
    hub_process = subprocess.Popen([
        sys.executable, 
        "src/consortium/consortium_hub.py"
    ], cwd=os.getcwd())
    
    # Wait for hub to start
    print("⏳ Waiting for hub to start...")
    for i in range(10):
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code == 200:
                print("✅ Consortium Hub is running!")
                return hub_process
        except:
            time.sleep(1)
    
    print("❌ Failed to start consortium hub")
    return None

def start_bank_nodes():
    """Start all bank participant nodes"""
    print("🏦 Starting Bank Nodes...")
    bank_process = subprocess.Popen([
        sys.executable,
        "src/banks/participant_node.py",
        "--all"
    ], cwd=os.getcwd())
    
    time.sleep(3)  # Give banks time to register
    print("✅ Bank nodes started!")
    return bank_process

def test_system():
    """Test the complete system"""
    print("\n🧪 Testing Complete System...")
    
    try:
        # Test with anonymized inference
        import json
        from src.consortium.account_anonymizer import AccountAnonymizer
        
        anonymizer = AccountAnonymizer()
        
        # Create test transaction
        test_transaction = {
            "email_content": "CEO urgent wire transfer $485,000 confidential acquisition deadline",
            "transaction_data": {
                "amount": 485000,
                "sender_account": "ACCA12345",
                "receiver_account": "ACCB67890",
                "transaction_type": "wire_transfer"
            },
            "use_case": "fraud_detection"
        }
        
        print(f"📧 Test email: {test_transaction['email_content'][:50]}...")
        print(f"💰 Amount: ${test_transaction['transaction_data']['amount']:,}")
        print(f"🔄 Accounts: {test_transaction['transaction_data']['sender_account']} → {test_transaction['transaction_data']['receiver_account']}")
        
        # Submit to consortium
        response = requests.post(
            "http://localhost:8080/inference",
            json=test_transaction,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ System Test PASSED!")
            print(f"📊 Consensus Score: {result.get('consensus_score', 'N/A')}")
            print(f"🏦 Participants: {len(result.get('individual_scores', {}))}")
            
            # Show anonymization
            if 'anonymized_accounts' in result:
                anon = result['anonymized_accounts']
                print(f"🔐 Anonymized: {anon.get('sender_anonymous', 'N/A')} → {anon.get('receiver_anonymous', 'N/A')}")
            
            return True
        else:
            print(f"❌ Test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Start the complete distributed consortium system"""
    print("🌟 DISTRIBUTED CONSORTIUM STARTUP")
    print("=" * 50)
    
    processes = []
    
    try:
        # Start consortium hub
        hub_process = start_consortium_hub()
        if hub_process:
            processes.append(hub_process)
        else:
            print("❌ Failed to start hub, exiting...")
            return
        
        # Start bank nodes
        bank_process = start_bank_nodes()
        if bank_process:
            processes.append(bank_process)
        
        # Test the system
        if test_system():
            print("\n🎉 DISTRIBUTED CONSORTIUM IS RUNNING!")
            print("=" * 40)
            print("🌐 Consortium Hub: http://localhost:8080")
            print("🔍 Health Check: http://localhost:8080/health")
            print("🏦 Bank nodes: Connected and polling")
            print("\n💡 You can now:")
            print("   1. Test via HTTP API at http://localhost:8080/inference")
            print("   2. Use the UI (if started separately)")
            print("   3. Submit transactions for fraud analysis")
            print("\n📊 Features:")
            print("   • One-way hash account anonymization")
            print("   • Scenario-aware confidence weighting")
            print("   • Distributed bank participation")
            print("   • Privacy-preserving fraud detection")
            print("\nPress Ctrl+C to stop all services...")
            
            # Keep running
            while True:
                time.sleep(1)
        else:
            print("❌ System test failed!")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down consortium...")
        
    finally:
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        print("✅ All services stopped.")

if __name__ == "__main__":
    main()
