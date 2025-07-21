#!/usr/bin/env python3
"""
Register banks in the consortium for testing
"""

import requests
import json
import time

def register_bank(bank_id, specialty, endpoint="http://localhost:5000"):
    """Register a bank with the consortium"""
    data = {
        "node_id": bank_id,
        "specialty": specialty,
        "endpoint": endpoint,
        "geolocation": "US-East"
    }
    
    try:
        response = requests.post("http://localhost:8080/register", json=data)
        if response.status_code == 200:
            print(f"✅ Registered {bank_id}: {specialty}")
            return True
        else:
            print(f"❌ Failed to register {bank_id}: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error registering {bank_id}: {e}")
        return False

def main():
    print("🏦 REGISTERING BANKS WITH CONSORTIUM")
    print("="*50)
    
    # Wait for consortium to be ready
    print("⏳ Waiting for consortium server...")
    time.sleep(3)
    
    # Register our three specialized banks
    banks = [
        ("bank_A", "sender_transaction_specialist"),
        ("bank_B", "identity_receiver_specialist"), 
        ("bank_C", "network_account_specialist")
    ]
    
    registered = 0
    for bank_id, specialty in banks:
        if register_bank(bank_id, specialty):
            registered += 1
        time.sleep(1)  # Small delay between registrations
    
    print(f"\n🎯 Registration Summary: {registered}/{len(banks)} banks registered")
    
    # Check consortium status
    try:
        response = requests.get("http://localhost:8080/status")
        if response.status_code == 200:
            status = response.json()
            print(f"🌐 Consortium Status:")
            print(f"   👥 Participants: {status['participants']}")
            print(f"   📊 Active Sessions: {status['active_sessions']}")
            print(f"   🔧 Ready for inference: {'✅ YES' if status['participants'] >= 2 else '❌ NO'}")
        else:
            print(f"❌ Failed to get consortium status: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error checking status: {e}")

if __name__ == "__main__":
    main()
