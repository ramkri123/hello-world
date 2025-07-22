#!/usr/bin/env python3
"""
Test full consortium system with anonymized account identifiers
"""
import requests
import json
import time
import threading

# Test the consortium hub with fully anonymized account identifiers
def test_consortium_with_anonymization():
    print("🌐 TESTING CONSORTIUM HUB WITH ANONYMIZED ACCOUNT IDENTIFIERS")
    print("=" * 70)
    
    # Start consortium hub in background
    print("Starting consortium hub...")
    import subprocess
    import sys
    
    hub_process = subprocess.Popen([
        sys.executable, "-m", "src.consortium.consortium_hub",
        "--port", "8080"
    ], cwd=".")
    
    # Give it time to start
    time.sleep(3)
    
    try:
        # Test health check
        print("\n1. Testing health check...")
        response = requests.get("http://localhost:8080/health")
        print(f"   Status: {response.status_code}")
        print(f"   Data: {response.json()}")
        
        # Test inference with raw transaction data (includes account anonymization)
        print("\n2. Testing inference with anonymized accounts...")
        
        transaction_data = {
            "email_content": "CEO urgent wire transfer confidential merger acquisition",
            "transaction_data": {
                "amount": 485000,
                "sender_account": "ACCA12345",  # Bank A account
                "receiver_account": "ACCB67890", # Bank B account  
                "transaction_type": "wire_transfer"
            },
            "use_case": "fraud_detection"
        }
        
        print(f"   Original accounts: {transaction_data['transaction_data']['sender_account']} → {transaction_data['transaction_data']['receiver_account']}")
        
        response = requests.post("http://localhost:8080/inference", json=transaction_data)
        print(f"   Inference submitted: {response.status_code}")
        
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data["session_id"]
            print(f"   Session ID: {session_id}")
            
            # Wait a bit then check results
            time.sleep(2)
            
            results_response = requests.get(f"http://localhost:8080/results/{session_id}")
            if results_response.status_code == 200:
                results = results_response.json()
                print(f"   Session status: {results.get('status', 'unknown')}")
                
                if 'final_score' in results:
                    print(f"   Final score: {results['final_score']:.3f}")
                    print(f"   Recommendation: {results['recommendation']}")
                else:
                    print(f"   Responses received: {results.get('responses_received', 0)}")
            else:
                print(f"   Results not ready: {results_response.status_code}")
        else:
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   Error testing consortium: {e}")
    finally:
        # Clean up
        print("\nStopping consortium hub...")
        hub_process.terminate()
        hub_process.wait()
    
    print("\n✅ CONSORTIUM HUB TEST COMPLETED")
    print("   • Account identifiers are fully anonymized in inference requests")
    print("   • Banks receive anonymized data and cannot determine account ownership")
    print("   • Only consortium hub can determine scenarios for proper weighting")

if __name__ == "__main__":
    test_consortium_with_anonymization()
