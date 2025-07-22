import requests
import json
import time

# Test the distributed consortium API
def test_consortium_api():
    print("🧪 Testing Consortium API...")
    
    # Test data - CEO fraud scenario
    test_data = {
        "email_content": "URGENT: CEO here. Need immediate wire transfer of $485,000 to close confidential acquisition. Please transfer from our main account ACCA12345 to vendor account ACCB67890. Time-sensitive deal - board approval already secured. Handle discreetly.",
        "transaction_data": {
            "amount": 485000,
            "sender_account": "ACCA12345",
            "receiver_account": "ACCB67890",
            "transaction_type": "wire_transfer"
        },
        "sender_data": {"bank": "bank_A"},
        "receiver_data": {"bank": "bank_B"}
    }
    
    try:
        # Submit inference request
        response = requests.post(
            "http://localhost:8080/inference",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            session_id = result['session_id']
            print(f"✅ Session started: {session_id}")
            print(f"📊 Participants: {result['participants']}")
            
            # Wait for completion
            print("⏳ Waiting for consensus...")
            for i in range(6):  # Wait up to 30 seconds
                time.sleep(5)
                
                # Check results
                result_response = requests.get(f"http://localhost:8080/results/{session_id}")
                if result_response.status_code == 200:
                    final_result = result_response.json()
                    
                    if final_result.get('status') == 'completed':
                        print("🎉 SUCCESS! Session completed!")
                        consensus_score = final_result.get('final_score', 0)
                        if isinstance(consensus_score, (int, float)):
                            print(f"📊 Consensus Score: {consensus_score:.3f}")
                        else:
                            print(f"📊 Consensus Score: {consensus_score}")
                        print(f"📋 Recommendation: {final_result.get('recommendation', 'N/A').upper()}")
                        print(f"🏦 Participants: {final_result.get('participant_consensus', {})}")
                        return True
                    else:
                        print(f"   Status: {final_result.get('status', 'unknown')}")
                        print(f"   Responses: {final_result.get('responses_received', 0)}/{final_result.get('total_participants', 0)}")
                else:
                    print(f"   Checking... ({i+1}/6)")
            
            print("⏰ Timeout waiting for results")
            return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_consortium_api()
