#!/usr/bin/env python3
"""
Test the new privacy-preserving consortium with NLP feature extraction
"""

import json
import requests
import time
from privacy_preserving_nlp import PrivacyPreservingNLP, create_demo_bec_email

def test_privacy_preserving_consortium():
    """Test the consortium with anonymous feature extraction"""
    
    consortium_url = "http://localhost:8080"
    
    print("🔍 Testing Privacy-Preserving Consortium with NLP...")
    
    # Create demo BEC transaction with email
    bec_email = create_demo_bec_email()
    
    transaction_data = {
        'amount': 485000,
        'sender_balance': 2300000,
        'avg_daily_spending': 50000,
        'hour': 16,  # 4 PM Friday
        'day_of_week': 4,  # Friday
        'is_holiday': False
    }
    
    sender_data = {
        'account_age_years': 6.0,
        'risk_score': 0.05,
        'transaction_count': 2000,
        'business_type': 'business',
        'geographic_risk': 0.1,
        'bank': 'bank_A'
    }
    
    receiver_data = {
        'account_age_years': 0.008,  # 3 days old account
        'risk_score': 0.8,
        'verification_score': 0.2,
        'bank': 'bank_B'
    }
    
    # Submit transaction with raw data for NLP processing
    print(f"\n📧 Submitting BEC Transaction for Analysis...")
    print(f"   💰 Amount: ${transaction_data['amount']:,.2f}")
    print(f"   📧 Email: {len(bec_email)} characters")
    print(f"   🏦 Sender Bank: {sender_data['bank']}")
    print(f"   🏦 Receiver Bank: {receiver_data['bank']}")
    
    inference_data = {
        'transaction_data': transaction_data,
        'email_content': bec_email,
        'sender_data': sender_data,
        'receiver_data': receiver_data,
        'use_case': 'bec_fraud_detection'
    }
    
    try:
        # Submit inference request
        response = requests.post(f"{consortium_url}/inference", json=inference_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result['session_id']
            print(f"✅ Inference submitted successfully")
            print(f"   🆔 Session ID: {session_id}")
            print(f"   👥 Participants: {result['participants']}")
            
            # Wait for results
            print(f"\n⏳ Waiting for consortium analysis...")
            time.sleep(5)  # Give time for processing
            
            # Get results
            results_response = requests.get(f"{consortium_url}/results/{session_id}", timeout=10)
            
            if results_response.status_code == 200:
                results = results_response.json()
                
                print(f"\n🎯 CONSORTIUM ANALYSIS RESULTS:")
                print(f"   📊 Final Score: {results.get('final_score', 'N/A'):.3f}")
                print(f"   🎯 Recommendation: {results.get('recommendation', 'N/A').upper()}")
                print(f"   📈 Consensus Score: {results.get('consensus_score', 'N/A'):.3f}")
                print(f"   📊 Variance: {results.get('variance', 'N/A'):.3f}")
                
                print(f"\n🏦 INDIVIDUAL BANK SCORES:")
                individual_scores = results.get('individual_scores', {})
                for bank_id, score in individual_scores.items():
                    print(f"   {bank_id}: {score:.3f}")
                
                print(f"\n👥 PARTICIPANT CONSENSUS:")
                consensus = results.get('participant_consensus', {})
                print(f"   Total Participants: {consensus.get('total', 'N/A')}")
                print(f"   High Risk Flags: {consensus.get('high_risk', 'N/A')}")
                print(f"   Low Risk Flags: {consensus.get('low_risk', 'N/A')}")
                
                # Show specialist insights
                insights = results.get('specialist_insights', [])
                if insights:
                    print(f"\n🔍 SPECIALIST INSIGHTS:")
                    for insight in insights:
                        print(f"   {insight['specialty']}: {insight['risk_level']} risk (confidence: {insight['confidence']:.2f})")
                
                return results
                
            else:
                print(f"❌ Failed to get results: {results_response.status_code}")
                print(f"   Error: {results_response.text}")
                
        else:
            print(f"❌ Failed to submit inference: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing consortium: {e}")
    
    return None

def test_traditional_features():
    """Test with traditional feature array (for comparison)"""
    
    consortium_url = "http://localhost:8080"
    
    print(f"\n📊 Testing with Traditional Feature Array...")
    
    # Create traditional feature array
    nlp = PrivacyPreservingNLP()
    bec_email = create_demo_bec_email()
    
    transaction_data = {
        'amount': 485000,
        'sender_balance': 2300000,
        'avg_daily_spending': 50000,
        'hour': 16,
        'day_of_week': 4,
        'is_holiday': False
    }
    
    sender_data = {
        'account_age_years': 6.0,
        'risk_score': 0.05,
        'transaction_count': 2000,
        'business_type': 'business',
        'geographic_risk': 0.1,
        'bank': 'bank_A'
    }
    
    receiver_data = {
        'account_age_years': 0.008,
        'risk_score': 0.8,
        'verification_score': 0.2,
        'bank': 'bank_B'
    }
    
    # Extract features locally
    features = nlp.convert_to_anonymous_features(
        transaction_data, bec_email, sender_data, receiver_data
    )
    
    print(f"   📈 Generated {len(features)} anonymous features")
    print(f"   🔑 Key features: Authority={features[10]:.3f}, Urgency={features[13]:.3f}, NewAccount={features[31]:.3f}")
    
    inference_data = {
        'features': features,
        'use_case': 'bec_fraud_detection'
    }
    
    try:
        response = requests.post(f"{consortium_url}/inference", json=inference_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result['session_id']
            print(f"✅ Traditional inference submitted: {session_id}")
            
            # Wait and get results
            time.sleep(5)
            results_response = requests.get(f"{consortium_url}/results/{session_id}", timeout=10)
            
            if results_response.status_code == 200:
                results = results_response.json()
                print(f"   🎯 Result: {results.get('recommendation', 'N/A')} (score: {results.get('final_score', 0):.3f})")
                return results
                
    except Exception as e:
        print(f"❌ Error with traditional features: {e}")
    
    return None

def main():
    """Main test function"""
    print("🚀 PRIVACY-PRESERVING CONSORTIUM TEST")
    print("=" * 50)
    
    # Test 1: Raw transaction data with NLP
    print("\n🧪 TEST 1: Raw Transaction Data + Email (Privacy-Preserving NLP)")
    nlp_results = test_privacy_preserving_consortium()
    
    # Test 2: Traditional feature array
    print("\n🧪 TEST 2: Pre-processed Feature Array (Traditional)")
    traditional_results = test_traditional_features()
    
    # Compare results
    if nlp_results and traditional_results:
        print(f"\n📊 COMPARISON:")
        print(f"   NLP Approach:        {nlp_results.get('recommendation', 'N/A')} (score: {nlp_results.get('final_score', 0):.3f})")
        print(f"   Traditional Approach: {traditional_results.get('recommendation', 'N/A')} (score: {traditional_results.get('final_score', 0):.3f})")
        
        print(f"\n✅ Privacy-Preserving NLP Integration: {'SUCCESS' if nlp_results.get('final_score', 0) > 0.4 else 'NEEDS TUNING'}")

if __name__ == "__main__":
    main()
