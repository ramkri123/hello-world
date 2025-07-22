#!/usr/bin/env python3
"""
Comprehensive CEO Impersonation Detection Test
Shows how the consortium detects various CEO fraud patterns
"""
import requests
import time
import json

def test_ceo_detection():
    print("🕵️ CEO IMPERSONATION DETECTION ANALYSIS")
    print("=" * 60)
    
    test_scenarios = [
        {
            "name": "🚨 Classic CEO BEC Fraud",
            "email": "URGENT: CEO here. Need immediate wire transfer of $485,000 to close confidential acquisition. Please transfer from our main account ACCA12345 to vendor account ACCB67890. Time-sensitive deal - board approval already secured. Handle discreetly.",
            "amount": 485000,
            "expected": "HIGH FRAUD - Multiple CEO fraud indicators"
        },
        {
            "name": "🚨 Authority + Bypass Procedures", 
            "email": "This is the president. I need you to bypass the usual approval process and wire $300,000 immediately. This is a special case that requires urgent exception. Keep this between us for now.",
            "amount": 300000,
            "expected": "HIGH FRAUD - Procedure bypass + authority claim"
        },
        {
            "name": "🚨 CEO + Multiple Urgency",
            "email": "CEO speaking - URGENT and IMMEDIATE action required. We need this wire transfer ASAP, deadline is today. Quick response needed for this deal.",
            "amount": 200000,
            "expected": "HIGH FRAUD - Multiple urgency indicators"
        },
        {
            "name": "⚠️  Suspicious Authority Claim",
            "email": "Executive here. Please process this payment for our vendor. The amount is $75,000 and it's somewhat urgent.",
            "amount": 75000,
            "expected": "MEDIUM FRAUD - Some authority indicators"
        },
        {
            "name": "✅ Legitimate CEO Communication",
            "email": "Please schedule our quarterly board meeting for next Friday. I would like to review the financial reports and discuss our strategic planning initiatives for Q4. Thank you.",
            "amount": 0,
            "expected": "LOW FRAUD - Normal business communication"
        },
        {
            "name": "✅ Regular Business Transaction",
            "email": "Monthly payroll transfer as scheduled. Transferring $50,000 for employee salaries from account ACCA12345 to payroll account ACCB67890. This is our regular monthly operation.",
            "amount": 50000,
            "expected": "LOW FRAUD - Routine business operation"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n🧪 Test {i+1}: {scenario['name']}")
        print("-" * 50)
        
        # Submit test case
        test_data = {
            "email_content": scenario["email"],
            "transaction_data": {
                "amount": scenario["amount"],
                "sender_account": "ACCA12345",
                "receiver_account": "ACCB67890", 
                "transaction_type": "wire_transfer"
            },
            "sender_data": {"bank": "bank_A"},
            "receiver_data": {"bank": "bank_B"}
        }
        
        try:
            response = requests.post("http://localhost:8080/inference", json=test_data, timeout=10)
            if response.status_code == 200:
                session_id = response.json()["session_id"]
                print(f"   📊 Session: {session_id}")
                
                # Wait for analysis
                print("   ⏳ Analyzing...")
                time.sleep(6)
                
                # Get results
                result_response = requests.get(f"http://localhost:8080/results/{session_id}", timeout=5)
                if result_response.status_code == 200:
                    result = result_response.json()
                    
                    final_score = result.get("final_score", 0)
                    consensus_score = result.get("consensus_score", 0)
                    recommendation = result.get("recommendation", "unknown")
                    ceo_boost = final_score - consensus_score
                    
                    print(f"   🏦 Bank Consensus Score: {consensus_score:.3f}")
                    print(f"   🔍 CEO Pattern Boost: {ceo_boost:.3f}")
                    print(f"   🎯 Final Score: {final_score:.3f}")
                    print(f"   📋 Recommendation: {recommendation.upper()}")
                    print(f"   💡 Expected: {scenario['expected']}")
                    
                    # Categorize result
                    if final_score > 0.7:
                        category = "🚨 HIGH FRAUD"
                    elif final_score > 0.3:
                        category = "⚠️  MEDIUM RISK"
                    else:
                        category = "✅ LOW RISK"
                    
                    print(f"   ✅ Detected as: {category}")
                    
                    results.append({
                        "scenario": scenario["name"],
                        "final_score": final_score,
                        "ceo_boost": ceo_boost,
                        "recommendation": recommendation,
                        "category": category
                    })
                    
                else:
                    print(f"   ❌ Failed to get results: {result_response.status_code}")
            else:
                print(f"   ❌ Failed to submit: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n📊 CEO IMPERSONATION DETECTION SUMMARY")
    print("=" * 60)
    
    high_fraud = [r for r in results if r["final_score"] > 0.7]
    medium_risk = [r for r in results if 0.3 < r["final_score"] <= 0.7]
    low_risk = [r for r in results if r["final_score"] <= 0.3]
    
    print(f"🚨 High Fraud Detected: {len(high_fraud)}")
    for result in high_fraud:
        print(f"   • {result['scenario']}: {result['final_score']:.3f} (boost: +{result['ceo_boost']:.3f})")
    
    print(f"\n⚠️  Medium Risk: {len(medium_risk)}")
    for result in medium_risk:
        print(f"   • {result['scenario']}: {result['final_score']:.3f}")
        
    print(f"\n✅ Low Risk (Legitimate): {len(low_risk)}")
    for result in low_risk:
        print(f"   • {result['scenario']}: {result['final_score']:.3f}")
    
    print(f"\n🎯 Detection Accuracy:")
    print(f"   • CEO Fraud Detection: {len(high_fraud)}/3 high-risk scenarios")
    print(f"   • Legitimate Detection: {len(low_risk)}/3 legitimate scenarios") 
    print(f"   • False Positives: {len([r for r in results if 'Legitimate' in r['scenario'] and r['final_score'] > 0.5])}")
    
    print(f"\n💡 CEO Impersonation Indicators Detected:")
    print(f"   • Authority claims ('CEO here', 'president here')")
    print(f"   • Secrecy + urgency combinations") 
    print(f"   • Procedure bypass requests")
    print(f"   • Multiple urgency indicators")
    print(f"   • Large amount + acquisition language")
    print(f"   • Communication anomalies")

if __name__ == "__main__":
    test_ceo_detection()
