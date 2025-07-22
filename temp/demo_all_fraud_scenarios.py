#!/usr/bin/env python3
"""
Comprehensive Fraud Scenario Demo
Tests all fraud scenarios with the distributed consortium system
"""

import requests
import time
import json

def test_scenario(name, scenario_data, api_url="http://localhost:8080"):
    """Test a specific fraud scenario"""
    print(f"\n🎯 TESTING: {name}")
    print("=" * 60)
    
    # Prepare the transaction data
    transaction_data = {
        "email_content": scenario_data['email'],
        "transaction_data": {
            "amount": scenario_data['amount'],
            "sender_account": scenario_data['sender'],
            "receiver_account": scenario_data['receiver'],
            "transaction_type": scenario_data['type']
        },
        "use_case": "fraud_detection"
    }
    
    print(f"📧 Email Preview: {scenario_data['email'][:100]}...")
    print(f"💰 Amount: ${scenario_data['amount']:,}")
    print(f"🔄 {scenario_data['sender']} → {scenario_data['receiver']}")
    print(f"📝 Type: {scenario_data['type']}")
    
    try:
        # Submit to consortium
        print(f"\n📤 Submitting to consortium...")
        response = requests.post(
            f"{api_url}/inference",
            json=transaction_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ ANALYSIS COMPLETE")
            print(f"📊 Consensus Score: {result.get('consensus_score', 'N/A')}")
            
            # Show individual bank scores
            individual_scores = result.get('individual_scores', {})
            if individual_scores:
                print(f"🏦 Bank Scores:")
                for bank, score in individual_scores.items():
                    print(f"   {bank}: {score:.3f}")
            
            # Show anonymization
            if 'anonymized_accounts' in result:
                anon = result['anonymized_accounts']
                print(f"🔐 Anonymized: {anon.get('sender_anonymous', 'N/A')} → {anon.get('receiver_anonymous', 'N/A')}")
            
            # Risk assessment
            consensus = result.get('consensus_score', 0)
            if consensus > 0.7:
                print(f"🚨 HIGH RISK - Likely fraud detected!")
            elif consensus > 0.4:
                print(f"⚠️  MEDIUM RISK - Requires review")
            else:
                print(f"✅ LOW RISK - Likely legitimate")
                
            return True
            
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def main():
    """Demo all fraud scenarios"""
    print("🌟 COMPREHENSIVE FRAUD SCENARIO DEMO")
    print("=" * 70)
    print("Testing all fraud scenarios with the distributed consortium...")
    
    # All the fraud scenarios from the UI
    scenarios = {
        '🏢 CEO Business Email Compromise': {
            'sender': 'ACCA12345',
            'receiver': 'ACCB67890',
            'amount': 485000,
            'type': 'wire_transfer',
            'email': """Hi John,

This is CEO Sarah Wilson. We have an urgent strategic acquisition opportunity that requires immediate action. Please wire $485,000 to our new strategic partner Global Tech Solutions for the acquisition deposit.

This is highly confidential and time sensitive - we need to complete this before market close Friday. Please process this immediately and don't discuss with anyone else on the team.

Thanks for your help with this critical transaction.

Best regards,
Sarah Wilson
CEO"""
        },
        
        '₿ Cryptocurrency Investment Scam': {
            'sender': 'ACCA67890',
            'receiver': 'ACCB22222',
            'amount': 250000,
            'type': 'wire_transfer',
            'email': """URGENT: Crypto Investment Opportunity - LIMITED TIME

Dear Investor,

This is a once-in-a-lifetime opportunity to invest in the new CryptoMax currency before it goes public. Our algorithm has identified massive profit potential.

Send $250,000 IMMEDIATELY to secure your position. This offer expires in 4 hours. Don't miss out on 10x returns!

Wire to our secure account for immediate processing. Complete confidentiality guaranteed.

Best regards,
Investment Director"""
        },
        
        '🔒 Ransomware Payment': {
            'sender': 'ACCB12345',
            'receiver': 'ACCC99999',
            'amount': 75000,
            'type': 'wire_transfer',
            'email': """Your files have been encrypted. Pay $75,000 in Bitcoin equivalent to decrypt your systems. Wire transfer to this account for Bitcoin purchase. You have 48 hours before files are permanently deleted. Do not contact authorities."""
        },
        
        '📄 Fake Invoice Fraud': {
            'sender': 'ACCB12345',
            'receiver': 'ACCC77777',
            'amount': 180000,
            'type': 'wire_transfer',
            'email': """RE: Invoice #INV-2024-8832 - URGENT PAYMENT REQUIRED

Dear Accounts Payable,

Our records indicate that Invoice #INV-2024-8832 for $180,000 is now 45 days overdue. We have updated our banking details due to a merger.

NEW PAYMENT DETAILS:
Account: ACCC77777
Reference: INV-2024-8832

Please process payment immediately to avoid service interruption and late fees. Our legal team has been notified of this overdue payment.

Best regards,
Global Services Ltd
Accounts Receivable"""
        },
        
        '🎰 Lottery/Prize Scam': {
            'sender': 'ACCA11111',
            'receiver': 'ACCB44444',
            'amount': 25000,
            'type': 'wire_transfer',
            'email': """🎉 CONGRATULATIONS! YOU'VE WON THE INTERNATIONAL LOTTERY! 🎉

Dear Lucky Winner,

You have been selected as the winner of $2,500,000 in the International Business Lottery! Your winning number is: ILB-2024-7749.

To claim your prize, you must pay the processing fee of $25,000 immediately. This covers:
- Government taxes and fees
- International transfer charges  
- Prize verification costs

Wire the fee to account ACCB44444 within 48 hours or forfeit your winnings!

Congratulations again!
International Lottery Commission"""
        },
        
        '📈 Ponzi Investment Scheme': {
            'sender': 'ACCA22222',
            'receiver': 'ACCB11111',
            'amount': 450000,
            'type': 'wire_transfer',
            'email': """EXCLUSIVE: Quantum AI Trading Fund - Early Investor Opportunity

Dear High-Net-Worth Investor,

Congratulations! You've been selected for our exclusive Quantum AI Trading Fund based on your investment profile.

Our proprietary AI algorithm has generated:
- 47% returns in Q3 2024
- 52% returns in Q4 2024  
- Projected 65% for Q1 2025

Minimum investment: $450,000
Expected annual return: 180%
Risk level: Guaranteed principal protection

This opportunity is limited to 25 investors worldwide. We have 3 spots remaining.

Wire your investment to ACCB11111 by midnight tonight to secure your position.

Best regards,
Dr. Michael Chen, PhD
Quantum Financial Technologies"""
        },
        
        '✅ Legitimate Business Payment': {
            'sender': 'ACCA12345',
            'receiver': 'ACCB67890',
            'amount': 50000,
            'type': 'ach',
            'email': """Monthly vendor payment for office supplies and equipment maintenance. Invoice #12455 dated November 15, 2024. Net 30 payment terms. Standard business transaction per our service agreement."""
        }
    }
    
    # Test each scenario
    successful_tests = 0
    total_tests = len(scenarios)
    
    for name, data in scenarios.items():
        success = test_scenario(name, data)
        if success:
            successful_tests += 1
        
        # Wait between tests to not overwhelm the system
        time.sleep(2)
    
    # Summary
    print(f"\n🎯 DEMO SUMMARY")
    print("=" * 40)
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"📊 Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print(f"\n🎉 ALL FRAUD SCENARIOS TESTED SUCCESSFULLY!")
        print(f"🔐 Privacy-preserving distributed consortium is operational")
        print(f"🏦 All banks participating in fraud detection")
        print(f"🎯 Scenario-aware confidence weighting active")
    else:
        print(f"\n⚠️  Some tests failed - check consortium system status")

if __name__ == "__main__":
    main()
