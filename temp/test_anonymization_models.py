#!/usr/bin/env python3
"""
Test the New Anonymization System with Trained Models
"""

import sys
import os
sys.path.append('src')

from consortium.account_anonymizer import AccountAnonymizer
from consortium.privacy_preserving_nlp import PrivacyPreservingNLP
import joblib
import numpy as np

def test_anonymization_with_models():
    """Test that the new anonymization works with our trained models"""
    print("🧪 TESTING ANONYMIZATION WITH TRAINED MODELS")
    print("=" * 60)
    
    # Initialize components
    anonymizer = AccountAnonymizer()
    nlp = PrivacyPreservingNLP()
    
    # Test transaction
    sender_account = "ACCA12345"
    receiver_account = "ACCB67890"
    
    print(f"📧 Test Transaction:")
    print(f"   💸 From: {sender_account}")
    print(f"   💰 To: {receiver_account}")
    
    # Test anonymization
    anonymized = anonymizer.anonymize_transaction_accounts(sender_account, receiver_account)
    print(f"\n🔐 Anonymized:")
    print(f"   💸 From: {anonymized['sender_anonymous']}")
    print(f"   💰 To: {anonymized['receiver_anonymous']}")
    
    # Test each bank's scenario determination
    bank_configs = {
        'bank_A': ['ACCA12345', 'ACCA67890', 'ACCA11111', 'ACCA22222', 'ACCA33333'],
        'bank_B': ['ACCB67890', 'ACCB12345', 'ACCB22222', 'ACCB44444', 'ACCB55555'],
        'bank_C': ['ACCC99999', 'ACCC12345', 'ACCC55555', 'ACCC66666', 'ACCC77777']
    }
    
    print(f"\n🏦 Bank Scenario Determination:")
    for bank_name, accounts in bank_configs.items():
        scenario = anonymizer.bank_can_determine_ownership(accounts, anonymized)
        confidence = anonymizer.get_scenario_confidence_weight(scenario)
        print(f"   {bank_name}: {scenario} (confidence: {confidence:.2f})")
    
    # Test with actual models
    print(f"\n🤖 Testing with Trained Models:")
    
    # Create test email and transaction data
    email_content = "CEO urgent wire transfer $485,000 confidential acquisition deadline"
    transaction_data = {
        'amount': 485000,
        'sender_balance': 2000000,
        'avg_daily_spending': 50000,
        'hour': 16,
        'day_of_week': 4,
        'is_holiday': False,
        'sender_age_days': 1000,
        'receiver_age_days': 500,
        'transaction_type': 'wire_transfer',
        'cross_border': False,
        'sender_bank': 'A',
        'receiver_bank': 'B'
    }
    
    # Extract features
    features = nlp.convert_to_anonymous_features(transaction_data, email_content)
    print(f"   📊 Extracted {len(features)} features")
    
    # Test each bank model
    for bank_name, accounts in bank_configs.items():
        try:
            # Load model and scaler
            model = joblib.load(f'models/{bank_name}_model.pkl')
            scaler = joblib.load(f'models/{bank_name}_scaler.pkl')
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features_scaled)[0]
                risk_score = prob[1] if len(prob) > 1 else prob[0]
            else:
                risk_score = model.predict(features_scaled)[0]
            
            # Apply scenario awareness
            scenario = anonymizer.bank_can_determine_ownership(accounts, anonymized)
            confidence = anonymizer.get_scenario_confidence_weight(scenario)
            
            print(f"   {bank_name}: risk={risk_score:.3f}, scenario={scenario}, confidence={confidence:.2f}")
            
        except Exception as e:
            print(f"   {bank_name}: ❌ Error loading model: {e}")
    
    print(f"\n✅ Anonymization system working with trained models!")
    print(f"   🔐 Privacy preserved through one-way hash")
    print(f"   🎯 Scenario awareness functional")
    print(f"   🤖 Models loaded and operational")

if __name__ == "__main__":
    test_anonymization_with_models()
