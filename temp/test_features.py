#!/usr/bin/env python3
"""Quick test to see what features are being generated"""

import sys
import os
sys.path.append('src')

from consortium.privacy_preserving_nlp import PrivacyPreservingNLP

nlp = PrivacyPreservingNLP()

# Test high fraud scenario
fraud_data = {
    'amount': 485000,
    'sender_account': 'ACC789012', 
    'receiver_account': 'ACC345678'
}

fraud_email = "CEO urgent wire transfer $485000 crypto investment opportunity confidential"

# Test legitimate scenario  
legit_data = {
    'amount': 50000,
    'sender_account': 'ACC123456',
    'receiver_account': 'ACC987654'
}

legit_email = "Invoice payment for office supplies as discussed"

print("🔍 FEATURE EXTRACTION TEST")
print("=" * 40)

fraud_features = nlp.convert_to_anonymous_features(fraud_data, fraud_email)
legit_features = nlp.convert_to_anonymous_features(legit_data, legit_email)

print(f"Fraud features:  {fraud_features[:10]}")
print(f"Legit features:  {legit_features[:10]}")

if fraud_features == legit_features:
    print("❌ PROBLEM: Features are identical!")
else:
    print("✅ Features are different")
