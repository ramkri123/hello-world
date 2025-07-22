#!/usr/bin/env python3
"""Direct test of the bank models"""

import sys
sys.path.append('src')

import joblib
import numpy as np
from consortium.privacy_preserving_nlp import PrivacyPreservingNLP

print("🧪 DIRECT BANK MODEL TEST")
print("=" * 40)

nlp = PrivacyPreservingNLP()

# Test high fraud scenario
fraud_data = {
    'amount': 485000,
    'sender_balance': 500000,
    'avg_daily_spending': 50000,
    'hour': 16,
    'day_of_week': 4,
    'is_holiday': False
}
fraud_email = "CEO urgent crypto investment wire transfer confidential"

fraud_features = nlp.convert_to_anonymous_features(fraud_data, fraud_email)
print(f"Fraud features: {fraud_features[:10]}")

# Test each bank model directly
for bank in ['A', 'B', 'C']:
    try:
        model = joblib.load(f'models/bank_{bank}_model.pkl')
        score = model.predict_proba([fraud_features])[0][1]
        print(f"Bank {bank} fraud score: {score:.3f}")
    except Exception as e:
        print(f"Bank {bank} ERROR: {e}")

# Test legitimate scenario
legit_data = {
    'amount': 50000,
    'sender_balance': 1000000,
    'avg_daily_spending': 75000,
    'hour': 14,
    'day_of_week': 2,
    'is_holiday': False
}
legit_email = "invoice payment office supplies monthly billing"

legit_features = nlp.convert_to_anonymous_features(legit_data, legit_email)
print(f"\nLegit features: {legit_features[:10]}")

for bank in ['A', 'B', 'C']:
    try:
        model = joblib.load(f'models/bank_{bank}_model.pkl')
        score = model.predict_proba([legit_features])[0][1]
        print(f"Bank {bank} legit score: {score:.3f}")
    except Exception as e:
        print(f"Bank {bank} ERROR: {e}")
