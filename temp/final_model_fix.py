#!/usr/bin/env python3
"""
FINAL FIX: Retrain models to match EXACT consortium hub feature extraction
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json
import sys

sys.path.append('src')
from consortium.privacy_preserving_nlp import PrivacyPreservingNLP

def generate_final_training_data(n_samples=25000):
    """Generate training data that exactly matches consortium hub processing"""
    
    nlp = PrivacyPreservingNLP()
    features_list = []
    labels = []
    
    n_fraud = int(n_samples * 0.005)  # 0.5% fraud
    
    print(f"Generating {n_samples} samples matching consortium hub exactly...")
    print(f"  Fraud: {n_fraud}, Legitimate: {n_samples - n_fraud}")
    
    # Fraud scenarios
    for i in range(n_fraud):
        amount = np.random.choice([75000, 150000, 300000, 485000, 750000])
        
        transaction_data = {
            'amount': amount,
            'sender_account': 'ACC123456',
            'receiver_account': 'ACC789012', 
            'transaction_type': 'wire_transfer'
        }
        
        email_content = np.random.choice([
            "CEO urgent crypto investment wire transfer confidential",
            "President emergency payment strategic acquisition secret", 
            "Executive director immediate transfer vendor confidential",
            "CFO urgent payment business opportunity exclusive"
        ])
        
        # Call NLP the SAME way consortium hub does (with sender_data=None, receiver_data=None)
        features = nlp.convert_to_anonymous_features(
            transaction_data, email_content, sender_data=None, receiver_data=None
        )
        
        features_list.append(features)
        labels.append(1)
    
    # Legitimate scenarios  
    for i in range(n_samples - n_fraud):
        amount = np.random.choice([10000, 25000, 50000, 100000, 200000])
        
        transaction_data = {
            'amount': amount,
            'sender_account': 'ACC654321',
            'receiver_account': 'ACC321987',
            'transaction_type': 'wire_transfer'
        }
        
        email_content = np.random.choice([
            "Invoice payment office supplies monthly billing",
            "Vendor payment contracted services quarterly",
            "Regular payment insurance premium",
            "Supplier payment materials standard order"
        ])
        
        # Call NLP the SAME way consortium hub does
        features = nlp.convert_to_anonymous_features(
            transaction_data, email_content, sender_data=None, receiver_data=None
        )
        
        features_list.append(features)
        labels.append(0)
    
    return np.array(features_list), np.array(labels)

def main():
    print("🔧 FINAL MODEL RETRAINING - Matching Consortium Hub Exactly")
    print("=" * 60)
    
    # Generate training data
    X, y = generate_final_training_data(25000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Test data: {len(X_test)} samples")
    
    # Train all banks with same approach
    for bank in ['A', 'B', 'C']:
        print(f"\n🏦 Training Bank {bank}...")
        
        model = RandomForestClassifier(
            n_estimators=40,
            max_depth=7,
            min_samples_split=80,
            min_samples_leaf=40,
            class_weight='balanced',
            random_state=42 + ord(bank) - ord('A')
        )
        
        model.fit(X_train, y_train)
        
        # Test
        test_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, test_proba)
        
        fraud_scores = test_proba[y_test == 1]
        legit_scores = test_proba[y_test == 0]
        
        print(f"   Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        print(f"   Fraud avg: {np.mean(fraud_scores):.3f}")
        print(f"   Legit avg: {np.mean(legit_scores):.3f}")
        
        # Save
        joblib.dump(model, f'models/bank_{bank}_model.pkl')
        print(f"   ✅ Saved bank_{bank}_model.pkl")
    
    # Test with actual consortium-style call
    print(f"\n🧪 Testing with consortium-style call:")
    
    nlp = PrivacyPreservingNLP()
    
    test_transaction = {
        'amount': 485000,
        'sender_account': 'ACC789012',
        'receiver_account': 'ACC345678',
        'transaction_type': 'wire_transfer'
    }
    
    test_email = "CEO urgent crypto investment wire transfer confidential"
    
    # Extract features the SAME way consortium does
    test_features = nlp.convert_to_anonymous_features(
        test_transaction, test_email, sender_data=None, receiver_data=None
    )
    
    print(f"Test features: {test_features[:5]}...")
    
    for bank in ['A', 'B', 'C']:
        model = joblib.load(f'models/bank_{bank}_model.pkl')
        score = model.predict_proba([test_features])[0][1]
        print(f"Bank {bank} fraud score: {score:.3f}")

if __name__ == "__main__":
    main()
