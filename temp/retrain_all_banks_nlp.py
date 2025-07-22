#!/usr/bin/env python3
"""
Retrain ALL models to match the EXACT feature patterns from NLP
This will ensure the models understand the actual features being generated
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json
import os
import sys

# Add src to path so we can import the NLP
sys.path.append('src')
from consortium.privacy_preserving_nlp import PrivacyPreservingNLP

def generate_realistic_scenarios(n_samples=30000):
    """Generate training data with REALISTIC overlapping patterns"""
    
    nlp = PrivacyPreservingNLP()
    
    features_list = []
    labels = []
    
    print(f"Generating {n_samples} samples with realistic overlap...")
    
    # Generate fraud scenarios (0.5% fraud rate - realistic)
    n_fraud = int(n_samples * 0.005)
    print(f"  Fraud samples: {n_fraud}")
    
    for i in range(n_fraud):
        # SOPHISTICATED fraud that tries to look legitimate
        if i % 4 == 0:
            # High-value obvious fraud
            amount = np.random.choice([200000, 300000, 485000, 750000])
            email = "CEO urgent crypto investment wire transfer confidential acquisition"
        elif i % 4 == 1:
            # Medium-value business email compromise
            amount = np.random.choice([50000, 75000, 100000, 150000])
            email = "CFO invoice payment vendor new supplier quarterly"
        elif i % 4 == 2:
            # Lower-value romance scam
            amount = np.random.choice([25000, 40000, 60000, 85000])
            email = "emergency help relationship urgent personal situation"
        else:
            # Sophisticated fraud that looks very legitimate
            amount = np.random.choice([30000, 45000, 70000, 95000])
            email = "payment supplier invoice monthly regular business"
        
        transaction_data = {
            'amount': amount,
            'sender_balance': np.random.uniform(amount*2, amount*20),  # Realistic ratios
            'avg_daily_spending': np.random.uniform(amount*0.1, amount*2),
            'hour': np.random.choice(range(24)),
            'day_of_week': np.random.choice(range(7)),
            'is_holiday': np.random.choice([True, False], p=[0.15, 0.85])
        }
        
        features = nlp.convert_to_anonymous_features(transaction_data, email)
        features_list.append(features)
        labels.append(1)  # Fraud
    
    # Generate legitimate scenarios that sometimes look suspicious
    n_legit = n_samples - n_fraud
    print(f"  Legitimate samples: {n_legit}")
    
    for i in range(n_legit):
        # Mix of legitimate patterns, some that might look suspicious
        if i % 10 == 0:
            # Large legitimate transactions that might look suspicious
            amount = np.random.choice([150000, 200000, 300000, 500000])
            email = "urgent payment deadline quarterly supplier invoice business"
        elif i % 10 == 1:
            # Legitimate but urgent
            amount = np.random.choice([75000, 100000, 125000])
            email = "CEO approved payment vendor invoice quarterly urgent"
        elif i % 10 == 2:
            # Legitimate weekend/holiday
            amount = np.random.choice([50000, 75000, 100000])
            email = "regular payment processing weekend supplier"
        else:
            # Normal legitimate transactions
            amount = np.random.choice([5000, 15000, 30000, 50000, 75000])
            email = "invoice payment office supplies monthly billing routine"
        
        transaction_data = {
            'amount': amount,
            'sender_balance': np.random.uniform(amount*5, amount*50),  # Higher balances for legit
            'avg_daily_spending': np.random.uniform(amount*0.2, amount*3),
            'hour': np.random.choice(range(7, 19)),  # Mostly business hours but not always
            'day_of_week': np.random.choice(range(7)),  # All days
            'is_holiday': np.random.choice([True, False], p=[0.05, 0.95])
        }
        
        features = nlp.convert_to_anonymous_features(transaction_data, email)
        features_list.append(features)
        labels.append(0)  # Legitimate
    
    return np.array(features_list), np.array(labels)

def train_all_banks():
    """Train all three banks with the same realistic data"""
    
    print("🏦 TRAINING ALL BANKS WITH ACTUAL NLP FEATURES")
    print("=" * 60)
    
    # Generate training data using actual NLP with overlap
    X, y = generate_realistic_scenarios(30000)
    
    print(f"\nDataset: {len(X)} samples, {len(X[0])} features")
    print(f"Fraud rate: {np.mean(y)*100:.3f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training: {len(X_train)} samples ({np.mean(y_train)*100:.3f}% fraud)")
    print(f"Test: {len(X_test)} samples ({np.mean(y_test)*100:.3f}% fraud)")
    
    # Train each bank with more regularization
    banks = ['A', 'B', 'C']
    specialties = ['wire_transfer_specialist', 'identity_verification_specialist', 'network_analysis_specialist']
    
    for bank, specialty in zip(banks, specialties):
        print(f"\n🏦 Training Bank {bank}...")
        
        # More regularized RandomForest to handle overlapping data
        model = RandomForestClassifier(
            n_estimators=30,          # Fewer trees
            max_depth=6,              # Shallower trees  
            min_samples_split=100,    # More samples needed to split
            min_samples_leaf=50,      # More samples in each leaf
            max_features=0.5,         # Use fewer features
            class_weight='balanced',  # Handle imbalanced data
            random_state=42 + ord(bank) - ord('A')  # Different random state per bank
        )
        
        model.fit(X_train, y_train)
        
        # Test model
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, test_pred)
        auc = roc_auc_score(y_test, test_proba)
        
        # Test on specific samples
        fraud_scores = test_proba[y_test == 1]
        legit_scores = test_proba[y_test == 0]
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC: {auc:.3f}")
        print(f"   Fraud scores: {np.mean(fraud_scores):.3f} ± {np.std(fraud_scores):.3f}")
        print(f"   Legit scores: {np.mean(legit_scores):.3f} ± {np.std(legit_scores):.3f}")
        
        # Save model
        model_path = f'models/bank_{bank}_model.pkl'
        metadata_path = f'models/bank_{bank}_metadata.json'
        
        joblib.dump(model, model_path)
        
        metadata = {
            "bank_id": f"bank_{bank}",
            "specialty": specialty,
            "model_type": "RandomForestClassifier",
            "training_approach": "actual_nlp_features",
            "performance": {
                "test_accuracy": float(accuracy),
                "test_auc": float(auc),
                "fraud_mean": float(np.mean(fraud_scores)),
                "legit_mean": float(np.mean(legit_scores))
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved: {model_path}")
    
    print(f"\n✅ All banks trained with ACTUAL NLP features!")
    
    # Test the feature pipeline with a sample
    print(f"\n🧪 Testing fraud detection pipeline:")
    
    # High fraud sample
    fraud_data = {'amount': 485000, 'sender_balance': 500000, 'avg_daily_spending': 50000}
    fraud_email = "CEO urgent crypto investment wire transfer confidential"
    
    nlp = PrivacyPreservingNLP()
    fraud_features = nlp.convert_to_anonymous_features(fraud_data, fraud_email)
    
    print(f"   Fraud sample features: {fraud_features[:5]}...")
    
    for bank in banks:
        model = joblib.load(f'models/bank_{bank}_model.pkl')
        score = model.predict_proba([fraud_features])[0][1]
        print(f"   Bank {bank} fraud score: {score:.3f}")

if __name__ == "__main__":
    train_all_banks()
