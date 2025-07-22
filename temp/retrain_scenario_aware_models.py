#!/usr/bin/env python3
"""
Retrain bank models with one-way hash anonymization and scenario-aware training
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

# Add src to path for imports
sys.path.append('src')
from consortium.privacy_preserving_nlp import PrivacyPreservingNLP
from consortium.account_anonymizer import AccountAnonymizer

# Bank account assignments for scenario-aware training
BANK_ACCOUNTS = {
    'bank_A': ['ACCA12345', 'ACCA67890', 'ACCA11111', 'ACCA99999', 'ACCA55555'],
    'bank_B': ['ACCB67890', 'ACCB12345', 'ACCB22222', 'ACCB88888', 'ACCB44444'],
    'bank_C': ['ACCC99999', 'ACCC12345', 'ACCC55555', 'ACCC77777', 'ACCC33333']
}

def generate_scenario_aware_training_data(bank_name, n_samples=30000):
    """Generate training data with scenario-aware confidence weights for specific bank"""
    
    nlp = PrivacyPreservingNLP()
    
    features_list = []
    labels = []
    confidences = []
    scenarios = []
    
    print(f"🏦 Generating {n_samples} samples for {bank_name} with scenario awareness...")
    
    # Get this bank's accounts
    bank_accounts = BANK_ACCOUNTS[bank_name]
    all_accounts = []
    for accounts in BANK_ACCOUNTS.values():
        all_accounts.extend(accounts)
    
    # Generate fraud scenarios (0.5% fraud rate - realistic)
    n_fraud = int(n_samples * 0.005)
    print(f"  Fraud samples: {n_fraud}")
    
    for i in range(n_fraud):
        # Generate fraud transaction
        sender_account = np.random.choice(all_accounts)
        receiver_account = np.random.choice(all_accounts)
        
        # Ensure sender != receiver
        while sender_account == receiver_account:
            receiver_account = np.random.choice(all_accounts)
        
        # Create anonymized identifiers
        anonymized = AccountAnonymizer.anonymize_transaction_accounts(sender_account, receiver_account)
        
        # Determine scenario for this bank
        scenario = AccountAnonymizer.bank_can_determine_ownership(bank_accounts, anonymized)
        confidence_weight = AccountAnonymizer.get_scenario_confidence_weight(scenario)
        
        # Generate fraud email based on scenario type
        if i % 4 == 0:
            # High-value CEO fraud
            amount = np.random.choice([200000, 300000, 485000, 750000])
            email = "CEO urgent crypto investment wire transfer confidential acquisition merger"
        elif i % 4 == 1:
            # Business email compromise
            amount = np.random.choice([50000, 75000, 100000, 150000])
            email = "CFO invoice payment vendor supplier urgent deadline quarterly"
        elif i % 4 == 2:
            # Romance scam
            amount = np.random.choice([25000, 40000, 60000, 85000])
            email = "emergency help relationship urgent personal situation hospital"
        else:
            # Sophisticated fraud
            amount = np.random.choice([75000, 125000, 200000])
            email = "invoice payment consulting services due quarterly business"
        
        # Create transaction data
        transaction_data = {
            'amount': amount,
            'sender_account': sender_account,
            'receiver_account': receiver_account,
            'sender_balance': amount * np.random.uniform(0.8, 5.0),
            'receiver_balance': np.random.uniform(10000, 1000000),
            'avg_daily_spending': amount * np.random.uniform(0.1, 0.8),
            'hour': np.random.randint(0, 24),
            'day_of_week': np.random.randint(0, 7),
            'is_holiday': np.random.choice([True, False], p=[0.1, 0.9]),
            'transaction_type': np.random.choice(['wire_transfer', 'ach', 'check'])
        }
        
        # Generate features using NLP
        features = nlp.convert_to_anonymous_features(transaction_data, email)
        
        features_list.append(features)
        labels.append(1)  # Fraud
        confidences.append(confidence_weight)
        scenarios.append(scenario)
    
    # Generate legitimate scenarios
    n_legit = n_samples - n_fraud
    print(f"  Legitimate samples: {n_legit}")
    
    for i in range(n_legit):
        # Generate legitimate transaction
        sender_account = np.random.choice(all_accounts)
        receiver_account = np.random.choice(all_accounts)
        
        # Ensure sender != receiver
        while sender_account == receiver_account:
            receiver_account = np.random.choice(all_accounts)
        
        # Create anonymized identifiers
        anonymized = AccountAnonymizer.anonymize_transaction_accounts(sender_account, receiver_account)
        
        # Determine scenario for this bank
        scenario = AccountAnonymizer.bank_can_determine_ownership(bank_accounts, anonymized)
        confidence_weight = AccountAnonymizer.get_scenario_confidence_weight(scenario)
        
        # Generate legitimate email
        if i % 4 == 0:
            amount = np.random.uniform(1000, 50000)
            email = "invoice payment office supplies monthly billing vendor"
        elif i % 4 == 1:
            amount = np.random.uniform(5000, 75000)
            email = "salary payment employee payroll quarterly bonus"
        elif i % 4 == 2:
            amount = np.random.uniform(2000, 30000)
            email = "utility bill payment monthly electricity water services"
        else:
            amount = np.random.uniform(10000, 100000)
            email = "contract payment services rendered consulting project"
        
        # Create transaction data
        transaction_data = {
            'amount': amount,
            'sender_account': sender_account,
            'receiver_account': receiver_account,
            'sender_balance': amount * np.random.uniform(2.0, 10.0),
            'receiver_balance': np.random.uniform(10000, 500000),
            'avg_daily_spending': amount * np.random.uniform(0.5, 2.0),
            'hour': np.random.randint(8, 18),  # Business hours
            'day_of_week': np.random.randint(0, 5),  # Weekdays
            'is_holiday': False,
            'transaction_type': np.random.choice(['ach', 'check', 'wire_transfer'])
        }
        
        # Generate features using NLP
        features = nlp.convert_to_anonymous_features(transaction_data, email)
        
        features_list.append(features)
        labels.append(0)  # Legitimate
        confidences.append(confidence_weight)
        scenarios.append(scenario)
    
    print(f"✅ Generated {len(features_list)} samples for {bank_name}")
    
    # Show scenario distribution
    scenario_counts = {}
    for scenario in scenarios:
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    print(f"📊 Scenario distribution for {bank_name}:")
    for scenario, count in scenario_counts.items():
        weight = AccountAnonymizer.get_scenario_confidence_weight(scenario)
        pct = (count / len(scenarios)) * 100
        print(f"   {scenario}: {count} samples ({pct:.1f}%) - weight {weight:.2f}")
    
    return np.array(features_list), np.array(labels), np.array(confidences), scenarios

def train_scenario_aware_model(bank_name):
    """Train a scenario-aware model for a specific bank"""
    print(f"\n🚀 TRAINING SCENARIO-AWARE MODEL FOR {bank_name.upper()}")
    print("=" * 60)
    
    # Generate training data
    X, y, confidences, scenarios = generate_scenario_aware_training_data(bank_name, 30000)
    
    print(f"📊 Training data shape: {X.shape}")
    print(f"📊 Fraud rate: {(y.sum() / len(y)) * 100:.2f}%")
    print(f"📊 Average confidence: {confidences.mean():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test, conf_train, conf_test = train_test_split(
        X, y, confidences, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with scenario-aware sample weights
    print(f"🤖 Training Random Forest with scenario-aware weighting...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    # Use confidence weights as sample weights during training
    model.fit(X_train, y_train, sample_weight=conf_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_auc = roc_auc_score(y_train, train_pred_proba)
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    print(f"📈 Training Accuracy: {train_acc:.3f}")
    print(f"📈 Test Accuracy: {test_acc:.3f}")
    print(f"📈 Training AUC: {train_auc:.3f}")
    print(f"📈 Test AUC: {test_auc:.3f}")
    
    # Save model
    model_path = f'models/{bank_name}_model.pkl'
    joblib.dump(model, model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'bank_name': bank_name,
        'training_samples': len(X),
        'fraud_rate': float(y.sum() / len(y)),
        'avg_confidence': float(confidences.mean()),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'feature_count': X.shape[1],
        'scenario_aware': True,
        'anonymization_method': 'one_way_hash'
    }
    
    metadata_path = f'models/{bank_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📋 Metadata saved: {metadata_path}")
    
    return model, metadata

def main():
    print("🏦 SCENARIO-AWARE BANK MODEL TRAINING")
    print("=====================================")
    print("Training models with one-way hash anonymization and scenario awareness")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Train all bank models
    for bank_name in ['bank_A', 'bank_B', 'bank_C']:
        try:
            model, metadata = train_scenario_aware_model(bank_name)
            print(f"✅ {bank_name} training completed successfully")
        except Exception as e:
            print(f"❌ {bank_name} training failed: {e}")
    
    print("\n🎉 ALL MODELS TRAINED WITH SCENARIO AWARENESS!")
    print("🔐 Models now support one-way hash anonymization")
    print("🎯 Banks can determine their own knowledge scenarios")
    print("⚖️ Confidence weights applied based on customer knowledge")

if __name__ == '__main__':
    main()
