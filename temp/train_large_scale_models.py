#!/usr/bin/env python3
"""
High-Volume Scenario-Aware Bank Model Training
Creates realistic models with 50,000+ samples to prevent overfitting
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from consortium.privacy_preserving_nlp import PrivacyPreservingNLP
from consortium.account_anonymizer import AccountAnonymizer

def generate_large_training_data(num_samples=50000):
    """Generate large diverse training dataset with realistic fraud patterns"""
    print(f"🔄 Generating {num_samples:,} training samples...")
    
    samples = []
    labels = []
    
    np.random.seed(42)  # For reproducible results
    nlp = PrivacyPreservingNLP()
    
    # Email templates for variety
    fraud_templates = [
        "CEO urgent wire transfer confidential acquisition {amount} immediately",
        "Director needs {amount} crypto investment opportunity deadline expires",
        "Manager emergency payment {amount} secret project confidential",
        "Executive urgent transfer {amount} private deal time sensitive",
        "President requires {amount} wire transfer merger confidential ASAP",
        "VP needs immediate {amount} transfer acquisition confidential",
        "CFO urgent {amount} payment crypto deadline confidential",
        "Ransomware payment {amount} bitcoin decrypt files immediately",
        "Investment opportunity {amount} crypto limited time urgent",
        "Romance emergency {amount} medical treatment Dubai urgent",
        "Vendor payment update banking details {amount} urgent process",
        "Nigerian prince {amount} inheritance urgent transfer confidential"
    ]
    
    legit_templates = [
        "Monthly vendor payment {amount} invoice net 30 terms",
        "Quarterly office supplies {amount} regular business payment",
        "Annual service contract {amount} maintenance agreement",
        "Weekly payroll {amount} standard employee wages",
        "Bi-weekly utilities {amount} electricity water gas",
        "Insurance premium {amount} quarterly business coverage",
        "Equipment lease {amount} monthly rental payment",
        "Software subscription {amount} annual license renewal",
        "Professional services {amount} consulting agreement",
        "Rent payment {amount} monthly office space",
        "Tax payment {amount} quarterly estimated taxes",
        "Loan payment {amount} monthly business loan"
    ]
    
    for i in range(num_samples):
        # 20% fraud, 80% legitimate (more realistic banking ratio)
        is_fraud = np.random.random() < 0.20
        
        if is_fraud:
            # Generate fraud sample
            template = np.random.choice(fraud_templates)
            amount = np.random.choice([
                25000, 35000, 50000, 75000, 100000, 150000, 200000, 250000,
                300000, 485000, 500000, 750000, 1000000  # High-risk amounts
            ])
            
            # Add noise to amounts
            amount = amount * np.random.uniform(0.8, 1.2)
            
            email_content = template.format(amount=int(amount))
            
            # High-risk transaction features
            transaction_data = {
                'amount': amount,
                'sender_balance': amount * np.random.uniform(0.5, 2.0),
                'avg_daily_spending': amount * np.random.uniform(0.1, 0.8),
                'hour': np.random.choice([8, 9, 16, 17, 18, 19, 20]),  # Urgency hours
                'day_of_week': np.random.randint(1, 8),
                'is_holiday': np.random.random() < 0.1,
                'sender_age_days': np.random.uniform(30, 3650),
                'receiver_age_days': np.random.uniform(1, 365),  # Often new accounts
                'transaction_type': np.random.choice(['wire_transfer', 'ach']),
                'cross_border': np.random.random() < 0.4,  # Higher for fraud
                'sender_bank': np.random.choice(['A', 'B', 'C']),
                'receiver_bank': np.random.choice(['A', 'B', 'C'])
            }
            
        else:
            # Generate legitimate sample
            template = np.random.choice(legit_templates)
            amount = np.random.choice([
                500, 1000, 2500, 5000, 7500, 10000, 15000, 20000,
                25000, 30000, 40000, 50000, 75000  # Normal business amounts
            ])
            
            # Add noise to amounts
            amount = amount * np.random.uniform(0.7, 1.3)
            
            email_content = template.format(amount=int(amount))
            
            # Normal transaction features
            transaction_data = {
                'amount': amount,
                'sender_balance': amount * np.random.uniform(2.0, 10.0),
                'avg_daily_spending': amount * np.random.uniform(0.5, 2.0),
                'hour': np.random.choice([9, 10, 11, 14, 15, 16]),  # Business hours
                'day_of_week': np.random.randint(1, 6),  # Weekdays mostly
                'is_holiday': np.random.random() < 0.05,
                'sender_age_days': np.random.uniform(365, 7300),  # Established accounts
                'receiver_age_days': np.random.uniform(365, 7300),
                'transaction_type': np.random.choice(['ach', 'wire_transfer', 'check']),
                'cross_border': np.random.random() < 0.1,  # Lower for legit
                'sender_bank': np.random.choice(['A', 'B', 'C']),
                'receiver_bank': np.random.choice(['A', 'B', 'C'])
            }
        
        # Convert to features using NLP
        features = nlp.convert_to_anonymous_features(transaction_data, email_content)
        
        samples.append(features)
        labels.append(1 if is_fraud else 0)
        
        if (i + 1) % 5000 == 0:
            print(f"   Generated {i + 1:,} samples...")
    
    print(f"✅ Generated {len(samples):,} total samples")
    fraud_count = sum(labels)
    legit_count = len(labels) - fraud_count
    print(f"   📊 Fraud: {fraud_count:,} ({fraud_count/len(labels)*100:.1f}%)")
    print(f"   📊 Legitimate: {legit_count:,} ({legit_count/len(labels)*100:.1f}%)")
    
    return np.array(samples), np.array(labels)

def train_bank_with_scenario_awareness(bank_name, X, y, accounts):
    """Train a bank model with scenario awareness and regularization"""
    print(f"\n🏦 Training {bank_name} with scenario awareness...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest with regularization to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=100,           # Reduced from default to prevent overfitting
        max_depth=10,               # Limit tree depth
        min_samples_split=10,       # Require more samples to split
        min_samples_leaf=5,         # Require more samples in leaf nodes
        max_features='sqrt',        # Limit features per tree
        random_state=42,
        class_weight='balanced'     # Handle class imbalance
    )
    
    print(f"   🔄 Training with {X_train.shape[0]:,} samples...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"   📊 Training accuracy: {train_score:.3f}")
    print(f"   📊 Test accuracy: {test_score:.3f}")
    print(f"   📊 Overfitting check: {abs(train_score - test_score):.3f} (should be < 0.1)")
    
    # Test scenario awareness
    print(f"   🎭 Testing scenario awareness...")
    anonymizer = AccountAnonymizer()
    
    # Test different scenarios
    test_accounts = {
        'sender_anonymous': anonymizer.anonymize_account(accounts[0]),
        'receiver_anonymous': anonymizer.anonymize_account('OTHER_BANK_ACCOUNT')
    }
    
    scenario = anonymizer.bank_can_determine_ownership(accounts, test_accounts)
    confidence = anonymizer.get_scenario_confidence_weight(scenario)
    print(f"   🎯 Example scenario: {scenario} (confidence: {confidence:.2f})")
    
    # Save model and scaler
    model_path = f'models/{bank_name}_model.pkl'
    scaler_path = f'models/{bank_name}_scaler.pkl'
    metadata_path = f'models/{bank_name}_metadata.json'
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'bank_name': bank_name,
        'accounts': accounts,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'features': X.shape[1],
        'model_type': 'RandomForestClassifier',
        'regularization': 'max_depth=10, min_samples_split=10'
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Saved model: {model_path}")
    print(f"   ✅ Saved scaler: {scaler_path}")
    print(f"   ✅ Saved metadata: {metadata_path}")
    
    return model, scaler, test_score

def main():
    """Train all banks with large dataset and scenario awareness"""
    print("🚀 LARGE-SCALE SCENARIO-AWARE BANK TRAINING")
    print("=" * 60)
    
    # Generate large training dataset
    X, y = generate_large_training_data(num_samples=50000)
    
    # Bank account configurations
    bank_configs = {
        'bank_A': ['ACCA12345', 'ACCA67890', 'ACCA11111', 'ACCA22222', 'ACCA33333'],
        'bank_B': ['ACCB67890', 'ACCB12345', 'ACCB22222', 'ACCB44444', 'ACCB55555'],
        'bank_C': ['ACCC99999', 'ACCC12345', 'ACCC55555', 'ACCC66666', 'ACCC77777']
    }
    
    # Train each bank
    results = {}
    for bank_name, accounts in bank_configs.items():
        model, scaler, test_score = train_bank_with_scenario_awareness(bank_name, X, y, accounts)
        results[bank_name] = test_score
    
    print(f"\n🎯 TRAINING SUMMARY")
    print("=" * 30)
    for bank_name, test_score in results.items():
        print(f"   {bank_name}: {test_score:.3f} test accuracy")
    
    avg_score = np.mean(list(results.values()))
    print(f"\n📊 Average test accuracy: {avg_score:.3f}")
    
    if avg_score > 0.95:
        print("⚠️  Still high accuracy - data may be too separable")
        print("   This is normal for clear fraud vs legitimate patterns")
    elif avg_score > 0.85:
        print("✅ Good accuracy with realistic overfitting control")
    else:
        print("⚠️  Low accuracy - may need more features or data")
    
    print("\n🎉 Training complete! Models ready for distributed consortium.")

if __name__ == "__main__":
    main()
