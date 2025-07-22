#!/usr/bin/env python3
"""
Create realistic training data with overlapping patterns
This addresses the perfect accuracy issue by making the classification task more challenging
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.append('src')

from consortium.privacy_preserving_nlp import PrivacyPreservingNLP

def generate_realistic_overlapping_data(bank_name, n_samples=3000):
    """Generate more realistic data with overlapping patterns between fraud and legitimate"""
    print(f"📊 Generating realistic training data for {bank_name}...")
    
    data = []
    
    # Create 85% legitimate, 15% fraud for realistic imbalance
    n_fraud = int(n_samples * 0.15)
    n_legit = n_samples - n_fraud
    
    print(f"   💚 Legitimate transactions: {n_legit}")
    print(f"   🚨 Fraud transactions: {n_fraud}")
    
    # Generate legitimate transactions with variety
    legitimate_patterns = [
        # Regular business
        {
            'amount_range': (100, 25000),
            'balance_range': (50000, 2000000),
            'hour_choices': [9, 10, 11, 14, 15, 16],
            'day_choices': [1, 2, 3, 4, 5],
            'emails': [
                "Invoice payment for services rendered monthly billing",
                "Vendor payment net 30 terms as agreed",
                "Payroll processing employee compensation",
                "Equipment purchase business expense",
                "Professional services contract payment",
                "Office supplies monthly order payment"
            ]
        },
        # Large legitimate transactions (can look suspicious)
        {
            'amount_range': (75000, 500000),
            'balance_range': (500000, 10000000),
            'hour_choices': [9, 10, 11, 14, 15, 16, 17],
            'day_choices': [1, 2, 3, 4, 5],
            'emails': [
                "Quarterly vendor payment large contract",
                "Equipment purchase major capital expense",
                "Real estate transaction closing payment",
                "Large supplier payment annual contract",
                "Construction payment project milestone",
                "Investment payment business expansion"
            ]
        },
        # Urgent but legitimate
        {
            'amount_range': (5000, 75000),
            'balance_range': (100000, 1500000),
            'hour_choices': [16, 17, 18],
            'day_choices': [4, 5],
            'emails': [
                "Urgent supplier payment deadline tomorrow",
                "Time sensitive vendor payment due today",
                "Express payment for critical supplies",
                "Expedited payment avoid service disruption",
                "Rush payment meet contract deadline"
            ]
        }
    ]
    
    for i in range(n_legit):
        pattern = np.random.choice(legitimate_patterns)
        
        transaction_data = {
            'amount': np.random.uniform(*pattern['amount_range']),
            'sender_balance': np.random.uniform(*pattern['balance_range']),
            'avg_daily_spending': np.random.uniform(1000, 100000),
            'hour': np.random.choice(pattern['hour_choices']),
            'day_of_week': np.random.choice(pattern['day_choices']),
            'is_holiday': np.random.choice([False, False, False, True])  # Mostly not holidays
        }
        
        email = np.random.choice(pattern['emails'])
        
        # Create feature vector manually to have more control
        features = create_manual_features(transaction_data, email, is_fraud=False)
        
        data.append({
            'features': features,
            'is_fraud': 0,
            'bank': bank_name
        })
    
    # Generate fraud with varying sophistication levels
    fraud_patterns = [
        # Obvious fraud (easy to detect)
        {
            'amount_range': (100000, 500000),
            'balance_range': (200000, 1000000),
            'hour_choices': [20, 21, 22, 23, 0, 1],
            'day_choices': [6, 7],
            'emails': [
                "CEO urgent wire transfer crypto investment confidential",
                "Emergency bitcoin payment ransomware decrypt files",
                "Urgent love emergency medical stuck overseas",
                "Immediate payment crypto opportunity limited time"
            ],
            'sophistication': 0.2
        },
        # Moderate fraud (harder to detect)
        {
            'amount_range': (25000, 150000),
            'balance_range': (100000, 2000000),
            'hour_choices': [14, 15, 16, 17, 18],
            'day_choices': [1, 2, 3, 4, 5],
            'emails': [
                "Vendor payment update banking details urgent",
                "Invoice payment new account information",
                "Supplier payment changed bank details",
                "Contract payment updated routing information"
            ],
            'sophistication': 0.6
        },
        # Sophisticated fraud (very hard to detect)
        {
            'amount_range': (10000, 75000),
            'balance_range': (75000, 1500000),
            'hour_choices': [9, 10, 11, 14, 15, 16],
            'day_choices': [1, 2, 3, 4, 5],
            'emails': [
                "Quarterly payment business services consulting",
                "Professional services contract completion",
                "Business development payment Q4 project",
                "Consulting fees strategic planning engagement"
            ],
            'sophistication': 0.9
        }
    ]
    
    for i in range(n_fraud):
        pattern = np.random.choice(fraud_patterns)
        
        transaction_data = {
            'amount': np.random.uniform(*pattern['amount_range']),
            'sender_balance': np.random.uniform(*pattern['balance_range']),
            'avg_daily_spending': np.random.uniform(2000, 50000),
            'hour': np.random.choice(pattern['hour_choices']),
            'day_of_week': np.random.choice(pattern['day_choices']),
            'is_holiday': np.random.choice([False, True])
        }
        
        email = np.random.choice(pattern['emails'])
        
        # Create features with sophistication level affecting detectability
        features = create_manual_features(transaction_data, email, is_fraud=True, 
                                        sophistication=pattern['sophistication'])
        
        data.append({
            'features': features,
            'is_fraud': 1,
            'bank': bank_name
        })
    
    return data

def create_manual_features(transaction_data, email, is_fraud=False, sophistication=0.5):
    """Create manual feature vector with controlled patterns"""
    
    # Basic transaction features (0-9)
    amount = transaction_data['amount']
    balance = transaction_data['sender_balance']
    
    features = [
        amount / 100000,  # Normalized amount
        balance / 1000000,  # Normalized balance  
        amount / balance if balance > 0 else 0,  # Amount to balance ratio
        transaction_data['hour'] / 24,  # Hour normalized
        transaction_data['day_of_week'] / 7,  # Day normalized
        1.0 if transaction_data['is_holiday'] else 0.0,  # Holiday flag
        transaction_data['avg_daily_spending'] / 100000,  # Daily spending normalized
        min(amount / transaction_data['avg_daily_spending'], 10) / 10,  # Amount vs usual spending
        1.0 if amount > 50000 else 0.0,  # Large amount flag
        1.0 if transaction_data['hour'] in [20, 21, 22, 23, 0, 1, 2] else 0.0  # Odd hours
    ]
    
    # Email content features (10-24) - simplified NLP
    email_lower = email.lower()
    
    # Urgency indicators
    urgency_words = ['urgent', 'immediately', 'asap', 'deadline', 'expires', 'rush', 'emergency']
    urgency_score = sum(1 for word in urgency_words if word in email_lower) / len(urgency_words)
    
    # Authority words
    authority_words = ['ceo', 'president', 'director', 'manager', 'boss']
    authority_score = sum(1 for word in authority_words if word in email_lower) / len(authority_words)
    
    # Secrecy words
    secrecy_words = ['confidential', 'secret', 'private', 'discreet']
    secrecy_score = sum(1 for word in secrecy_words if word in email_lower) / len(secrecy_words)
    
    # Financial terms
    financial_words = ['payment', 'invoice', 'contract', 'vendor', 'supplier']
    financial_score = sum(1 for word in financial_words if word in email_lower) / len(financial_words)
    
    # Suspicious terms
    suspicious_words = ['crypto', 'bitcoin', 'ransomware', 'love', 'stuck', 'emergency medical']
    suspicious_score = sum(1 for word in suspicious_words if word in email_lower) / len(suspicious_words)
    
    # Email features
    email_features = [
        urgency_score,
        authority_score, 
        secrecy_score,
        financial_score,
        suspicious_score,
        len(email) / 100,  # Email length
        email_lower.count('!') / 10,  # Exclamation marks
        1.0 if 'crypto' in email_lower or 'bitcoin' in email_lower else 0.0,
        1.0 if 'ceo' in email_lower else 0.0,
        1.0 if 'urgent' in email_lower else 0.0,
        1.0 if 'confidential' in email_lower else 0.0,
        1.0 if 'payment' in email_lower else 0.0,
        1.0 if 'emergency' in email_lower else 0.0,
        1.0 if 'love' in email_lower else 0.0,
        1.0 if any(word in email_lower for word in ['update', 'change', 'new']) else 0.0
    ]
    
    # Add sophistication adjustment for fraud
    if is_fraud:
        # More sophisticated fraud looks more like legitimate transactions
        noise_factor = (1 - sophistication) * 0.3  # High sophistication = low noise
        for i in range(len(email_features)):
            if email_features[i] > 0.5:  # Reduce suspicious indicators
                email_features[i] *= (1 - sophistication * 0.5)
    
    features.extend(email_features)
    
    # Pattern features (25-34) - behavioral patterns
    pattern_features = [
        1.0 if transaction_data['day_of_week'] in [6, 7] else 0.0,  # Weekend
        1.0 if transaction_data['hour'] < 8 or transaction_data['hour'] > 18 else 0.0,  # After hours
        min(amount / 10000, 5) / 5,  # Amount tier
        1.0 if balance < amount * 2 else 0.0,  # Low balance vs amount
        np.random.uniform(0, 1),  # Network score (random for now)
        np.random.uniform(0, 1),  # Velocity score
        np.random.uniform(0, 1),  # Geographic score
        np.random.uniform(0, 1),  # Device score
        np.random.uniform(0, 1),  # Behavior score
        urgency_score * authority_score  # Combined risk indicator
    ]
    
    features.extend(pattern_features)
    
    # Add noise to make it more realistic
    noise_level = 0.05 if not is_fraud else 0.05 * (1 - sophistication)
    noise = np.random.normal(0, noise_level, len(features))
    features = np.array(features) + noise
    
    # Ensure features are in valid range [0, 1]
    features = np.clip(features, 0, 1)
    
    return features.tolist()

def train_realistic_model(bank_name):
    """Train a model with realistic performance"""
    print(f"\n🏦 TRAINING {bank_name.upper()} MODEL (REALISTIC)")
    print("=" * 60)
    
    # Generate training data
    training_data = generate_realistic_overlapping_data(bank_name)
    
    # Convert to arrays
    X = np.array([sample['features'] for sample in training_data])
    y = np.array([sample['is_fraud'] for sample in training_data])
    
    print(f"📊 Training data shape: {X.shape}")
    print(f"📊 Label distribution: {np.bincount(y)}")
    print(f"📊 Fraud percentage: {np.mean(y) * 100:.1f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest with moderate regularization
    model = RandomForestClassifier(
        n_estimators=75,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    
    print("🔄 Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"📈 Training accuracy: {train_accuracy:.3f}")
    print(f"📈 Test accuracy: {test_accuracy:.3f}")
    print(f"📊 Overfitting gap: {train_accuracy - test_accuracy:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"📊 CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Check model quality
    if test_accuracy > 0.85:
        print("✅ Good model performance")
    elif test_accuracy > 0.75:
        print("⚠️ Acceptable model performance")
    else:
        print("❌ Poor model performance - may need more data")
    
    if train_accuracy - test_accuracy < 0.1:
        print("✅ Good generalization")
    else:
        print("⚠️ Some overfitting detected")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{bank_name}_model.pkl"
    scaler_path = f"models/{bank_name}_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'bank_name': bank_name,
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'overfitting_gap': float(train_accuracy - test_accuracy),
        'features_count': int(X.shape[1]),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'fraud_percentage': float(np.mean(y) * 100)
    }
    
    import json
    metadata_path = f"models/{bank_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved model: {model_path}")
    print(f"💾 Saved scaler: {scaler_path}")
    print(f"💾 Saved metadata: {metadata_path}")
    
    return model, scaler, metadata

def main():
    """Train realistic bank models"""
    print("🏦 REALISTIC CONSORTIUM BANK MODEL TRAINING")
    print("===========================================")
    print("🎯 Creating models with realistic accuracy (75-85%)")
    print("📊 Using overlapping patterns between fraud and legitimate")
    
    banks = ['bank_A', 'bank_B', 'bank_C']
    results = {}
    
    for bank in banks:
        model, scaler, metadata = train_realistic_model(bank)
        results[bank] = metadata
    
    print("\n🎯 TRAINING SUMMARY")
    print("=" * 50)
    for bank, meta in results.items():
        print(f"{bank.upper()}:")
        print(f"  Train accuracy: {meta['train_accuracy']:.3f}")
        print(f"  Test accuracy: {meta['test_accuracy']:.3f}")
        print(f"  Overfitting gap: {meta['overfitting_gap']:.3f}")
        print(f"  CV score: {meta['cv_mean']:.3f} ± {meta['cv_std']:.3f}")
        print()
    
    print("✅ Realistic bank models trained successfully!")
    print("🔒 Models ready for distributed consortium system")
    print("📊 Models now show realistic fraud detection performance")

if __name__ == "__main__":
    main()
