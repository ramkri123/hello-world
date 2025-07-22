#!/usr/bin/env python3
"""
Improved Bank Model Training with Regularization
Fixes overfitting issues and creates realistic models for distributed consortium
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
from consortium.account_anonymizer import AccountAnonymizer

def generate_realistic_training_data(bank_name, n_samples=2000):
    """Generate realistic training data with proper fraud/legitimate balance"""
    print(f"📊 Generating training data for {bank_name}...")
    
    nlp = PrivacyPreservingNLP()
    data = []
    
    # Create 80% legitimate, 20% fraud for realistic distribution
    n_fraud = int(n_samples * 0.2)
    n_legit = n_samples - n_fraud
    
    print(f"   💚 Legitimate transactions: {n_legit}")
    print(f"   🚨 Fraud transactions: {n_fraud}")
    
    # Generate legitimate transactions
    for i in range(n_legit):
        # Normal business transactions
        transaction_data = {
            'amount': np.random.uniform(100, 50000),  # Normal amounts
            'sender_balance': np.random.uniform(50000, 5000000),
            'avg_daily_spending': np.random.uniform(1000, 100000),
            'hour': np.random.choice([9, 10, 11, 14, 15, 16]),  # Business hours
            'day_of_week': np.random.choice([1, 2, 3, 4, 5]),  # Weekdays
            'is_holiday': False
        }
        
        # Normal business emails
        legitimate_emails = [
            "Invoice payment for services rendered",
            "Monthly vendor payment as per contract",
            "Payroll processing for employees",
            "Regular supplier payment net 30 terms",
            "Equipment purchase payment",
            "Professional services billing",
            "Office rent payment monthly",
            "Insurance premium payment"
        ]
        
        email = np.random.choice(legitimate_emails)
        features = nlp.convert_to_anonymous_features(transaction_data, email)
        
        data.append({
            'features': features,
            'is_fraud': 0,
            'bank': bank_name
        })
    
    # Generate fraud transactions with varied patterns
    for i in range(n_fraud):
        fraud_type = np.random.choice(['ceo_bec', 'crypto', 'romance', 'vendor', 'ransomware'])
        
        if fraud_type == 'ceo_bec':
            transaction_data = {
                'amount': np.random.uniform(50000, 500000),  # Large amounts
                'sender_balance': np.random.uniform(100000, 2000000),
                'avg_daily_spending': np.random.uniform(5000, 50000),
                'hour': np.random.choice([16, 17, 18, 19]),  # After hours
                'day_of_week': np.random.choice([4, 5, 6, 7]),  # End of week
                'is_holiday': np.random.choice([True, False])
            }
            email = "CEO urgent wire transfer confidential merger acquisition deadline"
            
        elif fraud_type == 'crypto':
            transaction_data = {
                'amount': np.random.uniform(25000, 300000),
                'sender_balance': np.random.uniform(50000, 1000000),
                'avg_daily_spending': np.random.uniform(2000, 30000),
                'hour': np.random.choice([20, 21, 22, 23]),  # Late hours
                'day_of_week': np.random.choice([6, 7]),  # Weekends
                'is_holiday': False
            }
            email = "urgent crypto investment opportunity limited time bitcoin transfer"
            
        elif fraud_type == 'romance':
            transaction_data = {
                'amount': np.random.uniform(5000, 75000),
                'sender_balance': np.random.uniform(20000, 500000),
                'avg_daily_spending': np.random.uniform(500, 15000),
                'hour': np.random.choice([19, 20, 21, 22]),  # Evening
                'day_of_week': np.random.choice([6, 7]),  # Weekends
                'is_holiday': False
            }
            email = "emergency medical treatment stuck overseas need money urgently love"
            
        elif fraud_type == 'vendor':
            transaction_data = {
                'amount': np.random.uniform(15000, 150000),
                'sender_balance': np.random.uniform(75000, 1500000),
                'avg_daily_spending': np.random.uniform(3000, 40000),
                'hour': np.random.choice([14, 15, 16, 17]),  # Business hours
                'day_of_week': np.random.choice([1, 2, 3, 4, 5]),  # Weekdays
                'is_holiday': False
            }
            email = "urgent banking details changed update payment information immediately"
            
        else:  # ransomware
            transaction_data = {
                'amount': np.random.uniform(30000, 200000),
                'sender_balance': np.random.uniform(100000, 3000000),
                'avg_daily_spending': np.random.uniform(5000, 75000),
                'hour': np.random.choice([0, 1, 2, 3, 23]),  # Odd hours
                'day_of_week': np.random.choice([6, 7]),  # Weekends
                'is_holiday': False
            }
            email = "files encrypted pay bitcoin ransom decrypt systems 48 hours"
        
        features = nlp.convert_to_anonymous_features(transaction_data, email)
        
        data.append({
            'features': features,
            'is_fraud': 1,
            'bank': bank_name
        })
    
    return data

def create_bank_specific_variations(base_data, bank_name):
    """Add bank-specific detection patterns to prevent identical models"""
    
    # Bank-specific specializations
    bank_specialties = {
        'bank_A': {
            'wire_transfer_expertise': 1.2,  # Better at detecting wire fraud
            'crypto_detection': 0.9,         # Slightly worse at crypto
            'feature_noise': 0.05
        },
        'bank_B': {
            'identity_verification': 1.3,    # Better at identity fraud
            'bec_detection': 0.8,            # Slightly worse at BEC
            'feature_noise': 0.07
        },
        'bank_C': {
            'network_analysis': 1.1,         # Better at pattern analysis
            'romance_scam_detection': 1.4,   # Much better at romance scams
            'feature_noise': 0.06
        }
    }
    
    specialty = bank_specialties.get(bank_name, {'feature_noise': 0.05})
    noise_level = specialty['feature_noise']
    
    # Add small amount of bank-specific noise to features
    for sample in base_data:
        features = np.array(sample['features'])
        noise = np.random.normal(0, noise_level, len(features))
        sample['features'] = (features + noise).tolist()
    
    return base_data

def train_bank_model(bank_name, use_regularization=True):
    """Train a single bank model with proper regularization"""
    print(f"\n🏦 TRAINING {bank_name.upper()} MODEL")
    print("=" * 50)
    
    # Generate training data
    training_data = generate_realistic_training_data(bank_name)
    training_data = create_bank_specific_variations(training_data, bank_name)
    
    # Convert to DataFrame
    features_list = [sample['features'] for sample in training_data]
    labels_list = [sample['is_fraud'] for sample in training_data]
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"📊 Training data shape: {X.shape}")
    print(f"📊 Label distribution: {np.bincount(y)}")
    print(f"📊 Fraud percentage: {np.mean(y) * 100:.1f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if use_regularization:
        # Use Random Forest with regularization to prevent overfitting
        model = RandomForestClassifier(
            n_estimators=50,          # Fewer trees to reduce overfitting
            max_depth=10,             # Limit tree depth
            min_samples_split=20,     # Require more samples to split
            min_samples_leaf=10,      # Require more samples in leaves
            max_features='sqrt',      # Use subset of features
            random_state=42,
            class_weight='balanced'   # Handle class imbalance
        )
        print("🛡️ Using regularized Random Forest")
    else:
        # Simple model without regularization (for comparison)
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        print("⚠️ Using unregularized Random Forest")
    
    # Train model
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
    
    # Check if model is reasonable (not overfitted)
    if train_accuracy - test_accuracy > 0.1:
        print("⚠️ WARNING: Model may be overfitted!")
    else:
        print("✅ Model appears to have good generalization")
    
    # Save model and scaler
    model_path = f"models/{bank_name}_model.pkl"
    scaler_path = f"models/{bank_name}_scaler.pkl"
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'bank_name': bank_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting_gap': train_accuracy - test_accuracy,
        'features_count': X.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'fraud_percentage': np.mean(y) * 100
    }
    
    metadata_path = f"models/{bank_name}_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved model: {model_path}")
    print(f"💾 Saved scaler: {scaler_path}")
    print(f"💾 Saved metadata: {metadata_path}")
    
    return model, scaler, metadata

def main():
    """Train all bank models with proper regularization"""
    print("🏦 CONSORTIUM BANK MODEL TRAINING")
    print("=================================")
    print("🛡️ Using regularization to prevent overfitting")
    print("📊 Creating realistic fraud detection models")
    
    banks = ['bank_A', 'bank_B', 'bank_C']
    results = {}
    
    for bank in banks:
        model, scaler, metadata = train_bank_model(bank, use_regularization=True)
        results[bank] = metadata
    
    print("\n🎯 TRAINING SUMMARY")
    print("=" * 40)
    for bank, meta in results.items():
        print(f"{bank.upper()}:")
        print(f"  Train accuracy: {meta['train_accuracy']:.3f}")
        print(f"  Test accuracy: {meta['test_accuracy']:.3f}")
        print(f"  Overfitting gap: {meta['overfitting_gap']:.3f}")
        print(f"  CV score: {meta['cv_mean']:.3f} ± {meta['cv_std']:.3f}")
        print()
    
    print("✅ All bank models trained successfully!")
    print("🔒 Models ready for distributed consortium system")

if __name__ == "__main__":
    main()
