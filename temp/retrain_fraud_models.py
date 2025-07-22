#!/usr/bin/env python3
"""
Retrain Fraud Detection Models with Better Discrimination
Creates models that properly distinguish between obvious fraud and legitimate transactions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
import os
from datetime import datetime

def create_enhanced_fraud_dataset():
    """Create a training dataset with clear fraud vs legitimate patterns"""
    np.random.seed(42)
    
    # Generate more samples for better model training
    n_samples = 5000
    n_fraud = 2000  # 40% fraud rate for training
    n_legit = n_samples - n_fraud
    
    print(f"🔄 Generating enhanced training dataset...")
    print(f"   📊 Total samples: {n_samples}")
    print(f"   🚨 Fraud samples: {n_fraud} (40%)")
    print(f"   ✅ Legitimate samples: {n_legit} (60%)")
    
    # Feature names for clarity
    feature_names = [
        'amount_ratio_to_balance', 'amount_ratio_to_daily', 'large_amount_flag', 
        'round_amount', 'business_hours', 'weekend_flag', 'holiday_flag', 
        'late_day_flag', 'friday_afternoon', 'off_hours',
        'authority_score', 'exec_bypass', 'authority_urgency', 'urgency_score',
        'timing_pressure', 'multiple_urgency', 'manipulation_score', 'confidentiality',
        'trust_exploitation', 'business_score', 'new_relationship', 'acquisition_language',
        'communication_score', 'grammar_issues', 'external_indicators', 'sender_age_norm',
        'sender_risk_score', 'transaction_frequency', 'business_type_risk', 'geographic_risk',
        'receiver_age_norm', 'receiver_risk_score', 'verification_score', 'cross_bank_flag',
        'amount_velocity'
    ]
    
    # Initialize arrays
    X = np.zeros((n_samples, 35))
    y = np.zeros(n_samples)
    
    # Generate LEGITIMATE transactions (labels = 0)
    for i in range(n_legit):
        # Legitimate patterns
        X[i, 0] = np.random.uniform(0.01, 0.3)    # Low amount ratio to balance
        X[i, 1] = np.random.uniform(0.1, 0.8)     # Normal ratio to daily spending
        X[i, 2] = np.random.choice([0, 1], p=[0.8, 0.2])  # Rarely large amounts
        X[i, 3] = np.random.choice([0, 1], p=[0.7, 0.3])  # Sometimes round amounts
        X[i, 4] = np.random.choice([0, 1], p=[0.3, 0.7])  # Usually business hours
        X[i, 5] = np.random.choice([0, 1], p=[0.8, 0.2])  # Rarely weekends
        X[i, 6] = np.random.choice([0, 1], p=[0.95, 0.05]) # Rarely holidays
        X[i, 7] = np.random.choice([0, 1], p=[0.7, 0.3])  # Sometimes late day
        X[i, 8] = np.random.choice([0, 1], p=[0.9, 0.1])  # Rarely Friday afternoon
        X[i, 9] = np.random.choice([0, 1], p=[0.8, 0.2])  # Rarely off hours
        
        # Email features - legitimate (low scores)
        X[i, 10] = np.random.uniform(0.0, 0.2)    # Low authority score
        X[i, 11] = 0.0                            # No exec bypass
        X[i, 12] = np.random.uniform(0.0, 0.25)   # Low authority urgency
        X[i, 13] = np.random.uniform(0.0, 0.3)    # Low urgency score
        X[i, 14] = np.random.uniform(0.0, 0.2)    # Low timing pressure
        X[i, 15] = np.random.uniform(0.0, 0.2)    # Low multiple urgency
        X[i, 16] = np.random.uniform(0.0, 0.15)   # Low manipulation score
        X[i, 17] = np.random.choice([0, 1], p=[0.9, 0.1])  # Rarely confidential
        X[i, 18] = np.random.choice([0, 1], p=[0.95, 0.05]) # Rarely trust exploitation
        
        # Business features - legitimate
        X[i, 19] = np.random.uniform(0.1, 0.6)    # Normal business score
        X[i, 20] = np.random.choice([0, 1], p=[0.8, 0.2])  # Sometimes new relationship
        X[i, 21] = np.random.choice([0, 1], p=[0.95, 0.05]) # Rarely acquisition language
        
        # Communication features - legitimate (good quality)
        X[i, 22] = np.random.uniform(0.0, 0.2)    # Low communication issues
        X[i, 23] = np.random.uniform(0.0, 0.1)    # Low grammar issues
        X[i, 24] = np.random.uniform(0.0, 0.3)    # Some external indicators
        
        # Account features - established accounts
        X[i, 25] = np.random.uniform(0.5, 1.0)    # Established sender account
        X[i, 26] = np.random.uniform(0.0, 0.2)    # Low sender risk
        X[i, 27] = np.random.uniform(0.3, 0.9)    # Good transaction frequency
        X[i, 28] = np.random.uniform(0.0, 0.3)    # Low business type risk
        X[i, 29] = np.random.uniform(0.0, 0.3)    # Low geographic risk
        X[i, 30] = np.random.uniform(0.4, 1.0)    # Established receiver account
        X[i, 31] = np.random.uniform(0.0, 0.4)    # Low receiver risk
        X[i, 32] = np.random.uniform(0.6, 1.0)    # Good verification score
        X[i, 33] = np.random.choice([0, 1], p=[0.7, 0.3])  # Sometimes cross bank
        X[i, 34] = np.random.uniform(0.0, 0.4)    # Normal velocity
        
        y[i] = 0  # Legitimate
    
    # Generate FRAUD transactions (labels = 1)
    for i in range(n_legit, n_samples):
        # Fraud patterns - much more obvious
        X[i, 0] = np.random.uniform(0.4, 1.0)     # High amount ratio to balance
        X[i, 1] = np.random.uniform(0.8, 5.0)     # Very high ratio to daily spending
        X[i, 2] = np.random.choice([0, 1], p=[0.2, 0.8])  # Usually large amounts
        X[i, 3] = np.random.choice([0, 1], p=[0.3, 0.7])  # Often round amounts
        X[i, 4] = np.random.choice([0, 1], p=[0.6, 0.4])  # Mixed business hours
        X[i, 5] = np.random.choice([0, 1], p=[0.4, 0.6])  # Often weekends
        X[i, 6] = np.random.choice([0, 1], p=[0.7, 0.3])  # Sometimes holidays
        X[i, 7] = np.random.choice([0, 1], p=[0.3, 0.7])  # Often late day
        X[i, 8] = np.random.choice([0, 1], p=[0.5, 0.5])  # Mixed Friday afternoon
        X[i, 9] = np.random.choice([0, 1], p=[0.4, 0.6])  # Often off hours
        
        # Email features - fraud (high scores)
        X[i, 10] = np.random.uniform(0.6, 1.0)    # High authority score
        X[i, 11] = np.random.choice([0, 1], p=[0.3, 0.7])  # Often exec bypass
        X[i, 12] = np.random.uniform(0.7, 1.0)    # High authority urgency
        X[i, 13] = np.random.uniform(0.6, 1.0)    # High urgency score
        X[i, 14] = np.random.uniform(0.5, 1.0)    # High timing pressure
        X[i, 15] = np.random.uniform(0.4, 1.0)    # High multiple urgency
        X[i, 16] = np.random.uniform(0.5, 1.0)    # High manipulation score
        X[i, 17] = np.random.choice([0, 1], p=[0.2, 0.8])  # Often confidential
        X[i, 18] = np.random.choice([0, 1], p=[0.3, 0.7])  # Often trust exploitation
        
        # Business features - fraud
        X[i, 19] = np.random.uniform(0.3, 0.9)    # Mixed business score
        X[i, 20] = np.random.choice([0, 1], p=[0.2, 0.8])  # Often new relationship
        X[i, 21] = np.random.choice([0, 1], p=[0.4, 0.6])  # Often acquisition language
        
        # Communication features - fraud (poor quality)
        X[i, 22] = np.random.uniform(0.4, 1.0)    # High communication issues
        X[i, 23] = np.random.uniform(0.3, 0.8)    # Grammar issues
        X[i, 24] = np.random.uniform(0.5, 1.0)    # High external indicators
        
        # Account features - suspicious accounts
        X[i, 25] = np.random.uniform(0.0, 0.4)    # New/young sender account
        X[i, 26] = np.random.uniform(0.3, 0.9)    # High sender risk
        X[i, 27] = np.random.uniform(0.0, 0.4)    # Low transaction frequency
        X[i, 28] = np.random.uniform(0.4, 0.9)    # High business type risk
        X[i, 29] = np.random.uniform(0.3, 0.9)    # High geographic risk
        X[i, 30] = np.random.uniform(0.0, 0.3)    # Very new receiver account
        X[i, 31] = np.random.uniform(0.5, 1.0)    # High receiver risk
        X[i, 32] = np.random.uniform(0.0, 0.4)    # Poor verification score
        X[i, 33] = np.random.choice([0, 1], p=[0.3, 0.7])  # Often cross bank
        X[i, 34] = np.random.uniform(0.6, 1.0)    # High velocity
        
        y[i] = 1  # Fraud
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    df['is_fraud'] = y
    
    print(f"✅ Dataset created with clear fraud discrimination patterns")
    return df, feature_names

def train_specialist_model(df, feature_names, bank_name, specialty, feature_range):
    """Train a specialist model for a specific bank"""
    print(f"\n🏦 Training {bank_name} - {specialty}")
    print(f"   📊 Feature range: {feature_range[0]}-{feature_range[1]}")
    
    # Extract features for this specialist bank
    start_idx, end_idx = feature_range
    X_specialist = df.iloc[:, start_idx:end_idx+1].values
    y = df['is_fraud'].values
    
    print(f"   📈 Using {X_specialist.shape[1]} specialized features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_specialist, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with settings optimized for fraud detection
    model = RandomForestClassifier(
        n_estimators=200,           # More trees for stability
        max_depth=15,              # Deeper trees for complex patterns
        min_samples_split=5,       # Allow smaller splits
        min_samples_leaf=2,        # Allow smaller leaves
        max_features='sqrt',       # Good balance for fraud detection
        class_weight='balanced',   # Handle class imbalance
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"   📊 Model Performance:")
    print(f"      Training Accuracy: {model.score(X_train, y_train):.3f}")
    print(f"      Test Accuracy: {model.score(X_test, y_test):.3f}")
    
    # Check fraud detection capability
    fraud_indices = y_test == 1
    legit_indices = y_test == 0
    
    if np.sum(fraud_indices) > 0:
        fraud_proba = y_pred_proba[fraud_indices, 1]  # Probability of fraud
        legit_proba = y_pred_proba[legit_indices, 1]  # Probability of fraud
        
        print(f"      Avg Fraud Score: {np.mean(fraud_proba):.3f} (should be >0.7)")
        print(f"      Avg Legit Score: {np.mean(legit_proba):.3f} (should be <0.4)")
        print(f"      Separation: {np.mean(fraud_proba) - np.mean(legit_proba):.3f}")
    
    # Save model
    model_path = f"models/{bank_name}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    specialist_features = feature_names[start_idx:end_idx+1]
    metadata = {
        "bank_name": bank_name,
        "specialty": specialty,
        "feature_range": feature_range,
        "feature_names": specialist_features,
        "model_type": "RandomForestClassifier",
        "training_samples": len(X_train),
        "test_accuracy": float(model.score(X_test, y_test)),
        "created_at": datetime.now().isoformat()
    }
    
    metadata_path = f"models/{bank_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Saved: {model_path}")
    print(f"   ✅ Saved: {metadata_path}")
    
    return model

def main():
    """Retrain all fraud detection models"""
    print("🚀 RETRAINING FRAUD DETECTION MODELS")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Generate enhanced training data
    df, feature_names = create_enhanced_fraud_dataset()
    
    # Train specialist models for each bank
    banks = [
        ("bank_A", "Wire Transfer Specialist", (0, 14)),
        ("bank_B", "Identity Verification Specialist", (15, 24)), 
        ("bank_C", "Network Analysis Specialist", (25, 34))
    ]
    
    models = {}
    for bank_name, specialty, feature_range in banks:
        model = train_specialist_model(df, feature_names, bank_name, specialty, feature_range)
        models[bank_name] = model
    
    # Create a test bank model as well
    print(f"\n🏦 Training test_bank - General Purpose Model")
    X_all = df.iloc[:, :-1].values  # All features except is_fraud
    y = df['is_fraud'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    
    test_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    test_model.fit(X_train, y_train)
    
    # Save test model
    with open("models/test_bank_model.pkl", 'wb') as f:
        pickle.dump(test_model, f)
    
    test_metadata = {
        "bank_name": "test_bank",
        "specialty": "general_purpose",
        "feature_range": [0, 34],
        "feature_names": feature_names,
        "model_type": "RandomForestClassifier",
        "training_samples": len(X_train),
        "test_accuracy": float(test_model.score(X_test, y_test)),
        "created_at": datetime.now().isoformat()
    }
    
    with open("models/test_bank_metadata.json", 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"   ✅ Saved: models/test_bank_model.pkl")
    print(f"   ✅ Saved: models/test_bank_metadata.json")
    
    print("\n🎯 MODEL RETRAINING COMPLETE!")
    print("=" * 50)
    print("✅ All models now have better fraud discrimination")
    print("✅ Fraud scenarios should score 70%+ (HIGH RISK)")
    print("✅ Legitimate transactions should score <40% (LOW RISK)")
    print("🚀 Ready to restart the consortium system!")

if __name__ == "__main__":
    main()
