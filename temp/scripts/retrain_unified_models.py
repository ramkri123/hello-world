#!/usr/bin/env python3
"""
Retrain models with unified privacy-preserving architecture
All banks trained on same 35 anonymous features with specialized training data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import json
import sys

# Add the consortium path for imports
sys.path.append(os.path.join('..', 'src', 'consortium'))

def generate_anonymized_training_data(n_samples=5000):
    """Generate synthetic training data with 35 anonymous behavioral features for all banks"""
    
    print(f"🔄 Generating {n_samples} realistic training samples...")
    data = []
    
    # Account anonymization mapping (simulate 1:1 mapping)
    known_sender_accounts = set([f"ANON_SENDER_{i}" for i in range(100)])
    known_receiver_accounts = set([f"ANON_RECEIVER_{i}" for i in range(50)])
    
    for i in range(n_samples):
        # Generate random transaction parameters with more realistic fraud rate
        is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud rate (more realistic)
        
        # Generate all 35 anonymous features for each sample
        features = []
        
        if is_fraud:
            # === FRAUD TRANSACTION FEATURES (with more variation) ===
            
            # Features 0-9: Transaction behavioral features
            features.extend([
                np.random.normal(0.75, 0.15),    # amount_risk_score (high but variable)
                np.random.normal(0.70, 0.20),    # velocity_score
                np.random.normal(0.20, 0.15),    # sender_balance_ratio (low balance but variable)
                np.random.choice([0, 1], p=[0.4, 0.6]),  # unusual_amount_flag
                np.random.choice([0, 1], p=[0.6, 0.4]),  # weekend_flag
                np.random.choice([0, 1], p=[0.7, 0.3]),  # holiday_flag
                np.random.normal(0.65, 0.20),    # after_hours_risk
                np.random.normal(0.70, 0.15),    # rush_transaction
                np.random.choice([0, 1], p=[0.3, 0.7]),  # friday_afternoon
                np.random.normal(0.80, 0.15),    # off_hours_risk
            ])
            
            # Features 10-19: Email behavioral features (more realistic variation)
            features.extend([
                np.random.normal(0.60, 0.25),    # authority_score (CEO claims but variable)
                np.random.normal(0.40, 0.20),    # title_formality
                np.random.normal(0.65, 0.20),    # executive_bypass
                np.random.normal(0.70, 0.20),    # urgency_score
                np.random.normal(0.60, 0.25),    # deadline_pressure
                np.random.normal(0.55, 0.20),    # time_sensitivity
                np.random.normal(0.65, 0.20),    # manipulation_score
                np.random.normal(0.50, 0.25),    # confidentiality_claims
                np.random.normal(0.70, 0.15),    # business_justification
                np.random.normal(0.45, 0.20),    # acquisition_language
            ])
            
            # Features 20-29: Email communication features
            features.extend([
                np.random.normal(0.25, 0.15),    # email_formality (low for fraud but variable)
                np.random.normal(0.50, 0.20),    # complexity_score
                np.random.normal(0.65, 0.20),    # spelling_errors
                np.random.normal(0.60, 0.25),    # grammar_issues
                np.random.normal(0.75, 0.15),    # external_indicators
                np.random.normal(0.55, 0.20),    # reply_mismatch
                np.random.normal(0.70, 0.20),    # domain_spoofing
                np.random.normal(0.60, 0.20),    # header_anomalies
                np.random.normal(0.30, 0.20),    # sender_reputation (low but variable)
                np.random.normal(0.65, 0.15),    # communication_pattern
            ])
            
            # Features 30-34: Account anonymization features
            # Add more complexity in account matching
            sender_exact = np.random.choice([0, 1], p=[0.5, 0.5])  # 50/50 for fraud
            receiver_exact = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% known receivers (new accounts common in fraud)
            
            features.extend([
                float(sender_exact),                           # sender_exact_match
                1.0 - float(sender_exact),                     # sender_wildcard_match  
                float(receiver_exact),                         # receiver_exact_match
                1.0 - float(receiver_exact),                   # receiver_wildcard_match
                np.random.normal(0.8 if receiver_exact == 0 else 0.2, 0.15)  # new_account_risk
            ])
            
        else:
            # === LEGITIMATE TRANSACTION FEATURES (overlapping with fraud to make it harder) ===
            
            # Features 0-9: Transaction behavioral features
            features.extend([
                np.random.normal(0.25, 0.20),    # amount_risk_score (low but some overlap)
                np.random.normal(0.20, 0.15),    # velocity_score
                np.random.normal(0.75, 0.20),    # sender_balance_ratio (good balance but variable)
                np.random.choice([0, 1], p=[0.85, 0.15]),  # unusual_amount_flag
                np.random.choice([0, 1], p=[0.8, 0.2]),  # weekend_flag
                np.random.choice([0, 1], p=[0.9, 0.1]),  # holiday_flag
                np.random.normal(0.15, 0.15),    # after_hours_risk
                np.random.normal(0.10, 0.10),    # rush_transaction
                np.random.choice([0, 1], p=[0.7, 0.3]),  # friday_afternoon
                np.random.normal(0.20, 0.15),    # off_hours_risk
            ])
            
            # Features 10-19: Email behavioral features (some overlap with fraud)
            features.extend([
                np.random.normal(0.15, 0.15),    # authority_score (usually low but some legitimate authority)
                np.random.normal(0.80, 0.15),    # title_formality
                np.random.normal(0.10, 0.10),    # executive_bypass
                np.random.normal(0.20, 0.20),    # urgency_score (some legitimate urgency)
                np.random.normal(0.15, 0.15),    # deadline_pressure
                np.random.normal(0.25, 0.20),    # time_sensitivity
                np.random.normal(0.10, 0.10),    # manipulation_score
                np.random.normal(0.20, 0.15),    # confidentiality_claims
                np.random.normal(0.30, 0.20),    # business_justification
                np.random.normal(0.15, 0.15),    # acquisition_language
            ])
            
            # Features 20-29: Email communication features
            features.extend([
                np.random.normal(0.80, 0.15),    # email_formality (high for legitimate)
                np.random.normal(0.75, 0.15),    # complexity_score
                np.random.normal(0.10, 0.10),    # spelling_errors
                np.random.normal(0.10, 0.10),    # grammar_issues
                np.random.normal(0.15, 0.15),    # external_indicators
                np.random.normal(0.10, 0.10),    # reply_mismatch
                np.random.normal(0.05, 0.05),    # domain_spoofing
                np.random.normal(0.15, 0.10),    # header_anomalies
                np.random.normal(0.85, 0.10),    # sender_reputation
                np.random.normal(0.80, 0.15),    # communication_pattern
            ])
            
            # Features 30-34: Account anonymization features
            sender_exact = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% known senders
            receiver_exact = np.random.choice([0, 1], p=[0.4, 0.6])  # 60% known receivers
            
            features.extend([
                float(sender_exact),                           # sender_exact_match
                1.0 - float(sender_exact),                     # sender_wildcard_match
                float(receiver_exact),                         # receiver_exact_match  
                1.0 - float(receiver_exact),                   # receiver_wildcard_match
                np.random.normal(0.15 if receiver_exact == 0 else 0.05, 0.10)  # new_account_risk
            ])
        
        # Clip all features to [0, 1] range and add noise for realism
        features = [max(0.0, min(1.0, f + np.random.normal(0, 0.05))) for f in features]
        
        # Add sample to dataset
        sample = features + [is_fraud]
        data.append(sample)
    
    # Create DataFrame
    feature_names = [
        # Transaction features (0-9)
        'amount_risk_score', 'velocity_score', 'sender_balance_ratio', 'unusual_amount_flag',
        'weekend_flag', 'holiday_flag', 'after_hours_risk', 'rush_transaction', 
        'friday_afternoon', 'off_hours_risk',
        
        # Email behavioral features (10-19)  
        'authority_score', 'title_formality', 'executive_bypass', 'urgency_score',
        'deadline_pressure', 'time_sensitivity', 'manipulation_score', 'confidentiality_claims',
        'business_justification', 'acquisition_language',
        
        # Email communication features (20-29)
        'email_formality', 'complexity_score', 'spelling_errors', 'grammar_issues',
        'external_indicators', 'reply_mismatch', 'domain_spoofing', 'header_anomalies',
        'sender_reputation', 'communication_pattern',
        
        # Account anonymization features (30-34)
        'sender_exact_match', 'sender_wildcard_match', 'receiver_exact_match',
        'receiver_wildcard_match', 'new_account_risk',
        
        # Target
        'is_fraud'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    
    print(f"✅ Generated unified dataset: {len(df)} samples")
    print(f"   Fraud samples: {df['is_fraud'].sum()}")
    print(f"   Legitimate samples: {len(df) - df['is_fraud'].sum()}")
    print(f"   Features generated: {len(feature_names) - 1}")
    
    return df

def create_specialized_training_sets(df):
    """Create specialized training sets for each bank using same features but different emphasis"""
    
    # All banks get same features but different training data emphasis
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    X = df[feature_cols]
    y = df['is_fraud']
    
    training_sets = {}
    
    # Bank A: Wire Transfer Fraud Specialist
    # Emphasize transaction and timing features in training data
    print(f"   🏦 Bank A - Wire Transfer Specialist (features 0-9 emphasis)")
    # Weight samples based on transaction risk patterns
    wire_fraud_mask = (
        (df['amount_risk_score'] > 0.5) | 
        (df['rush_transaction'] > 0.5) |
        (df['friday_afternoon'] == 1)
    )
    # Oversample wire transfer fraud patterns
    wire_samples = df[wire_fraud_mask & (df['is_fraud'] == 1)]
    regular_samples = df[~wire_fraud_mask]
    bank_A_data = pd.concat([regular_samples, wire_samples, wire_samples], ignore_index=True)  # Double wire fraud samples
    
    training_sets['bank_A'] = {
        'X': bank_A_data[feature_cols],
        'y': bank_A_data['is_fraud'],
        'specialty': 'wire_transfer_specialist',
        'focus_features': 'Transaction timing and amount patterns (features 0-9)'
    }
    
    # Bank B: Identity Verification Specialist  
    # Emphasize email behavioral and account features
    print(f"   🔍 Bank B - Identity Specialist (features 10-19, 30-34 emphasis)")
    identity_fraud_mask = (
        (df['authority_score'] > 0.4) |
        (df['executive_bypass'] > 0.5) |
        (df['new_account_risk'] > 0.6)
    )
    identity_samples = df[identity_fraud_mask & (df['is_fraud'] == 1)]
    regular_samples = df[~identity_fraud_mask]
    bank_B_data = pd.concat([regular_samples, identity_samples, identity_samples], ignore_index=True)
    
    training_sets['bank_B'] = {
        'X': bank_B_data[feature_cols],
        'y': bank_B_data['is_fraud'],
        'specialty': 'identity_receiver_specialist',
        'focus_features': 'Authority claims and account verification (features 10-19, 30-34)'
    }
    
    # Bank C: Network Analysis Specialist
    # Emphasize communication patterns and network features
    print(f"   🌐 Bank C - Network Specialist (features 20-29 emphasis)")
    network_fraud_mask = (
        (df['external_indicators'] > 0.5) |
        (df['domain_spoofing'] > 0.5) |
        (df['sender_reputation'] < 0.5)
    )
    network_samples = df[network_fraud_mask & (df['is_fraud'] == 1)]
    regular_samples = df[~network_fraud_mask]
    bank_C_data = pd.concat([regular_samples, network_samples, network_samples], ignore_index=True)
    
    training_sets['bank_C'] = {
        'X': bank_C_data[feature_cols],
        'y': bank_C_data['is_fraud'],
        'specialty': 'network_account_specialist',
        'focus_features': 'Communication patterns and network analysis (features 20-29)'
    }
    
    return training_sets

def train_unified_models(training_sets):
    """Train all banks on same 35 features with specialized training data"""
    
    print(f"\n🏦 Training Unified Bank Models (Same 35 Features)...")
    
    models = {}
    
    for bank_id, training_data in training_sets.items():
        print(f"   🏦 {bank_id.upper()} - {training_data['specialty']}")
        print(f"      Focus: {training_data['focus_features']}")
        
        X = training_data['X']
        y = training_data['y']
        
        # Split training data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model on all 35 features with more realistic parameters
        model = RandomForestClassifier(
            n_estimators=50,           # Reduced to prevent overfitting
            random_state=42,
            class_weight='balanced',
            max_depth=8,               # Reduced depth to prevent overfitting
            min_samples_split=10,      # Increased to prevent overfitting
            min_samples_leaf=5,        # Increased to prevent overfitting
            max_features='sqrt',       # Use subset of features
            bootstrap=True
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        score = model.score(X_test, y_test)
        print(f"      Accuracy: {score:.3f}")
        print(f"      Training samples: {len(X_train)}")
        print(f"      Features: {len(X.columns)} (same for all banks)")
        
        models[bank_id] = {
            'model': model,
            'feature_range': (0, 35),  # All banks use all 35 features
            'specialty': training_data['specialty'],
            'accuracy': score,
            'feature_count': len(X.columns)
        }
    
    return models

def save_unified_models(models):
    """Save the unified models and metadata"""
    
    print(f"\n💾 Saving Unified Models...")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    for bank_id, model_info in models.items():
        # Save model
        model_path = f'../models/{bank_id}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_info['model'], f)
        
        # Save metadata
        metadata = {
            'bank_id': bank_id,
            'specialty': model_info['specialty'],
            'feature_range': model_info['feature_range'],
            'accuracy': model_info['accuracy'],
            'model_type': 'unified_privacy_preserving_random_forest',
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': model_info['feature_count']
        }
        
        metadata_path = f'../models/{bank_id}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ {bank_id.upper()}: {model_path}")
        print(f"      Specialty: {model_info['specialty']}")
        print(f"      Features: {model_info['feature_count']} (unified)")
        print(f"      Accuracy: {model_info['accuracy']:.3f}")

def main():
    print("🚀 UNIFIED PRIVACY-PRESERVING MODEL TRAINING")
    print("=" * 50)
    print("🔐 All banks trained on same 35 anonymous features")
    print("🎯 Specialization through training data emphasis")
    print("🔗 Account anonymization with 1:1 mapping")
    print()
    
    # Generate training data with more samples for better generalization
    df = generate_anonymized_training_data(n_samples=20000)
    
    # Create specialized training sets
    training_sets = create_specialized_training_sets(df)
    
    # Train models
    models = train_unified_models(training_sets)
    
    # Save models
    save_unified_models(models)
    
    print(f"\n✅ UNIFIED MODEL TRAINING COMPLETE")
    print("=" * 50)
    print("🏦 All banks now trained on identical 35 anonymous features")
    print("🔐 Privacy-preserving consortium ready for deployment")
    print("🎯 Each bank specialized through training data patterns")

if __name__ == "__main__":
    main()
