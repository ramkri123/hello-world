#!/usr/bin/env python3
"""
Realistic Model Training with Proper Accuracy Scores
Generates challenging synthetic data with noise and edge cases
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os
import json
import sys

# Add the consortium path for imports
sys.path.append(os.path.join('..', 'src', 'consortium'))

def generate_realistic_training_data(n_samples=100000):
    """Generate realistic training data with noise, edge cases, and challenging examples"""
    
    print(f"🔄 Generating {n_samples} realistic training samples with noise...")
    data = []
    
    for i in range(n_samples):
        # Generate fraud vs legitimate with 0.5% fraud rate (realistic for banking)
        is_fraud = np.random.choice([0, 1], p=[0.995, 0.005])
        
        # Generate all 35 anonymous features for each sample
        features = []
        
        if is_fraud:
            # === FRAUD TRANSACTIONS ===
            # Add realistic variation - not all fraud looks the same
            fraud_intensity = np.random.uniform(0.3, 1.0)  # Some fraud is subtle
            
            # Features 0-9: Transaction behavioral features
            features.extend([
                np.random.beta(2, 1) * fraud_intensity,  # amount_risk_score - realistic distribution
                np.random.gamma(2, 0.3) * fraud_intensity,  # velocity_score
                max(0, 1 - np.random.exponential(0.5) * fraud_intensity),  # sender_balance_ratio
                np.random.choice([0, 1], p=[0.4, 0.6]) * fraud_intensity,  # unusual_amount_flag
                np.random.choice([0, 1], p=[0.7, 0.3]),  # weekend_flag
                np.random.choice([0, 1], p=[0.9, 0.1]),  # holiday_flag
                np.random.beta(3, 2) * fraud_intensity,  # after_hours_risk
                np.random.beta(2, 1) * fraud_intensity,  # rush_transaction
                np.random.choice([0, 1], p=[0.4, 0.6]),  # friday_afternoon
                np.random.beta(2, 2) * fraud_intensity + np.random.normal(0, 0.1),  # off_hours_risk + noise
            ])
            
            # Features 10-19: Email behavioral features
            # Add noise - even fraud emails vary significantly
            authority_base = np.random.beta(3, 1) * fraud_intensity
            features.extend([
                min(1.0, authority_base + np.random.normal(0, 0.15)),  # authority_score with noise
                max(0, 1 - np.random.beta(2, 1) * fraud_intensity),  # title_formality (inverse)
                np.random.beta(2, 2) * fraud_intensity + np.random.normal(0, 0.1),  # executive_bypass
                min(1.0, np.random.beta(2, 1) * fraud_intensity + np.random.normal(0, 0.2)),  # urgency_score
                np.random.beta(2, 3) * fraud_intensity,  # deadline_pressure
                np.random.gamma(1, 0.4) * fraud_intensity,  # time_sensitivity
                min(1.0, np.random.beta(2, 1) * fraud_intensity + np.random.normal(0, 0.15)),  # manipulation_score
                np.random.beta(1.5, 2) * fraud_intensity,  # confidentiality_claims
                np.random.beta(2, 2) * fraud_intensity,  # business_justification
                np.random.beta(1, 3) * fraud_intensity,  # acquisition_language
            ])
            
            # Features 20-29: Email communication features
            features.extend([
                max(0, 1 - np.random.beta(3, 1) * fraud_intensity),  # email_formality (inverse)
                np.random.beta(2, 3) + np.random.normal(0, 0.1),  # complexity_score
                np.random.beta(2, 3) * fraud_intensity + np.random.normal(0, 0.1),  # spelling_errors
                np.random.beta(1.5, 2) * fraud_intensity,  # grammar_issues
                np.random.beta(3, 2) * fraud_intensity,  # external_indicators
                np.random.beta(1, 4) * fraud_intensity,  # reply_mismatch
                np.random.beta(2, 5) * fraud_intensity,  # domain_spoofing
                np.random.beta(1, 3) * fraud_intensity,  # header_anomalies
                max(0, 1 - np.random.beta(2, 1) * fraud_intensity),  # sender_reputation (inverse)
                max(0, 1 - np.random.beta(1.5, 1) * fraud_intensity),  # communication_pattern (inverse)
            ])
            
            # Features 30-34: Account anonymization features
            # Realistic account matching - fraud often uses new accounts
            sender_exact = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% known senders (compromised accounts)
            receiver_exact = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% known receivers (new fraud accounts)
            
            features.extend([
                float(sender_exact),
                1.0 - float(sender_exact),
                float(receiver_exact),
                1.0 - float(receiver_exact),
                (1.0 - receiver_exact) * np.random.beta(2, 1) + np.random.normal(0, 0.05),  # new_account_risk
            ])
            
        else:
            # === LEGITIMATE TRANSACTIONS ===
            # Add realistic variation - legitimate transactions also vary
            legit_noise = np.random.uniform(0.0, 0.3)  # Some legitimate transactions look suspicious
            
            # Features 0-9: Transaction behavioral features
            features.extend([
                np.random.beta(1, 3) + legit_noise * 0.5,  # amount_risk_score
                np.random.beta(1, 4) + legit_noise * 0.3,  # velocity_score
                np.random.beta(3, 1) + np.random.normal(0, 0.1),  # sender_balance_ratio
                np.random.choice([0, 1], p=[0.9, 0.1]) * legit_noise,  # unusual_amount_flag
                np.random.choice([0, 1], p=[0.8, 0.2]),  # weekend_flag
                np.random.choice([0, 1], p=[0.95, 0.05]),  # holiday_flag
                np.random.beta(1, 4) + legit_noise * 0.4,  # after_hours_risk
                np.random.beta(1, 5) + legit_noise * 0.3,  # rush_transaction
                np.random.choice([0, 1], p=[0.7, 0.3]),  # friday_afternoon
                np.random.beta(1, 3) + legit_noise * 0.4,  # off_hours_risk
            ])
            
            # Features 10-19: Email behavioral features
            features.extend([
                np.random.beta(1, 5) + legit_noise * 0.3,  # authority_score
                np.random.beta(3, 1) + np.random.normal(0, 0.1),  # title_formality
                np.random.beta(1, 6) + legit_noise * 0.2,  # executive_bypass
                np.random.beta(1, 4) + legit_noise * 0.4,  # urgency_score
                np.random.beta(1, 5) + legit_noise * 0.2,  # deadline_pressure
                np.random.beta(1, 4) + legit_noise * 0.3,  # time_sensitivity
                np.random.beta(1, 6) + legit_noise * 0.2,  # manipulation_score
                np.random.beta(1, 5) + legit_noise * 0.2,  # confidentiality_claims
                np.random.beta(2, 3) + legit_noise * 0.3,  # business_justification
                np.random.beta(1, 4) + legit_noise * 0.2,  # acquisition_language
            ])
            
            # Features 20-29: Email communication features
            features.extend([
                np.random.beta(3, 1) + np.random.normal(0, 0.1),  # email_formality
                np.random.beta(2, 1) + np.random.normal(0, 0.15),  # complexity_score
                np.random.beta(1, 5) + legit_noise * 0.3,  # spelling_errors
                np.random.beta(1, 5) + legit_noise * 0.2,  # grammar_issues
                np.random.beta(1, 4) + legit_noise * 0.3,  # external_indicators
                np.random.beta(1, 6) + legit_noise * 0.2,  # reply_mismatch
                np.random.beta(1, 10) + legit_noise * 0.1,  # domain_spoofing
                np.random.beta(1, 5) + legit_noise * 0.2,  # header_anomalies
                np.random.beta(3, 1) + np.random.normal(0, 0.1),  # sender_reputation
                np.random.beta(3, 1) + np.random.normal(0, 0.1),  # communication_pattern
            ])
            
            # Features 30-34: Account anonymization features
            sender_exact = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% known senders
            receiver_exact = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% known receivers
            
            features.extend([
                float(sender_exact),
                1.0 - float(sender_exact),
                float(receiver_exact),
                1.0 - float(receiver_exact),
                (1.0 - receiver_exact) * np.random.beta(1, 3) + np.random.normal(0, 0.05),  # new_account_risk
            ])
        
        # Ensure all features are in [0,1] range and add sample to dataset
        features = [max(0, min(1, f)) for f in features]
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
    
    print(f"✅ Generated realistic dataset: {len(df)} samples")
    print(f"   Fraud samples: {df['is_fraud'].sum()}")
    print(f"   Legitimate samples: {len(df) - df['is_fraud'].sum()}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.1%}")
    print(f"   Features generated: {len(feature_names) - 1}")
    
    return df

def create_specialized_training_sets(df):
    """Create specialized training sets for each bank using same features but different emphasis"""
    
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    training_sets = {}
    
    # Bank A: Wire Transfer Fraud Specialist
    print(f"   🏦 Bank A - Wire Transfer Specialist")
    wire_fraud_mask = (
        (df['amount_risk_score'] > 0.4) | 
        (df['rush_transaction'] > 0.4) |
        (df['friday_afternoon'] == 1) |
        (df['velocity_score'] > 0.5)
    )
    # Moderate oversampling to handle class imbalance (0.5% fraud rate)
    wire_samples = df[wire_fraud_mask & (df['is_fraud'] == 1)]
    regular_samples = df[~wire_fraud_mask | (df['is_fraud'] == 0)]
    # Oversample fraud cases 3x to help with imbalanced learning
    bank_A_data = pd.concat([regular_samples, wire_samples, wire_samples, wire_samples], ignore_index=True)
    
    training_sets['bank_A'] = {
        'X': bank_A_data[feature_cols],
        'y': bank_A_data['is_fraud'],
        'specialty': 'wire_transfer_specialist',
        'focus': 'Transaction timing and amount patterns'
    }
    
    # Bank B: Identity Verification Specialist  
    print(f"   🔍 Bank B - Identity Specialist")
    identity_fraud_mask = (
        (df['authority_score'] > 0.3) |
        (df['executive_bypass'] > 0.3) |
        (df['new_account_risk'] > 0.5) |
        (df['manipulation_score'] > 0.4)
    )
    identity_samples = df[identity_fraud_mask & (df['is_fraud'] == 1)]
    regular_samples = df[~identity_fraud_mask | (df['is_fraud'] == 0)]
    # Oversample fraud cases 3x to help with imbalanced learning
    bank_B_data = pd.concat([regular_samples, identity_samples, identity_samples, identity_samples], ignore_index=True)
    
    training_sets['bank_B'] = {
        'X': bank_B_data[feature_cols],
        'y': bank_B_data['is_fraud'],
        'specialty': 'identity_receiver_specialist',
        'focus': 'Authority claims and account verification'
    }
    
    # Bank C: Network Analysis Specialist
    print(f"   🌐 Bank C - Network Specialist")
    network_fraud_mask = (
        (df['external_indicators'] > 0.4) |
        (df['domain_spoofing'] > 0.3) |
        (df['sender_reputation'] < 0.6) |
        (df['communication_pattern'] < 0.7)
    )
    network_samples = df[network_fraud_mask & (df['is_fraud'] == 1)]
    regular_samples = df[~network_fraud_mask | (df['is_fraud'] == 0)]
    # Oversample fraud cases 3x to help with imbalanced learning
    bank_C_data = pd.concat([regular_samples, network_samples, network_samples, network_samples], ignore_index=True)
    
    training_sets['bank_C'] = {
        'X': bank_C_data[feature_cols],
        'y': bank_C_data['is_fraud'],
        'specialty': 'network_account_specialist',
        'focus': 'Communication patterns and network analysis'
    }
    
    return training_sets

def train_realistic_models(training_sets):
    """Train models with realistic parameters to avoid overfitting"""
    
    print(f"\n🏦 Training Realistic Bank Models...")
    
    models = {}
    
    for bank_id, training_data in training_sets.items():
        print(f"   🏦 {bank_id.upper()} - {training_data['specialty']}")
        
        X = training_data['X']
        y = training_data['y']
        
        # Split training data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y  # Larger test set
        )
        
        # Train model with realistic parameters to prevent overfitting
        model = RandomForestClassifier(
            n_estimators=50,  # Fewer trees
            random_state=42,
            class_weight='balanced',
            max_depth=8,  # Limit depth
            min_samples_split=20,  # Require more samples to split
            min_samples_leaf=10,   # Require more samples in leaves
            max_features='sqrt',   # Limit features per tree
            bootstrap=True,
            oob_score=True
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_accuracy = model.score(X_test, y_test)
        train_accuracy = model.score(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # AUC score
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"      Train Accuracy: {train_accuracy:.3f}")
        print(f"      Test Accuracy: {test_accuracy:.3f}")
        print(f"      CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        print(f"      AUC Score: {auc_score:.3f}")
        print(f"      OOB Score: {model.oob_score_:.3f}")
        print(f"      Training samples: {len(X_train)}")
        print(f"      Test samples: {len(X_test)}")
        
        # Check for overfitting
        overfitting = train_accuracy - test_accuracy
        if overfitting > 0.1:
            print(f"      ⚠️  Overfitting detected: {overfitting:.3f}")
        else:
            print(f"      ✅ Good generalization: {overfitting:.3f}")
        
        models[bank_id] = {
            'model': model,
            'feature_range': (0, 35),
            'specialty': training_data['specialty'],
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'cv_score': cv_scores.mean(),
            'auc_score': auc_score,
            'oob_score': model.oob_score_,
            'feature_count': len(X.columns)
        }
    
    return models

def save_realistic_models(models):
    """Save the realistic models and metadata"""
    
    print(f"\n💾 Saving Realistic Models...")
    
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
            'test_accuracy': model_info['test_accuracy'],
            'train_accuracy': model_info['train_accuracy'],
            'cv_score': model_info['cv_score'],
            'auc_score': model_info['auc_score'],
            'oob_score': model_info['oob_score'],
            'model_type': 'realistic_privacy_preserving_random_forest',
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': model_info['feature_count']
        }
        
        metadata_path = f'../models/{bank_id}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ {bank_id.upper()}: {model_path}")
        print(f"      Test Accuracy: {model_info['test_accuracy']:.3f}")
        print(f"      AUC Score: {model_info['auc_score']:.3f}")

def main():
    print("🚀 REALISTIC PRIVACY-PRESERVING MODEL TRAINING")
    print("=" * 50)
    print("🎯 Generating challenging synthetic data with noise")
    print("🔐 All banks trained on same 35 anonymous features")
    print("📊 Realistic accuracy scores with proper validation")
    print()
    
    # Generate training data with realistic fraud rate
    df = generate_realistic_training_data(n_samples=100000)
    
    # Create specialized training sets
    training_sets = create_specialized_training_sets(df)
    
    # Train models
    models = train_realistic_models(training_sets)
    
    # Save models
    save_realistic_models(models)
    
    print(f"\n✅ REALISTIC MODEL TRAINING COMPLETE")
    print("=" * 50)
    print("🏦 All banks trained on identical 35 anonymous features")
    print("📊 Realistic accuracy scores prevent overfitting")
    print("🔐 Privacy-preserving consortium ready with proper validation")

if __name__ == "__main__":
    main()
