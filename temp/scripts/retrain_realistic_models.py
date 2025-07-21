#!/usr/bin/env python3
"""
Retrain models with realistic fraud detection challenges
Introduces noise, edge cases, and harder classification problems
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import json
import sys

# Add the consortium path for imports
sys.path.append(os.path.join('..', 'src', 'consortium'))

def generate_realistic_training_data(n_samples=20000):
    """Generate realistic training data with noise, edge cases, and ambiguous examples"""
    
    print(f"🔄 Generating {n_samples} realistic training samples with noise...")
    data = []
    
    for i in range(n_samples):
        # Determine if fraud with realistic distribution
        is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud rate (realistic)
        
        features = []
        
        if is_fraud:
            # === FRAUD SAMPLES WITH REALISTIC VARIATIONS ===
            
            # Not all fraud has high risk scores - sophisticated fraud can look normal
            fraud_sophistication = np.random.uniform(0, 1)
            
            if fraud_sophistication > 0.7:  # 30% sophisticated fraud (harder to detect)
                # Sophisticated fraud - lower risk scores, more subtle
                features.extend([
                    np.random.uniform(0.3, 0.7),    # amount_risk_score (moderate)
                    np.random.uniform(0.2, 0.6),    # velocity_score  
                    np.random.uniform(0.4, 0.8),    # sender_balance_ratio (looks normal)
                    np.random.uniform(0.2, 0.6),    # unusual_amount_flag
                    np.random.choice([0, 1], p=[0.7, 0.3]),  # weekend_flag
                    np.random.choice([0, 1], p=[0.8, 0.2]),  # holiday_flag
                    np.random.uniform(0.1, 0.5),    # after_hours_risk
                    np.random.uniform(0.2, 0.6),    # rush_transaction
                    np.random.choice([0, 1], p=[0.6, 0.4]),  # friday_afternoon
                    np.random.uniform(0.1, 0.5),    # off_hours_risk
                ])
                
                # Sophisticated email features (subtle fraud indicators)
                features.extend([
                    np.random.uniform(0.2, 0.6),    # authority_score (moderate CEO claims)
                    np.random.uniform(0.4, 0.8),    # title_formality (looks professional)
                    np.random.uniform(0.1, 0.5),    # executive_bypass
                    np.random.uniform(0.3, 0.7),    # urgency_score (moderate urgency)
                    np.random.uniform(0.1, 0.5),    # deadline_pressure
                    np.random.uniform(0.2, 0.6),    # time_sensitivity
                    np.random.uniform(0.1, 0.5),    # manipulation_score (subtle)
                    np.random.uniform(0.1, 0.4),    # confidentiality_claims
                    np.random.uniform(0.3, 0.7),    # business_justification (looks legit)
                    np.random.uniform(0.1, 0.4),    # acquisition_language
                ])
                
                # Well-crafted communication (hard to detect)
                features.extend([
                    np.random.uniform(0.5, 0.9),    # email_formality (professional)
                    np.random.uniform(0.4, 0.8),    # complexity_score
                    np.random.uniform(0.0, 0.3),    # spelling_errors (low)
                    np.random.uniform(0.0, 0.3),    # grammar_issues (low)
                    np.random.uniform(0.2, 0.6),    # external_indicators
                    np.random.uniform(0.1, 0.4),    # reply_mismatch
                    np.random.uniform(0.1, 0.4),    # domain_spoofing (subtle)
                    np.random.uniform(0.1, 0.4),    # header_anomalies
                    np.random.uniform(0.3, 0.7),    # sender_reputation (moderate)
                    np.random.uniform(0.4, 0.8),    # communication_pattern
                ])
                
            else:  # 70% obvious fraud (easier to detect)
                # Obvious fraud patterns
                features.extend([
                    np.random.uniform(0.6, 1.0),    # amount_risk_score (high)
                    np.random.uniform(0.5, 1.0),    # velocity_score
                    np.random.uniform(0.0, 0.4),    # sender_balance_ratio (low balance)
                    np.random.uniform(0.7, 1.0),    # unusual_amount_flag
                    np.random.choice([0, 1], p=[0.3, 0.7]),  # weekend_flag
                    np.random.choice([0, 1], p=[0.2, 0.8]),  # holiday_flag
                    np.random.uniform(0.6, 1.0),    # after_hours_risk
                    np.random.uniform(0.5, 1.0),    # rush_transaction
                    np.random.choice([0, 1], p=[0.4, 0.6]),  # friday_afternoon
                    np.random.uniform(0.7, 1.0),    # off_hours_risk
                ])
                
                # Obvious fraud email features
                features.extend([
                    np.random.uniform(0.5, 1.0),    # authority_score (high CEO claims)
                    np.random.uniform(0.2, 0.6),    # title_formality (unprofessional)
                    np.random.uniform(0.4, 1.0),    # executive_bypass
                    np.random.uniform(0.6, 1.0),    # urgency_score (high urgency)
                    np.random.uniform(0.5, 1.0),    # deadline_pressure
                    np.random.uniform(0.6, 1.0),    # time_sensitivity
                    np.random.uniform(0.4, 1.0),    # manipulation_score
                    np.random.uniform(0.3, 0.8),    # confidentiality_claims
                    np.random.uniform(0.5, 1.0),    # business_justification
                    np.random.uniform(0.2, 0.7),    # acquisition_language
                ])
                
                # Poor communication quality
                features.extend([
                    np.random.uniform(0.0, 0.5),    # email_formality (low)
                    np.random.uniform(0.2, 0.6),    # complexity_score
                    np.random.uniform(0.3, 1.0),    # spelling_errors (high)
                    np.random.uniform(0.3, 1.0),    # grammar_issues (high)
                    np.random.uniform(0.5, 1.0),    # external_indicators
                    np.random.uniform(0.4, 1.0),    # reply_mismatch
                    np.random.uniform(0.4, 1.0),    # domain_spoofing
                    np.random.uniform(0.4, 1.0),    # header_anomalies
                    np.random.uniform(0.0, 0.5),    # sender_reputation (low)
                    np.random.uniform(0.0, 0.5),    # communication_pattern
                ])
            
            # Account features for fraud (with noise)
            sender_exact = np.random.choice([0, 1], p=[0.5, 0.5])  # 50/50 known/unknown
            receiver_exact = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% known receivers (suspicious)
            
            features.extend([
                float(sender_exact),
                1.0 - float(sender_exact),
                float(receiver_exact),
                1.0 - float(receiver_exact),
                np.random.uniform(0.3, 1.0) if receiver_exact == 0 else np.random.uniform(0.0, 0.5)
            ])
            
        else:
            # === LEGITIMATE SAMPLES WITH REALISTIC VARIATIONS ===
            
            # Some legitimate transactions can look suspicious (false positives)
            legitimate_complexity = np.random.uniform(0, 1)
            
            if legitimate_complexity > 0.8:  # 20% suspicious-looking legitimate transactions
                # Legitimate but suspicious-looking (creates classification challenge)
                features.extend([
                    np.random.uniform(0.3, 0.8),    # amount_risk_score (can be high for large legit transfers)
                    np.random.uniform(0.2, 0.7),    # velocity_score
                    np.random.uniform(0.3, 1.0),    # sender_balance_ratio
                    np.random.uniform(0.2, 0.7),    # unusual_amount_flag
                    np.random.choice([0, 1], p=[0.6, 0.4]),  # weekend_flag (legit weekend transfers)
                    np.random.choice([0, 1], p=[0.8, 0.2]),  # holiday_flag
                    np.random.uniform(0.1, 0.6),    # after_hours_risk (urgent legit business)
                    np.random.uniform(0.1, 0.6),    # rush_transaction (legitimate urgency)
                    np.random.choice([0, 1], p=[0.6, 0.4]),  # friday_afternoon
                    np.random.uniform(0.1, 0.5),    # off_hours_risk
                ])
                
                # Legitimate but urgent business communication
                features.extend([
                    np.random.uniform(0.0, 0.3),    # authority_score (low CEO claims)
                    np.random.uniform(0.5, 1.0),    # title_formality (professional)
                    np.random.uniform(0.0, 0.2),    # executive_bypass
                    np.random.uniform(0.2, 0.6),    # urgency_score (moderate urgency is normal)
                    np.random.uniform(0.0, 0.4),    # deadline_pressure
                    np.random.uniform(0.1, 0.5),    # time_sensitivity (business deadlines)
                    np.random.uniform(0.0, 0.2),    # manipulation_score
                    np.random.uniform(0.0, 0.3),    # confidentiality_claims
                    np.random.uniform(0.2, 0.6),    # business_justification (normal business)
                    np.random.uniform(0.0, 0.3),    # acquisition_language
                ])
                
                # Professional communication with minor issues
                features.extend([
                    np.random.uniform(0.6, 1.0),    # email_formality
                    np.random.uniform(0.5, 1.0),    # complexity_score
                    np.random.uniform(0.0, 0.3),    # spelling_errors (professionals make mistakes)
                    np.random.uniform(0.0, 0.3),    # grammar_issues
                    np.random.uniform(0.0, 0.4),    # external_indicators
                    np.random.uniform(0.0, 0.3),    # reply_mismatch
                    np.random.uniform(0.0, 0.2),    # domain_spoofing
                    np.random.uniform(0.0, 0.3),    # header_anomalies
                    np.random.uniform(0.6, 1.0),    # sender_reputation
                    np.random.uniform(0.6, 1.0),    # communication_pattern
                ])
                
            else:  # 80% clearly legitimate transactions
                # Obviously legitimate patterns
                features.extend([
                    np.random.uniform(0.0, 0.4),    # amount_risk_score
                    np.random.uniform(0.0, 0.3),    # velocity_score
                    np.random.uniform(0.5, 1.0),    # sender_balance_ratio
                    np.random.uniform(0.0, 0.3),    # unusual_amount_flag
                    np.random.choice([0, 1], p=[0.8, 0.2]),  # weekend_flag
                    np.random.choice([0, 1], p=[0.9, 0.1]),  # holiday_flag
                    np.random.uniform(0.0, 0.3),    # after_hours_risk
                    np.random.uniform(0.0, 0.3),    # rush_transaction
                    np.random.choice([0, 1], p=[0.7, 0.3]),  # friday_afternoon
                    np.random.uniform(0.0, 0.3),    # off_hours_risk
                ])
                
                # Professional email features
                features.extend([
                    np.random.uniform(0.0, 0.2),    # authority_score
                    np.random.uniform(0.7, 1.0),    # title_formality
                    np.random.uniform(0.0, 0.2),    # executive_bypass
                    np.random.uniform(0.0, 0.3),    # urgency_score
                    np.random.uniform(0.0, 0.2),    # deadline_pressure
                    np.random.uniform(0.0, 0.3),    # time_sensitivity
                    np.random.uniform(0.0, 0.2),    # manipulation_score
                    np.random.uniform(0.0, 0.2),    # confidentiality_claims
                    np.random.uniform(0.0, 0.4),    # business_justification
                    np.random.uniform(0.0, 0.2),    # acquisition_language
                ])
                
                # High quality communication
                features.extend([
                    np.random.uniform(0.7, 1.0),    # email_formality
                    np.random.uniform(0.6, 1.0),    # complexity_score
                    np.random.uniform(0.0, 0.2),    # spelling_errors
                    np.random.uniform(0.0, 0.2),    # grammar_issues
                    np.random.uniform(0.0, 0.3),    # external_indicators
                    np.random.uniform(0.0, 0.2),    # reply_mismatch
                    np.random.uniform(0.0, 0.1),    # domain_spoofing
                    np.random.uniform(0.0, 0.2),    # header_anomalies
                    np.random.uniform(0.7, 1.0),    # sender_reputation
                    np.random.uniform(0.7, 1.0),    # communication_pattern
                ])
            
            # Account features for legitimate (with noise)
            sender_exact = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% known senders
            receiver_exact = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% known receivers
            
            features.extend([
                float(sender_exact),
                1.0 - float(sender_exact),
                float(receiver_exact),
                1.0 - float(receiver_exact),
                np.random.uniform(0.0, 0.3) if receiver_exact == 0 else np.random.uniform(0.0, 0.2)
            ])
        
        # Add noise to all features (real-world measurement noise)
        noise_level = 0.05  # 5% noise
        features = [max(0.0, min(1.0, f + np.random.normal(0, noise_level))) for f in features]
        
        # Add sample to dataset
        sample = features + [is_fraud]
        data.append(sample)
    
    # Create DataFrame with proper feature names
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
    print(f"   Fraud samples: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"   Legitimate samples: {len(df) - df['is_fraud'].sum()}")
    print(f"   Features generated: {len(feature_names) - 1}")
    
    return df

def create_specialized_training_sets(df):
    """Create specialized training sets with different emphasis and noise levels"""
    
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    training_sets = {}
    
    # Bank A: Wire Transfer Specialist
    print(f"   🏦 Bank A - Wire Transfer Specialist")
    wire_emphasis_mask = (
        (df['amount_risk_score'] > 0.4) | 
        (df['rush_transaction'] > 0.4) |
        (df['friday_afternoon'] == 1) |
        (df['unusual_amount_flag'] > 0.4)
    )
    
    # Create training set with wire transfer emphasis
    emphasized_samples = df[wire_emphasis_mask]
    regular_samples = df[~wire_emphasis_mask].sample(n=len(df)//2, random_state=42)  # Subsample
    bank_A_data = pd.concat([regular_samples, emphasized_samples], ignore_index=True)
    
    training_sets['bank_A'] = {
        'X': bank_A_data[feature_cols],
        'y': bank_A_data['is_fraud'],
        'specialty': 'wire_transfer_specialist',
        'focus': 'Transaction patterns and timing analysis'
    }
    
    # Bank B: Identity Verification Specialist  
    print(f"   🔍 Bank B - Identity Specialist")
    identity_emphasis_mask = (
        (df['authority_score'] > 0.3) |
        (df['executive_bypass'] > 0.3) |
        (df['new_account_risk'] > 0.5) |
        (df['sender_exact_match'] == 0)
    )
    
    emphasized_samples = df[identity_emphasis_mask]
    regular_samples = df[~identity_emphasis_mask].sample(n=len(df)//2, random_state=43)
    bank_B_data = pd.concat([regular_samples, emphasized_samples], ignore_index=True)
    
    training_sets['bank_B'] = {
        'X': bank_B_data[feature_cols],
        'y': bank_B_data['is_fraud'],
        'specialty': 'identity_receiver_specialist',
        'focus': 'Authority claims and account verification'
    }
    
    # Bank C: Network Analysis Specialist
    print(f"   🌐 Bank C - Network Specialist")
    network_emphasis_mask = (
        (df['external_indicators'] > 0.4) |
        (df['domain_spoofing'] > 0.3) |
        (df['sender_reputation'] < 0.6) |
        (df['communication_pattern'] < 0.6)
    )
    
    emphasized_samples = df[network_emphasis_mask]
    regular_samples = df[~network_emphasis_mask].sample(n=len(df)//2, random_state=44)
    bank_C_data = pd.concat([regular_samples, emphasized_samples], ignore_index=True)
    
    training_sets['bank_C'] = {
        'X': bank_C_data[feature_cols],
        'y': bank_C_data['is_fraud'],
        'specialty': 'network_account_specialist',
        'focus': 'Communication patterns and network analysis'
    }
    
    return training_sets

def train_realistic_models(training_sets):
    """Train models with realistic parameters and cross-validation"""
    
    print(f"\n🏦 Training Realistic Bank Models...")
    
    models = {}
    
    for bank_id, training_data in training_sets.items():
        print(f"   🏦 {bank_id.upper()} - {training_data['specialty']}")
        print(f"      Focus: {training_data['focus']}")
        
        X = training_data['X']
        y = training_data['y']
        
        # Split training data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with realistic parameters (prevent overfitting)
        model = RandomForestClassifier(
            n_estimators=50,         # Fewer trees to prevent overfitting
            random_state=42,
            class_weight='balanced',
            max_depth=8,             # Limit depth
            min_samples_split=20,    # Require more samples to split
            min_samples_leaf=10,     # Require more samples in leaves
            max_features='sqrt',     # Use subset of features
            bootstrap=True,
            oob_score=True          # Out-of-bag score for validation
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate with multiple metrics
        test_score = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None
        
        print(f"      Test Accuracy: {test_score:.3f}")
        print(f"      CV F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"      OOB Score: {oob_score:.3f}" if oob_score else "      OOB Score: N/A")
        print(f"      Training samples: {len(X_train)}")
        print(f"      Fraud rate: {y_train.mean()*100:.1f}%")
        
        models[bank_id] = {
            'model': model,
            'feature_range': (0, 35),
            'specialty': training_data['specialty'],
            'accuracy': test_score,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'oob_score': oob_score,
            'feature_count': len(X.columns)
        }
    
    return models

def save_realistic_models(models):
    """Save the realistic models and detailed metadata"""
    
    print(f"\n💾 Saving Realistic Models...")
    
    os.makedirs('../models', exist_ok=True)
    
    for bank_id, model_info in models.items():
        # Save model
        model_path = f'../models/{bank_id}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_info['model'], f)
        
        # Save detailed metadata
        metadata = {
            'bank_id': bank_id,
            'specialty': model_info['specialty'],
            'feature_range': model_info['feature_range'],
            'test_accuracy': model_info['accuracy'],
            'cv_f1_mean': model_info['cv_f1_mean'],
            'cv_f1_std': model_info['cv_f1_std'],
            'oob_score': model_info['oob_score'],
            'model_type': 'realistic_privacy_preserving_random_forest',
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': model_info['feature_count'],
            'training_notes': 'Trained with noise, edge cases, and realistic fraud patterns'
        }
        
        metadata_path = f'../models/{bank_id}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ {bank_id.upper()}: {model_path}")
        print(f"      Specialty: {model_info['specialty']}")
        print(f"      Test Accuracy: {model_info['accuracy']:.3f}")
        print(f"      CV F1 Score: {model_info['cv_f1_mean']:.3f}")

def main():
    print("🚀 REALISTIC PRIVACY-PRESERVING MODEL TRAINING")
    print("=" * 55)
    print("🎯 Training with noise, edge cases, and realistic fraud patterns")
    print("🔐 All banks trained on same 35 anonymous features")
    print("📊 Preventing overfitting with proper validation")
    print()
    
    # Generate realistic training data with noise and edge cases
    df = generate_realistic_training_data(n_samples=50000)  # Larger, more realistic dataset
    
    # Create specialized training sets
    training_sets = create_specialized_training_sets(df)
    
    # Train models with realistic parameters
    models = train_realistic_models(training_sets)
    
    # Save models
    save_realistic_models(models)
    
    print(f"\n✅ REALISTIC MODEL TRAINING COMPLETE")
    print("=" * 55)
    print("🏦 All banks trained with realistic accuracy scores")
    print("🔐 Privacy-preserving consortium ready with proper validation")
    print("🎯 Models now handle edge cases and noisy data")

if __name__ == "__main__":
    main()
