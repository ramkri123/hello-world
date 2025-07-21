#!/usr/bin/env python3
"""
Retrain models with privacy-preserving anonymous features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import json
from privacy_preserving_nlp import PrivacyPreservingNLP

def generate_synthetic_training_data(n_samples=5000):
    """Generate synthetic training data with anonymous behavioral features"""
    
    nlp = PrivacyPreservingNLP()
    data = []
    
    print(f"🔄 Generating {n_samples} synthetic training samples...")
    
    for i in range(n_samples):
        # Generate random transaction parameters
        is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% fraud rate
        
        if is_fraud:
            # Fraud transaction characteristics
            amount = np.random.uniform(50000, 500000)  # Large amounts
            sender_balance = np.random.uniform(amount, amount * 10)
            avg_daily_spending = np.random.uniform(5000, 50000)
            hour = np.random.choice([16, 17, 18, 19], p=[0.4, 0.3, 0.2, 0.1])  # Late day
            day_of_week = np.random.choice([4, 5, 6], p=[0.6, 0.3, 0.1])  # Friday/weekend
            is_holiday = np.random.choice([0, 1], p=[0.8, 0.2])
            
            # Fraud email characteristics
            authority_score = np.random.uniform(0.7, 1.0)  # High authority claims
            urgency_score = np.random.uniform(0.6, 1.0)    # High urgency
            manipulation_score = np.random.uniform(0.4, 0.9)  # Social engineering
            business_score = np.random.uniform(0.5, 0.8)   # Business justification
            
            # Fraud account characteristics
            sender_age = np.random.uniform(1.0, 10.0)
            receiver_age = np.random.uniform(0.001, 0.1)   # Very new accounts
            sender_risk = np.random.uniform(0.0, 0.3)
            receiver_risk = np.random.uniform(0.5, 1.0)    # High risk receivers
            
        else:
            # Legitimate transaction characteristics
            amount = np.random.uniform(100, 50000)  # Smaller amounts
            sender_balance = np.random.uniform(amount * 2, amount * 50)
            avg_daily_spending = np.random.uniform(1000, 20000)
            hour = np.random.choice(range(9, 17))  # Business hours
            day_of_week = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            is_holiday = 0
            
            # Normal email characteristics
            authority_score = np.random.uniform(0.0, 0.3)  # Low authority claims
            urgency_score = np.random.uniform(0.0, 0.4)    # Low urgency
            manipulation_score = np.random.uniform(0.0, 0.2)  # No manipulation
            business_score = np.random.uniform(0.1, 0.4)   # Normal business
            
            # Normal account characteristics
            sender_age = np.random.uniform(0.5, 15.0)
            receiver_age = np.random.uniform(0.1, 20.0)    # Established accounts
            sender_risk = np.random.uniform(0.0, 0.2)
            receiver_risk = np.random.uniform(0.0, 0.3)    # Low risk receivers
        
        # Create transaction data structure
        transaction_data = {
            'amount': amount,
            'sender_balance': sender_balance,
            'avg_daily_spending': avg_daily_spending,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_holiday': is_holiday
        }
        
        sender_data = {
            'account_age_years': sender_age,
            'risk_score': sender_risk,
            'transaction_count': np.random.randint(100, 5000),
            'business_type': 'business' if np.random.random() > 0.3 else 'individual',
            'geographic_risk': np.random.uniform(0.0, 0.3),
            'bank': f'bank_{np.random.choice(["A", "B", "C"])}'
        }
        
        receiver_data = {
            'account_age_years': receiver_age,
            'risk_score': receiver_risk,
            'verification_score': np.random.uniform(0.2, 0.9),
            'bank': f'bank_{np.random.choice(["A", "B", "C"])}'
        }
        
        # Simulate email content based on fraud characteristics
        email_content = ""
        if is_fraud and authority_score > 0.7:
            email_content = "CEO urgent strategic acquisition confidential time sensitive"
        elif urgency_score > 0.5:
            email_content = "urgent payment deadline ASAP"
        else:
            email_content = "payment invoice transaction"
        
        # Extract anonymous features
        features = nlp.convert_to_anonymous_features(
            transaction_data, email_content, sender_data, receiver_data
        )
        
        # Add to dataset
        sample = features + [is_fraud]
        data.append(sample)
        
        if (i + 1) % 1000 == 0:
            print(f"   Generated {i + 1}/{n_samples} samples")
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(len(data[0]) - 1)] + ['is_fraud']
    df = pd.DataFrame(data, columns=feature_names)
    
    print(f"✅ Generated dataset: {len(df)} samples")
    print(f"   Fraud samples: {df['is_fraud'].sum()}")
    print(f"   Legitimate samples: {len(df) - df['is_fraud'].sum()}")
    
    return df

def train_specialized_models(df):
    """Train specialized models for each bank based on feature ranges"""
    
    print(f"\n🏦 Training Specialized Bank Models...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    X = df[feature_cols]
    y = df['is_fraud']
    
    models = {}
    
    # Bank A: Sender/Transaction Specialist (Features 0-14)
    print(f"   🏦 Bank A - Sender/Transaction Specialist (Features 0-14)")
    X_bank_A = X.iloc[:, 0:15]  # Transaction and sender features
    X_train_A, X_test_A, y_train, y_test = train_test_split(X_bank_A, y, test_size=0.2, random_state=42, stratify=y)
    
    model_A = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced',
        max_depth=10
    )
    model_A.fit(X_train_A, y_train)
    
    score_A = model_A.score(X_test_A, y_test)
    print(f"      Accuracy: {score_A:.3f}")
    
    models['bank_A'] = {
        'model': model_A,
        'feature_range': (0, 15),
        'specialty': 'sender_transaction_specialist',
        'accuracy': score_A
    }
    
    # Bank B: Identity/Receiver Specialist (Features 15-29)  
    print(f"   🔍 Bank B - Identity/Receiver Specialist (Features 15-29)")
    X_bank_B = X.iloc[:, 15:30]  # Email behavioral and early receiver features
    X_train_B, X_test_B, y_train, y_test = train_test_split(X_bank_B, y, test_size=0.2, random_state=42, stratify=y)
    
    model_B = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced',
        max_depth=10
    )
    model_B.fit(X_train_B, y_train)
    
    score_B = model_B.score(X_test_B, y_test)
    print(f"      Accuracy: {score_B:.3f}")
    
    models['bank_B'] = {
        'model': model_B,
        'feature_range': (15, 30),
        'specialty': 'identity_receiver_specialist',
        'accuracy': score_B
    }
    
    # Bank C: Network/Account Pattern Specialist (Features 30+)
    print(f"   🌐 Bank C - Network/Account Pattern Specialist (Features 30+)")
    X_bank_C = X.iloc[:, 30:]  # Account and network features
    X_train_C, X_test_C, y_train, y_test = train_test_split(X_bank_C, y, test_size=0.2, random_state=42, stratify=y)
    
    model_C = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced',
        max_depth=10
    )
    model_C.fit(X_train_C, y_train)
    
    score_C = model_C.score(X_test_C, y_test)
    print(f"      Accuracy: {score_C:.3f}")
    
    models['bank_C'] = {
        'model': model_C,
        'feature_range': (30, len(feature_cols)),
        'specialty': 'network_account_specialist',
        'accuracy': score_C
    }
    
    return models

def save_models(models):
    """Save the trained models and metadata"""
    
    print(f"\n💾 Saving Models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    for bank_id, model_info in models.items():
        # Save model
        model_path = f'models/{bank_id}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_info['model'], f)
        
        # Save metadata
        metadata = {
            'bank_id': bank_id,
            'specialty': model_info['specialty'],
            'feature_range': model_info['feature_range'],
            'accuracy': model_info['accuracy'],
            'model_type': 'privacy_preserving_random_forest',
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': model_info['feature_range'][1] - model_info['feature_range'][0]
        }
        
        metadata_path = f'models/{bank_id}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ {bank_id}: {model_path} (accuracy: {model_info['accuracy']:.3f})")
    
    print(f"✅ All models saved to models/ directory")

def test_models_with_bec_case(models):
    """Test the models with a BEC fraud case"""
    
    print(f"\n🧪 Testing Models with BEC Fraud Case...")
    
    # Create BEC test case
    nlp = PrivacyPreservingNLP()
    bec_email = """
    Hi John, this is CEO Sarah Wilson. We have an urgent strategic acquisition 
    that requires immediate wire transfer of $485,000 to Global Tech Solutions.
    This is confidential and time sensitive - must complete before Friday close.
    """
    
    transaction_data = {
        'amount': 485000,
        'sender_balance': 2300000,
        'avg_daily_spending': 50000,
        'hour': 16,
        'day_of_week': 4,
        'is_holiday': False
    }
    
    sender_data = {
        'account_age_years': 6.0,
        'risk_score': 0.05,
        'transaction_count': 2000,
        'business_type': 'business',
        'geographic_risk': 0.1,
        'bank': 'bank_A'
    }
    
    receiver_data = {
        'account_age_years': 0.008,  # 3 days
        'risk_score': 0.8,
        'verification_score': 0.2,
        'bank': 'bank_B'
    }
    
    # Extract features
    features = nlp.convert_to_anonymous_features(
        transaction_data, bec_email, sender_data, receiver_data
    )
    
    print(f"   📊 Generated {len(features)} features for BEC test case")
    
    # Test each bank's model
    bank_scores = {}
    for bank_id, model_info in models.items():
        start_idx, end_idx = model_info['feature_range']
        bank_features = features[start_idx:end_idx]
        
        if len(bank_features) > 0:
            # Reshape for prediction
            bank_features_array = np.array(bank_features).reshape(1, -1)
            
            # Get probability of fraud
            fraud_prob = model_info['model'].predict_proba(bank_features_array)[0][1]
            bank_scores[bank_id] = fraud_prob
            
            print(f"   🏦 {bank_id} ({model_info['specialty']}): {fraud_prob:.3f}")
        else:
            print(f"   ⚠️  {bank_id}: No features in range {start_idx}-{end_idx}")
    
    # Calculate consensus
    if bank_scores:
        consensus_score = sum(bank_scores.values()) / len(bank_scores)
        recommendation = "BLOCK" if consensus_score > 0.7 else "REVIEW" if consensus_score > 0.3 else "APPROVE"
        
        print(f"\n   🎯 Consensus Score: {consensus_score:.3f}")
        print(f"   📋 Recommendation: {recommendation}")
        
        return bank_scores, consensus_score
    
    return {}, 0.0

def main():
    """Main training function"""
    print("🚀 PRIVACY-PRESERVING MODEL TRAINING")
    print("=" * 50)
    
    # Generate training data
    print("\n📊 Step 1: Generate Synthetic Training Data")
    df = generate_synthetic_training_data(n_samples=5000)
    
    # Train specialized models
    print("\n🎯 Step 2: Train Specialized Bank Models")
    models = train_specialized_models(df)
    
    # Save models
    print("\n💾 Step 3: Save Models")
    save_models(models)
    
    # Test with BEC case
    print("\n🧪 Step 4: Test Models")
    bank_scores, consensus = test_models_with_bec_case(models)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   🏦 Trained {len(models)} specialized models")
    print(f"   🎯 BEC Test Consensus: {consensus:.3f}")
    print(f"   📂 Models saved to models/ directory")

if __name__ == "__main__":
    main()
