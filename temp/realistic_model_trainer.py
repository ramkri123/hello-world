#!/usr/bin/env python3
"""
Realistic Fraud Detection Model Trainer
Generates large datasets with realistic 0.5% fraud rate and proper train/test splits
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
from typing import Tuple, Dict

class RealisticFraudDataGenerator:
    """Generate realistic fraud detection dataset"""
    
    def __init__(self, fraud_rate: float = 0.005):  # 0.5% fraud rate
        self.fraud_rate = fraud_rate
        
    def generate_transaction_features(self, n_samples: int, is_fraud: bool) -> np.ndarray:
        """Generate realistic transaction features"""
        features = np.zeros((n_samples, 35))
        
        if is_fraud:
            # Fraud patterns - more extreme values
            # Amount features (0-4) - fraud tends to be larger, rounder amounts
            features[:, 0] = np.random.beta(3, 1, n_samples)  # Higher amount ratios
            features[:, 1] = np.random.beta(4, 1, n_samples) * 0.8 + 0.2  # Above normal spending
            features[:, 2] = np.random.choice([0.8, 0.9, 1.0], n_samples, p=[0.3, 0.3, 0.4])  # Large amounts
            features[:, 3] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Round amounts more common
            features[:, 4] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Off business hours
            
            # Timing features (5-9) - fraud often off-hours
            features[:, 5] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # Weekend
            features[:, 6] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Holiday rare
            features[:, 7] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # Late day
            features[:, 8] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Friday afternoon
            features[:, 9] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Off hours
            
            # Email features (10-24) - fraud has strong email indicators
            features[:, 10] = np.random.beta(3, 2, n_samples) * 0.9 + 0.1  # Authority score
            features[:, 11] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # Executive bypass
            features[:, 12] = np.random.beta(3, 1, n_samples)  # Authority urgency
            features[:, 13] = np.random.beta(4, 1, n_samples)  # Urgency score
            features[:, 14] = np.random.beta(3, 1, n_samples)  # Timing pressure
            features[:, 15] = np.random.beta(2, 1, n_samples) * 0.8 + 0.2  # Multiple urgency
            features[:, 16] = np.random.beta(3, 2, n_samples)  # Manipulation score
            features[:, 17] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Confidentiality
            features[:, 18] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # Trust exploitation
            features[:, 19] = np.random.beta(2, 3, n_samples) * 0.6  # Business score (lower)
            features[:, 20] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # New relationship
            features[:, 21] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # Acquisition language
            features[:, 22] = np.random.beta(3, 1, n_samples)  # Communication anomalies
            features[:, 23] = np.random.beta(3, 2, n_samples)  # Grammar issues
            features[:, 24] = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])  # External indicators
            
            # Account features (25-34) - fraud accounts often newer, riskier
            features[:, 25] = np.random.beta(1, 4, n_samples) * 0.3  # Sender age (newer)
            features[:, 26] = np.random.beta(3, 1, n_samples) * 0.7 + 0.3  # Sender risk (higher)
            features[:, 27] = np.random.beta(1, 3, n_samples) * 0.4  # Transaction frequency (lower)
            features[:, 28] = np.random.choice([0.2, 0.8], n_samples, p=[0.6, 0.4])  # Business risk
            features[:, 29] = np.random.beta(3, 1, n_samples) * 0.8 + 0.2  # Geographic risk
            features[:, 30] = np.random.beta(1, 4, n_samples) * 0.1  # Receiver age (very new)
            features[:, 31] = np.random.beta(4, 1, n_samples) * 0.8 + 0.2  # Receiver risk (high)
            features[:, 32] = np.random.beta(1, 3, n_samples) * 0.3  # Verification score (low)
            features[:, 33] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Cross-border
            features[:, 34] = np.random.beta(2, 1, n_samples) * 0.8 + 0.2  # Velocity risk
            
        else:
            # Legitimate patterns - more normal distributions
            # Amount features (0-4) - legitimate tend to be normal amounts
            features[:, 0] = np.random.beta(2, 3, n_samples) * 0.6  # Lower amount ratios
            features[:, 1] = np.random.beta(2, 2, n_samples) * 0.5 + 0.1  # Normal spending patterns
            features[:, 2] = np.random.beta(1, 4, n_samples) * 0.7  # Smaller amounts mostly
            features[:, 3] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Round amounts less common
            features[:, 4] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Business hours
            
            # Timing features (5-9) - legitimate more during business hours
            features[:, 5] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Weekday
            features[:, 6] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # Holiday rare
            features[:, 7] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Normal hours
            features[:, 8] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Not Friday afternoon
            features[:, 9] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Business hours
            
            # Email features (10-24) - legitimate have normal email patterns
            features[:, 10] = np.random.beta(1, 3, n_samples) * 0.4  # Low authority score
            features[:, 11] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # No executive bypass
            features[:, 12] = np.random.beta(1, 3, n_samples) * 0.3  # Low authority urgency
            features[:, 13] = np.random.beta(1, 4, n_samples) * 0.4  # Low urgency
            features[:, 14] = np.random.beta(1, 4, n_samples) * 0.3  # Low timing pressure
            features[:, 15] = np.random.beta(1, 5, n_samples) * 0.2  # Low multiple urgency
            features[:, 16] = np.random.beta(1, 4, n_samples) * 0.3  # Low manipulation
            features[:, 17] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Low confidentiality
            features[:, 18] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # No trust exploitation
            features[:, 19] = np.random.beta(3, 2, n_samples) * 0.8  # Higher business score
            features[:, 20] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Established relationships
            features[:, 21] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # Normal business language
            features[:, 22] = np.random.beta(1, 4, n_samples) * 0.3  # Low communication anomalies
            features[:, 23] = np.random.beta(1, 5, n_samples) * 0.2  # Good grammar
            features[:, 24] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Internal communications
            
            # Account features (25-34) - legitimate accounts are established
            features[:, 25] = np.random.beta(2, 1, n_samples) * 0.8 + 0.2  # Sender age (older)
            features[:, 26] = np.random.beta(1, 4, n_samples) * 0.3  # Sender risk (lower)
            features[:, 27] = np.random.beta(3, 2, n_samples) * 0.8 + 0.2  # Transaction frequency (higher)
            features[:, 28] = np.random.choice([0.1, 0.3], n_samples, p=[0.7, 0.3])  # Low business risk
            features[:, 29] = np.random.beta(1, 3, n_samples) * 0.4  # Low geographic risk
            features[:, 30] = np.random.beta(2, 1, n_samples) * 0.7 + 0.3  # Receiver age (established)
            features[:, 31] = np.random.beta(1, 4, n_samples) * 0.4  # Receiver risk (lower)
            features[:, 32] = np.random.beta(3, 1, n_samples) * 0.7 + 0.3  # Verification score (higher)
            features[:, 33] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Domestic mostly
            features[:, 34] = np.random.beta(1, 3, n_samples) * 0.4  # Low velocity risk
            
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, features.shape)
        features = np.clip(features + noise, 0, 1)
        
        return features
    
    def generate_dataset(self, total_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset with specified fraud rate"""
        
        # Calculate fraud and legitimate sample counts
        n_fraud = int(total_samples * self.fraud_rate)
        n_legit = total_samples - n_fraud
        
        print(f"📊 Generating realistic fraud dataset:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Fraud samples: {n_fraud:,} ({self.fraud_rate*100:.1f}%)")
        print(f"   Legitimate samples: {n_legit:,} ({(1-self.fraud_rate)*100:.1f}%)")
        
        # Generate fraud samples
        fraud_features = self.generate_transaction_features(n_fraud, is_fraud=True)
        fraud_labels = np.ones(n_fraud)
        
        # Generate legitimate samples
        legit_features = self.generate_transaction_features(n_legit, is_fraud=False)
        legit_labels = np.zeros(n_legit)
        
        # Combine and shuffle
        X = np.vstack([fraud_features, legit_features])
        y = np.concatenate([fraud_labels, legit_labels])
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"✅ Dataset generated: {X.shape[0]:,} samples with {X.shape[1]} features")
        print(f"   Fraud rate in final dataset: {y.mean()*100:.3f}%")
        
        return X, y

def train_realistic_models():
    """Train realistic fraud detection models with proper evaluation"""
    
    # Generate large realistic dataset
    generator = RealisticFraudDataGenerator(fraud_rate=0.005)  # 0.5% fraud rate
    X, y = generator.generate_dataset(total_samples=200000)  # 200k samples
    
    # Create feature names
    feature_names = [
        'amount_ratio', 'daily_spending_ratio', 'large_amount_flag', 'round_amount', 'business_hours',
        'weekend_flag', 'holiday_flag', 'late_day_flag', 'friday_afternoon', 'off_hours',
        'authority_score', 'exec_bypass', 'authority_urgency', 'urgency_score', 'timing_pressure',
        'multiple_urgency', 'manipulation_score', 'confidentiality', 'trust_exploitation', 'business_score',
        'new_relationship', 'acquisition_language', 'communication_anomalies', 'grammar_issues', 'external_indicators',
        'sender_age', 'sender_risk', 'transaction_frequency', 'business_risk', 'geographic_risk',
        'receiver_age', 'receiver_risk', 'verification_score', 'cross_border', 'velocity_risk'
    ]
    
    # Split data properly: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"\n📊 Data splits:")
    print(f"   Training: {X_train.shape[0]:,} samples ({y_train.mean()*100:.3f}% fraud)")
    print(f"   Validation: {X_val.shape[0]:,} samples ({y_val.mean()*100:.3f}% fraud)")
    print(f"   Test: {X_test.shape[0]:,} samples ({y_test.mean()*100:.3f}% fraud)")
    
    # Train models for each bank with different specializations
    banks = {
        'bank_A': {
            'name': 'Wire Transfer Specialist', 
            'feature_focus': list(range(0, 15)),  # Amount and email features
            'max_depth': 15,
            'n_estimators': 200
        },
        'bank_B': {
            'name': 'Identity Verification Specialist',
            'feature_focus': list(range(10, 25)),  # Email and communication features
            'max_depth': 12,
            'n_estimators': 150
        },
        'bank_C': {
            'name': 'Network Analysis Specialist',
            'feature_focus': list(range(20, 35)),  # Account and network features
            'max_depth': 18,
            'n_estimators': 250
        }
    }
    
    # Train each bank model
    for bank_id, config in banks.items():
        print(f"\n🏦 Training {config['name']} ({bank_id})...")
        
        # Focus on relevant features for this bank's specialty
        focus_features = config['feature_focus']
        X_train_focus = X_train[:, focus_features]
        X_val_focus = X_val[:, focus_features]
        X_test_focus = X_test[:, focus_features]
        
        # Train model with regularization to prevent overfitting
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=10,  # Prevent overfitting
            min_samples_leaf=5,    # Prevent overfitting
            max_features='sqrt',   # Feature subsampling
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )
        
        # Fit with feature names for this bank's focus
        focus_feature_names = [feature_names[i] for i in focus_features]
        model.fit(X_train_focus, y_train)
        
        # Evaluate model
        val_score = model.score(X_val_focus, y_val)
        test_score = model.score(X_test_focus, y_test)
        
        # Get probability predictions for AUC
        val_proba = model.predict_proba(X_val_focus)[:, 1]
        test_proba = model.predict_proba(X_test_focus)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"   ✅ Validation Accuracy: {val_score:.3f}")
        print(f"   ✅ Test Accuracy: {test_score:.3f}")
        print(f"   🎯 Validation AUC: {val_auc:.3f}")
        print(f"   🎯 Test AUC: {test_auc:.3f}")
        
        # Test on a few fraud vs legitimate samples
        fraud_indices = np.where(y_test == 1)[0][:5]
        legit_indices = np.where(y_test == 0)[0][:5]
        
        fraud_scores = test_proba[fraud_indices]
        legit_scores = test_proba[legit_indices]
        
        print(f"   🚨 Fraud sample scores: {[f'{s:.3f}' for s in fraud_scores]}")
        print(f"   ✅ Legit sample scores: {[f'{s:.3f}' for s in legit_scores]}")
        
        # Save model
        model_path = f'models/{bank_id}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'bank_id': bank_id,
            'bank_name': config['name'],
            'feature_focus': focus_features,
            'feature_names': focus_feature_names,
            'validation_accuracy': float(val_score),
            'test_accuracy': float(test_score),
            'validation_auc': float(val_auc),
            'test_auc': float(test_auc),
            'training_samples': len(X_train),
            'fraud_rate': float(y_train.mean()),
            'model_params': {
                'n_estimators': config['n_estimators'],
                'max_depth': config['max_depth'],
                'min_samples_split': 10,
                'min_samples_leaf': 5
            }
        }
        
        metadata_path = f'models/{bank_id}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Saved model: {model_path}")
        print(f"   📄 Saved metadata: {metadata_path}")
    
    print(f"\n🎉 REALISTIC MODEL TRAINING COMPLETE!")
    print(f"   Models trained with realistic 0.5% fraud rate")
    print(f"   Proper train/validation/test splits")
    print(f"   Regularization to prevent overfitting")
    print(f"   Expected performance: 85-92% accuracy, 0.80-0.95 AUC")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Train realistic models
    train_realistic_models()
