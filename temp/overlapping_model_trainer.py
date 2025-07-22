#!/usr/bin/env python3
"""
Realistic Fraud Detection Model Trainer - Version 2
Creates overlapping, noisy data that mimics real-world fraud detection challenges
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
from typing import Tuple, Dict

class RealisticFraudDataGenerator:
    """Generate realistic fraud detection dataset with overlapping distributions"""
    
    def __init__(self, fraud_rate: float = 0.005):  # 0.5% fraud rate
        self.fraud_rate = fraud_rate
        
    def generate_overlapping_features(self, n_samples: int, is_fraud: bool) -> np.ndarray:
        """Generate realistic features with significant overlap between fraud and legitimate"""
        features = np.zeros((n_samples, 35))
        
        if is_fraud:
            # Fraud patterns - shifted distributions but with overlap
            # Amount features (0-4) - fraud tends to be larger but lots of overlap
            features[:, 0] = np.random.lognormal(0.3, 0.8, n_samples)  # Amount ratio - some high, some normal
            features[:, 0] = np.clip(features[:, 0] / 5.0, 0, 1)  # Normalize
            
            features[:, 1] = np.random.lognormal(0.2, 0.7, n_samples)  # Daily spending ratio
            features[:, 1] = np.clip(features[:, 1] / 3.0, 0, 1)
            
            features[:, 2] = np.random.beta(2, 3, n_samples)  # Large amount flag - mixed
            features[:, 3] = np.random.binomial(1, 0.4, n_samples)  # Round amounts - somewhat higher
            features[:, 4] = np.random.binomial(1, 0.3, n_samples)  # Business hours - some off-hours
            
            # Timing features (5-9) - fraud has some timing bias but not extreme
            features[:, 5] = np.random.binomial(1, 0.4, n_samples)  # Weekend - slightly higher
            features[:, 6] = np.random.binomial(1, 0.05, n_samples)  # Holiday - still rare
            features[:, 7] = np.random.binomial(1, 0.45, n_samples)  # Late day - slightly higher
            features[:, 8] = np.random.binomial(1, 0.25, n_samples)  # Friday afternoon
            features[:, 9] = np.random.binomial(1, 0.5, n_samples)  # Off hours - mixed
            
            # Email features (10-24) - these have the strongest but still imperfect signals
            features[:, 10] = np.random.beta(3, 5, n_samples) * 0.7  # Authority score - higher but not extreme
            features[:, 11] = np.random.binomial(1, 0.3, n_samples)  # Executive bypass
            features[:, 12] = np.random.beta(2, 4, n_samples) * 0.6  # Authority urgency
            features[:, 13] = np.random.beta(3, 4, n_samples) * 0.8  # Urgency score
            features[:, 14] = np.random.beta(2, 5, n_samples) * 0.6  # Timing pressure
            features[:, 15] = np.random.beta(2, 6, n_samples) * 0.5  # Multiple urgency
            features[:, 16] = np.random.beta(3, 5, n_samples) * 0.7  # Manipulation score
            features[:, 17] = np.random.binomial(1, 0.4, n_samples)  # Confidentiality
            features[:, 18] = np.random.binomial(1, 0.25, n_samples)  # Trust exploitation
            features[:, 19] = np.random.beta(2, 4, n_samples) * 0.4  # Business score - lower
            features[:, 20] = np.random.binomial(1, 0.35, n_samples)  # New relationship
            features[:, 21] = np.random.binomial(1, 0.2, n_samples)  # Acquisition language
            features[:, 22] = np.random.beta(3, 4, n_samples) * 0.8  # Communication anomalies
            features[:, 23] = np.random.beta(2, 5, n_samples) * 0.6  # Grammar issues
            features[:, 24] = np.random.binomial(1, 0.6, n_samples)  # External indicators
            
            # Account features (25-34) - some signal but lots of overlap
            features[:, 25] = np.random.beta(2, 4, n_samples) * 0.6  # Sender age - somewhat newer
            features[:, 26] = np.random.beta(4, 3, n_samples) * 0.7  # Sender risk - higher
            features[:, 27] = np.random.beta(2, 5, n_samples) * 0.5  # Transaction frequency - lower
            features[:, 28] = np.random.beta(3, 3, n_samples) * 0.6  # Business risk
            features[:, 29] = np.random.beta(3, 4, n_samples) * 0.7  # Geographic risk
            features[:, 30] = np.random.beta(1, 6, n_samples) * 0.4  # Receiver age - newer
            features[:, 31] = np.random.beta(5, 3, n_samples) * 0.8  # Receiver risk - higher
            features[:, 32] = np.random.beta(2, 5, n_samples) * 0.4  # Verification score - lower
            features[:, 33] = np.random.binomial(1, 0.35, n_samples)  # Cross-border
            features[:, 34] = np.random.beta(4, 3, n_samples) * 0.8  # Velocity risk
            
        else:
            # Legitimate patterns - overlapping with fraud but shifted
            # Amount features (0-4) - legitimate are mostly normal but some high-value legit transactions
            features[:, 0] = np.random.lognormal(-0.2, 0.6, n_samples)  # Amount ratio - lower but overlapping
            features[:, 0] = np.clip(features[:, 0] / 3.0, 0, 1)
            
            features[:, 1] = np.random.lognormal(-0.1, 0.5, n_samples)  # Daily spending ratio
            features[:, 1] = np.clip(features[:, 1] / 2.0, 0, 1)
            
            features[:, 2] = np.random.beta(1, 4, n_samples)  # Large amount flag - lower
            features[:, 3] = np.random.binomial(1, 0.2, n_samples)  # Round amounts - lower
            features[:, 4] = np.random.binomial(1, 0.7, n_samples)  # Business hours - higher
            
            # Timing features (5-9) - legitimate prefer business hours but not exclusively
            features[:, 5] = np.random.binomial(1, 0.25, n_samples)  # Weekend - lower
            features[:, 6] = np.random.binomial(1, 0.03, n_samples)  # Holiday - rare
            features[:, 7] = np.random.binomial(1, 0.25, n_samples)  # Late day - lower
            features[:, 8] = np.random.binomial(1, 0.15, n_samples)  # Friday afternoon
            features[:, 9] = np.random.binomial(1, 0.3, n_samples)  # Off hours - lower
            
            # Email features (10-24) - legitimate have lower but not zero values
            features[:, 10] = np.random.beta(1, 6, n_samples) * 0.4  # Authority score - much lower
            features[:, 11] = np.random.binomial(1, 0.05, n_samples)  # Executive bypass - rare
            features[:, 12] = np.random.beta(1, 8, n_samples) * 0.3  # Authority urgency - low
            features[:, 13] = np.random.beta(2, 8, n_samples) * 0.4  # Urgency score - lower
            features[:, 14] = np.random.beta(1, 10, n_samples) * 0.2  # Timing pressure - low
            features[:, 15] = np.random.beta(1, 12, n_samples) * 0.15  # Multiple urgency - very low
            features[:, 16] = np.random.beta(1, 8, n_samples) * 0.3  # Manipulation score - low
            features[:, 17] = np.random.binomial(1, 0.15, n_samples)  # Confidentiality - lower
            features[:, 18] = np.random.binomial(1, 0.05, n_samples)  # Trust exploitation - rare
            features[:, 19] = np.random.beta(4, 3, n_samples) * 0.8  # Business score - higher
            features[:, 20] = np.random.binomial(1, 0.15, n_samples)  # New relationship - lower
            features[:, 21] = np.random.binomial(1, 0.08, n_samples)  # Acquisition language - rare
            features[:, 22] = np.random.beta(1, 6, n_samples) * 0.4  # Communication anomalies - lower
            features[:, 23] = np.random.beta(1, 10, n_samples) * 0.3  # Grammar issues - lower
            features[:, 24] = np.random.binomial(1, 0.3, n_samples)  # External indicators - lower
            
            # Account features (25-34) - legitimate accounts are more established
            features[:, 25] = np.random.beta(4, 2, n_samples) * 0.9  # Sender age - older
            features[:, 26] = np.random.beta(2, 6, n_samples) * 0.4  # Sender risk - lower
            features[:, 27] = np.random.beta(5, 2, n_samples) * 0.8  # Transaction frequency - higher
            features[:, 28] = np.random.beta(2, 5, n_samples) * 0.3  # Business risk - lower
            features[:, 29] = np.random.beta(2, 6, n_samples) * 0.4  # Geographic risk - lower
            features[:, 30] = np.random.beta(3, 2, n_samples) * 0.8  # Receiver age - established
            features[:, 31] = np.random.beta(2, 6, n_samples) * 0.4  # Receiver risk - lower
            features[:, 32] = np.random.beta(5, 2, n_samples) * 0.8  # Verification score - higher
            features[:, 33] = np.random.binomial(1, 0.15, n_samples)  # Cross-border - lower
            features[:, 34] = np.random.beta(2, 6, n_samples) * 0.4  # Velocity risk - lower
            
        # Add significant noise to create more realistic overlap
        noise_level = 0.15  # 15% noise
        noise = np.random.normal(0, noise_level, features.shape)
        features = np.clip(features + noise, 0, 1)
        
        # Add some completely random features to simulate real-world noise
        for i in range(features.shape[1]):
            if np.random.random() < 0.1:  # 10% chance for each feature
                random_mask = np.random.random(n_samples) < 0.05  # 5% of samples
                features[random_mask, i] = np.random.random(np.sum(random_mask))
        
        return features
    
    def generate_dataset(self, total_samples: int = 150000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset with specified fraud rate"""
        
        # Calculate fraud and legitimate sample counts
        n_fraud = int(total_samples * self.fraud_rate)
        n_legit = total_samples - n_fraud
        
        print(f"📊 Generating OVERLAPPING fraud dataset:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Fraud samples: {n_fraud:,} ({self.fraud_rate*100:.1f}%)")
        print(f"   Legitimate samples: {n_legit:,} ({(1-self.fraud_rate)*100:.1f}%)")
        
        # Generate fraud samples
        fraud_features = self.generate_overlapping_features(n_fraud, is_fraud=True)
        fraud_labels = np.ones(n_fraud)
        
        # Generate legitimate samples
        legit_features = self.generate_overlapping_features(n_legit, is_fraud=False)
        legit_labels = np.zeros(n_legit)
        
        # Combine and shuffle
        X = np.vstack([fraud_features, legit_features])
        y = np.concatenate([fraud_labels, legit_labels])
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"✅ Overlapping dataset generated: {X.shape[0]:,} samples with {X.shape[1]} features")
        print(f"   Final fraud rate: {y.mean()*100:.3f}%")
        
        # Show feature overlap statistics
        fraud_mask = y == 1
        legit_mask = y == 0
        
        overlap_score = 0
        for i in range(X.shape[1]):
            fraud_mean = X[fraud_mask, i].mean()
            legit_mean = X[legit_mask, i].mean()
            fraud_std = X[fraud_mask, i].std()
            legit_std = X[legit_mask, i].std()
            
            # Calculate overlap (inverse of separation)
            separation = abs(fraud_mean - legit_mean) / (fraud_std + legit_std + 1e-6)
            overlap = 1 / (1 + separation)
            overlap_score += overlap
            
        avg_overlap = overlap_score / X.shape[1]
        print(f"   📊 Average feature overlap: {avg_overlap:.3f} (higher = more realistic)")
        
        return X, y

def train_realistic_models():
    """Train realistic fraud detection models with proper evaluation"""
    
    # Generate large realistic dataset with overlapping distributions
    generator = RealisticFraudDataGenerator(fraud_rate=0.005)  # 0.5% fraud rate
    X, y = generator.generate_dataset(total_samples=150000)  # 150k samples
    
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
            'max_depth': 8,  # Reduced to prevent overfitting
            'n_estimators': 100,  # Reduced
            'min_samples_leaf': 20  # Increased to prevent overfitting
        },
        'bank_B': {
            'name': 'Identity Verification Specialist',
            'feature_focus': list(range(10, 25)),  # Email and communication features
            'max_depth': 7,  # Reduced
            'n_estimators': 80,
            'min_samples_leaf': 25
        },
        'bank_C': {
            'name': 'Network Analysis Specialist',
            'feature_focus': list(range(20, 35)),  # Account and network features
            'max_depth': 9,  # Reduced
            'n_estimators': 120,
            'min_samples_leaf': 15
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
        
        # Train model with strong regularization to prevent overfitting
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=50,  # Strong regularization
            min_samples_leaf=config['min_samples_leaf'],  # Strong regularization
            max_features=0.5,   # Feature subsampling
            random_state=42,
            class_weight='balanced_subsample',  # Handle class imbalance with subsampling
            n_jobs=-1,
            bootstrap=True,
            oob_score=True  # Out-of-bag score for monitoring
        )
        
        # Fit model
        model.fit(X_train_focus, y_train)
        
        # Evaluate model
        val_score = model.score(X_val_focus, y_val)
        test_score = model.score(X_test_focus, y_test)
        oob_score = model.oob_score_
        
        # Get probability predictions for AUC
        val_proba = model.predict_proba(X_val_focus)[:, 1]
        test_proba = model.predict_proba(X_test_focus)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(model, X_train_focus, y_train, cv=5, scoring='roc_auc')
        
        print(f"   ✅ Validation Accuracy: {val_score:.3f}")
        print(f"   ✅ Test Accuracy: {test_score:.3f}")
        print(f"   📊 OOB Score: {oob_score:.3f}")
        print(f"   🎯 Validation AUC: {val_auc:.3f}")
        print(f"   🎯 Test AUC: {test_auc:.3f}")
        print(f"   🔄 CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Check for overfitting
        if val_score - test_score > 0.05:
            print(f"   ⚠️  WARNING: Possible overfitting detected (val-test gap: {val_score-test_score:.3f})")
        
        # Test on fraud vs legitimate samples
        fraud_indices = np.where(y_test == 1)[0]
        legit_indices = np.where(y_test == 0)[0]
        
        if len(fraud_indices) >= 5 and len(legit_indices) >= 5:
            fraud_scores = test_proba[fraud_indices[:5]]
            legit_scores = test_proba[legit_indices[:5]]
            
            print(f"   🚨 Sample fraud scores: {[f'{s:.3f}' for s in fraud_scores]}")
            print(f"   ✅ Sample legit scores: {[f'{s:.3f}' for s in legit_scores]}")
            
            # Calculate score separation
            fraud_mean = test_proba[fraud_indices].mean()
            legit_mean = test_proba[legit_indices].mean()
            separation = fraud_mean - legit_mean
            print(f"   📊 Score separation: {separation:.3f} (fraud avg: {fraud_mean:.3f}, legit avg: {legit_mean:.3f})")
        
        # Save model
        model_path = f'models/{bank_id}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'bank_id': bank_id,
            'bank_name': config['name'],
            'feature_focus': focus_features,
            'feature_names': [feature_names[i] for i in focus_features],
            'validation_accuracy': float(val_score),
            'test_accuracy': float(test_score),
            'oob_score': float(oob_score),
            'validation_auc': float(val_auc),
            'test_auc': float(test_auc),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'training_samples': len(X_train),
            'fraud_rate': float(y_train.mean()),
            'model_params': {
                'n_estimators': config['n_estimators'],
                'max_depth': config['max_depth'],
                'min_samples_split': 50,
                'min_samples_leaf': config['min_samples_leaf']
            }
        }
        
        metadata_path = f'models/{bank_id}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Saved: {model_path}")
    
    print(f"\n🎉 REALISTIC MODEL TRAINING COMPLETE!")
    print(f"   ✅ Models trained with overlapping distributions")
    print(f"   ✅ Strong regularization to prevent overfitting")
    print(f"   ✅ Realistic 0.5% fraud rate")
    print(f"   ✅ Expected: 85-95% accuracy, 0.75-0.90 AUC")
    print(f"   ✅ Models should now provide nuanced scores between 0.1-0.9")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Train realistic models
    train_realistic_models()
