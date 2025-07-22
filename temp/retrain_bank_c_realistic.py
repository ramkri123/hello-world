#!/usr/bin/env python3
"""
Ultra-Realistic Bank C Model Training
Create heavily overlapping fraud/legit distributions that reflect real-world complexity
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import json
import os

def generate_ultra_realistic_features(n_samples=50000, fraud_rate=0.005):
    """Generate ultra-realistic features with heavy overlap and noise"""
    
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    print(f"Generating {n_samples:,} samples with heavy overlap:")
    print(f"  Fraud: {n_fraud:,} ({fraud_rate*100:.1f}%)")
    print(f"  Legitimate: {n_legit:,} ({100-fraud_rate*100:.1f}%)")
    
    features = []
    labels = []
    
    # Generate LEGITIMATE transactions with varied patterns
    for i in range(n_legit):
        sample = np.zeros(35)
        
        # Create varied legitimate patterns (some look suspicious!)
        # Many legitimate transactions can have suspicious-looking patterns
        
        # Basic features - wide overlap with fraud
        sample[0] = np.random.beta(2, 5)  # Amount ratio - beta distribution
        sample[1] = np.random.beta(2, 4)  # Daily ratio
        sample[2] = np.random.beta(1.5, 6)  # Large amount flag
        sample[3] = np.random.choice([0, 1], p=[0.7, 0.3])  # Round amounts common in legit too
        sample[4] = np.random.choice([0, 1], p=[0.4, 0.6])  # Business hours
        sample[5] = np.random.choice([0, 1], p=[0.75, 0.25])  # Weekend
        sample[6] = np.random.choice([0, 1], p=[0.9, 0.1])   # Holiday
        sample[7] = np.random.choice([0, 1], p=[0.7, 0.3])   # Late day
        sample[8] = np.random.choice([0, 1], p=[0.85, 0.15]) # Friday afternoon
        sample[9] = np.random.choice([0, 1], p=[0.6, 0.4])   # Off hours
        
        # Email features - legitimate business emails can be urgent too!
        sample[10] = np.random.beta(1.5, 8)   # Authority score (some legit emails from authority)
        sample[11] = np.random.choice([0, 1], p=[0.92, 0.08])  # Exec bypass
        sample[12] = np.random.beta(1.5, 6)   # Authority urgency
        sample[13] = np.random.beta(2, 5)     # Urgency score (legit business is urgent!)
        sample[14] = np.random.beta(1.5, 7)   # Timing pressure
        sample[15] = np.random.beta(1, 9)     # Multiple urgency
        sample[16] = np.random.beta(1.5, 8)   # Manipulation score
        sample[17] = np.random.choice([0, 1], p=[0.8, 0.2])   # Confidentiality
        sample[18] = np.random.choice([0, 1], p=[0.9, 0.1])   # Trust exploitation
        sample[19] = np.random.beta(3, 4)     # Business score
        
        # Network features - HEAVY OVERLAP with fraud patterns
        sample[20] = np.random.beta(4, 2)     # Sender reputation
        sample[21] = np.random.beta(4, 2)     # Receiver reputation  
        sample[22] = np.random.beta(2, 6)     # Network risk
        sample[23] = np.random.beta(2, 6)     # Velocity anomaly
        sample[24] = np.random.beta(1.5, 6)   # Geographic anomaly
        sample[25] = np.random.beta(4, 2)     # Transaction history
        sample[26] = np.random.beta(3, 3)     # Relationship strength (mixed!)
        sample[27] = np.random.beta(2, 5)     # Behavioral deviation
        sample[28] = np.random.beta(2, 6)     # Time anomaly
        sample[29] = np.random.beta(4, 2)     # Account stability
        sample[30] = np.random.beta(2.5, 3.5) # Network centrality
        sample[31] = np.random.beta(1.5, 6)   # Suspicious connections
        sample[32] = np.random.beta(4, 2)     # KYC compliance
        sample[33] = np.random.beta(2, 5)     # Cross-border risk
        sample[34] = np.random.beta(3.5, 2.5) # Overall trust score
        
        # Add substantial noise to create more overlap
        noise = np.random.normal(0, 0.05, 35)
        sample += noise
        sample = np.clip(sample, 0, 1)
        
        features.append(sample)
        labels.append(0)
    
    # Generate FRAUD transactions - deliberately overlapping with legitimate!
    for i in range(n_fraud):
        sample = np.zeros(35)
        
        # Many fraud attempts try to look legitimate!
        # So we create patterns that are VERY similar to legitimate
        
        # Basic features - only SLIGHTLY different from legitimate
        sample[0] = np.random.beta(3, 4)      # Slightly higher amount ratios
        sample[1] = np.random.beta(3, 3.5)    # Slightly higher daily ratios
        sample[2] = np.random.beta(2.5, 4)    # More large amounts but not obvious
        sample[3] = np.random.choice([0, 1], p=[0.5, 0.5])   # More round amounts
        sample[4] = np.random.choice([0, 1], p=[0.5, 0.5])   # Less business hours
        sample[5] = np.random.choice([0, 1], p=[0.6, 0.4])   # More weekends
        sample[6] = np.random.choice([0, 1], p=[0.85, 0.15]) # Slightly more holidays
        sample[7] = np.random.choice([0, 1], p=[0.55, 0.45]) # More late day
        sample[8] = np.random.choice([0, 1], p=[0.7, 0.3])   # More Friday afternoon
        sample[9] = np.random.choice([0, 1], p=[0.45, 0.55]) # More off hours
        
        # Email features - sophisticated fraud tries to be subtle
        sample[10] = np.random.beta(2.5, 5)   # Slightly higher authority
        sample[11] = np.random.choice([0, 1], p=[0.8, 0.2])   # More exec bypass
        sample[12] = np.random.beta(2.5, 4)   # Higher authority urgency
        sample[13] = np.random.beta(3, 4)     # Higher urgency but overlapping
        sample[14] = np.random.beta(2.5, 5)   # More timing pressure
        sample[15] = np.random.beta(2, 6)     # More multiple urgency
        sample[16] = np.random.beta(3, 5)     # Higher manipulation but subtle
        sample[17] = np.random.choice([0, 1], p=[0.6, 0.4])   # More confidentiality
        sample[18] = np.random.choice([0, 1], p=[0.75, 0.25]) # More trust exploitation
        sample[19] = np.random.beta(2.5, 4)   # Similar business scores!
        
        # Network features - MINIMAL difference from legitimate (sophisticated fraud!)
        sample[20] = np.random.beta(3, 3)     # Slightly lower sender reputation
        sample[21] = np.random.beta(3, 3.5)   # Slightly lower receiver reputation
        sample[22] = np.random.beta(3, 4)     # Slightly higher network risk
        sample[23] = np.random.beta(3, 4)     # Slightly higher velocity anomaly
        sample[24] = np.random.beta(2.5, 4)   # Slightly higher geographic anomaly
        sample[25] = np.random.beta(3, 3.5)   # Slightly lower transaction history
        sample[26] = np.random.beta(2.5, 3.5) # Slightly lower relationship strength
        sample[27] = np.random.beta(3, 4)     # Slightly higher behavioral deviation
        sample[28] = np.random.beta(3, 4)     # Slightly higher time anomaly
        sample[29] = np.random.beta(3, 3.5)   # Slightly lower account stability
        sample[30] = np.random.beta(3, 3)     # Similar network centrality
        sample[31] = np.random.beta(3, 4)     # Slightly more suspicious connections
        sample[32] = np.random.beta(3, 3.5)   # Slightly lower KYC compliance
        sample[33] = np.random.beta(3.5, 3.5) # Similar cross-border risk
        sample[34] = np.random.beta(2.5, 3.5) # Slightly lower overall trust
        
        # Add even more noise to fraud to make it harder to detect
        noise = np.random.normal(0, 0.08, 35)  # More noise for fraud
        sample += noise
        sample = np.clip(sample, 0, 1)
        
        features.append(sample)
        labels.append(1)
    
    return np.array(features), np.array(labels)

def train_ultra_realistic_model():
    """Train with ultra-realistic overlapping data"""
    
    print("🏦 TRAINING BANK C - Ultra-Realistic Network Analysis")
    print("=" * 60)
    
    # Generate ultra-realistic overlapping dataset
    X, y = generate_ultra_realistic_features(n_samples=75000, fraud_rate=0.005)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(X_train):,} samples ({np.mean(y_train)*100:.3f}% fraud)")
    print(f"  Validation: {len(X_val):,} samples ({np.mean(y_val)*100:.3f}% fraud)")
    print(f"  Test: {len(X_test):,} samples ({np.mean(y_test)*100:.3f}% fraud)")
    
    # Train with strong regularization to handle overlapping data
    model = RandomForestClassifier(
        n_estimators=80,          # Fewer trees
        max_depth=6,              # Shallower trees
        min_samples_split=50,     # Require many samples to split
        min_samples_leaf=25,      # Require many samples in leaf
        max_features=0.3,         # Use fewer features
        max_samples=0.8,          # Use subset of samples per tree
        class_weight='balanced',  # Handle imbalanced data
        random_state=42
    )
    
    print("\n🤖 Training heavily regularized RandomForest...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\n📊 Model Performance:")
    print(f"  Training   - Accuracy: {train_acc:.3f}, AUC: {train_auc:.3f}")
    print(f"  Validation - Accuracy: {val_acc:.3f}, AUC: {val_auc:.3f}")
    print(f"  Test       - Accuracy: {test_acc:.3f}, AUC: {test_auc:.3f}")
    
    # Assess model quality
    if test_acc > 0.99:
        print("⚠️  WARNING: Still overfitting with perfect accuracy!")
    elif test_acc > 0.95:
        print("⚠️  WARNING: High accuracy may indicate some overfitting")
    elif test_acc >= 0.85:
        print("✅ Good realistic model performance")
    elif test_acc >= 0.75:
        print("✅ Acceptable model with realistic challenge")
    else:
        print("⚠️  Model may be underfitting or data too noisy")
    
    # Test sample predictions
    print(f"\n🧪 Sample Fraud Scores (should show realistic ranges):")
    fraud_indices = np.where(y_test == 1)[0][:8]
    legit_indices = np.where(y_test == 0)[0][:8]
    
    print("  Fraud samples:")
    for i in fraud_indices:
        score = test_proba[i]
        print(f"    Sample {i}: {score:.3f}")
    
    print("  Legitimate samples:")
    for i in legit_indices:
        score = test_proba[i]
        print(f"    Sample {i}: {score:.3f}")
    
    # Calculate score statistics
    fraud_scores = test_proba[y_test == 1]
    legit_scores = test_proba[y_test == 0]
    
    print(f"\n📈 Score Distribution Analysis:")
    print(f"  Fraud scores  - Mean: {np.mean(fraud_scores):.3f}, Std: {np.std(fraud_scores):.3f}")
    print(f"                  Range: {np.min(fraud_scores):.3f} to {np.max(fraud_scores):.3f}")
    print(f"  Legit scores  - Mean: {np.mean(legit_scores):.3f}, Std: {np.std(legit_scores):.3f}")
    print(f"                  Range: {np.min(legit_scores):.3f} to {np.max(legit_scores):.3f}")
    print(f"  Score separation: {np.mean(fraud_scores) - np.mean(legit_scores):.3f}")
    
    # Check overlap
    fraud_below_threshold = np.sum(fraud_scores < 0.5)
    legit_above_threshold = np.sum(legit_scores > 0.5)
    overlap = fraud_below_threshold + legit_above_threshold
    total_test = len(fraud_scores) + len(legit_scores)
    print(f"  Overlapping cases: {overlap}/{total_test} ({overlap/total_test*100:.1f}%)")
    print(f"    Fraud below 0.5: {fraud_below_threshold}/{len(fraud_scores)}")
    print(f"    Legit above 0.5: {legit_above_threshold}/{len(legit_scores)}")
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/bank_C_model.pkl'
    metadata_path = 'models/bank_C_metadata.json'
    
    joblib.dump(model, model_path)
    
    metadata = {
        "bank_id": "bank_C",
        "specialty": "network_analysis_specialist", 
        "feature_range": "20-34",
        "focus": "Network patterns, account relationships, transaction velocity",
        "model_type": "RandomForestClassifier",
        "training_approach": "ultra_realistic_overlapping",
        "training_samples": len(X_train),
        "fraud_rate": float(np.mean(y_train)),
        "performance": {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc), 
            "test_accuracy": float(test_acc),
            "train_auc": float(train_auc),
            "val_auc": float(val_auc),
            "test_auc": float(test_auc)
        },
        "score_stats": {
            "fraud_mean": float(np.mean(fraud_scores)),
            "fraud_std": float(np.std(fraud_scores)),
            "fraud_range": [float(np.min(fraud_scores)), float(np.max(fraud_scores))],
            "legit_mean": float(np.mean(legit_scores)),
            "legit_std": float(np.std(legit_scores)),
            "legit_range": [float(np.min(legit_scores)), float(np.max(legit_scores))],
            "separation": float(np.mean(fraud_scores) - np.mean(legit_scores)),
            "overlap_percentage": float(overlap/total_test*100),
            "fraud_below_threshold": int(fraud_below_threshold),
            "legit_above_threshold": int(legit_above_threshold)
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Model saved:")
    print(f"  Model: {model_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"\n✅ Ultra-realistic Bank C training complete!")
    
    return model, metadata

if __name__ == "__main__":
    train_ultra_realistic_model()
