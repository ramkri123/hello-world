#!/usr/bin/env python3
"""
Realistic Multi-Scenario Training System
Handles 4 different bank knowledge scenarios with appropriate weights
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json
import sys

sys.path.append('src')
from consortium.privacy_preserving_nlp import PrivacyPreservingNLP

class ScenarioBasedTraining:
    """Training system that handles different bank knowledge scenarios"""
    
    def __init__(self):
        self.nlp = PrivacyPreservingNLP()
        
    def generate_scenario_data(self, n_samples=30000):
        """Generate training data for all 4 scenarios"""
        
        all_features = []
        all_labels = []
        all_scenarios = []
        all_weights = []
        
        n_fraud = int(n_samples * 0.005)  # 0.5% fraud rate
        
        print(f"Generating {n_samples} samples across 4 knowledge scenarios...")
        
        # Scenario probabilities based on real-world distribution
        scenario_probs = [0.4, 0.25, 0.25, 0.1]  # Realistic distribution
        
        for sample_idx in range(n_samples):
            is_fraud = sample_idx < n_fraud
            
            # Choose scenario for this sample
            scenario = np.random.choice(4, p=scenario_probs)
            
            # Generate base transaction
            if is_fraud:
                amount = np.random.choice([75000, 150000, 300000, 485000])
                email = np.random.choice([
                    "CEO urgent crypto investment wire transfer confidential",
                    "President emergency payment strategic acquisition secret",
                    "Executive urgent vendor payment deadline confidential"
                ])
            else:
                amount = np.random.choice([10000, 25000, 50000, 100000])
                email = np.random.choice([
                    "Invoice payment office supplies monthly billing",
                    "Vendor payment contracted services quarterly",
                    "Regular payment insurance premium"
                ])
            
            # Base transaction data
            transaction_data = {
                'amount': amount,
                'sender_account': f'SENDER_{sample_idx}',
                'receiver_account': f'RECEIVER_{sample_idx}',
                'transaction_type': 'wire_transfer'
            }
            
            # Generate features for each scenario
            features, weight = self._generate_scenario_features(
                transaction_data, email, scenario, is_fraud
            )
            
            all_features.append(features)
            all_labels.append(1 if is_fraud else 0)
            all_scenarios.append(scenario)
            all_weights.append(weight)
        
        return (np.array(all_features), np.array(all_labels), 
                np.array(all_scenarios), np.array(all_weights))
    
    def generate_bank_specific_data(self, n_samples, scenario_distribution, bank_id):
        """Generate training data with bank-specific scenario distribution"""
        
        all_features = []
        all_labels = []
        all_scenarios = []
        all_weights = []
        
        n_fraud = int(n_samples * 0.005)  # 0.5% fraud rate
        
        print(f"   Generating {n_samples} samples for Bank {bank_id}...")
        
        for sample_idx in range(n_samples):
            is_fraud = sample_idx < n_fraud
            
            # Choose scenario based on bank's distribution
            scenario = np.random.choice(4, p=scenario_distribution)
            
            # Generate base transaction
            if is_fraud:
                amount = np.random.choice([75000, 150000, 300000, 485000])
                email = np.random.choice([
                    "CEO urgent crypto investment wire transfer confidential",
                    "President emergency payment strategic acquisition secret",
                    "Executive urgent vendor payment deadline confidential"
                ])
            else:
                amount = np.random.choice([10000, 25000, 50000, 100000])
                email = np.random.choice([
                    "Invoice payment office supplies monthly billing",
                    "Vendor payment contracted services quarterly",
                    "Regular payment insurance premium"
                ])
            
            # Base transaction data
            transaction_data = {
                'amount': amount,
                'sender_account': f'{bank_id}_SENDER_{sample_idx}',
                'receiver_account': f'{bank_id}_RECEIVER_{sample_idx}',
                'transaction_type': 'wire_transfer'
            }
            
            # Generate features for this bank's scenario
            features, weight = self._generate_scenario_features(
                transaction_data, email, scenario, is_fraud
            )
            
            all_features.append(features)
            all_labels.append(1 if is_fraud else 0)
            all_scenarios.append(scenario)
            all_weights.append(weight)
        
        return (np.array(all_features), np.array(all_labels), 
                np.array(all_scenarios), np.array(all_weights))
    
    def _generate_scenario_features(self, transaction_data, email, scenario, is_fraud):
        """Generate features based on bank's knowledge scenario"""
        
        # Base NLP features (always available)
        base_features = self.nlp.convert_to_anonymous_features(
            transaction_data, email, sender_data=None, receiver_data=None
        )
        
        # Scenario-specific account information
        if scenario == 0:
            # Bank knows BOTH sender AND receiver
            sender_data = self._generate_account_data(is_sender=True, is_known=True, is_fraud=is_fraud)
            receiver_data = self._generate_account_data(is_sender=False, is_known=True, is_fraud=is_fraud)
            confidence_weight = 1.0  # Highest confidence
            
        elif scenario == 1:
            # Bank knows ONLY sender
            sender_data = self._generate_account_data(is_sender=True, is_known=True, is_fraud=is_fraud)
            receiver_data = self._generate_account_data(is_sender=False, is_known=False, is_fraud=is_fraud)
            confidence_weight = 0.8  # High confidence (own customer)
            
        elif scenario == 2:
            # Bank knows ONLY receiver
            sender_data = self._generate_account_data(is_sender=True, is_known=False, is_fraud=is_fraud)
            receiver_data = self._generate_account_data(is_sender=False, is_known=True, is_fraud=is_fraud)
            confidence_weight = 0.7  # Medium-high confidence
            
        else:  # scenario == 3
            # Bank knows NEITHER sender NOR receiver
            sender_data = self._generate_account_data(is_sender=True, is_known=False, is_fraud=is_fraud)
            receiver_data = self._generate_account_data(is_sender=False, is_known=False, is_fraud=is_fraud)
            confidence_weight = 0.4  # Lower confidence (external transaction)
        
        # Generate enhanced features with account data
        enhanced_features = self.nlp.convert_to_anonymous_features(
            transaction_data, email, sender_data, receiver_data
        )
        
        return enhanced_features, confidence_weight
    
    def _generate_account_data(self, is_sender, is_known, is_fraud):
        """Generate realistic account data based on knowledge and fraud status"""
        
        if not is_known:
            # Unknown accounts - use conservative estimates
            return {
                'account_age_years': np.random.uniform(0.5, 5.0),
                'risk_score': np.random.uniform(0.3, 0.7),  # Unknown = medium risk
                'transaction_count': np.random.randint(10, 500),
                'business_type': 'unknown',
                'geographic_risk': np.random.uniform(0.4, 0.6),
                'verification_score': np.random.uniform(0.3, 0.7)
            }
        
        # Known accounts - realistic distributions
        if is_fraud:
            if is_sender:
                # Fraud sender accounts (often compromised legitimate accounts)
                return {
                    'account_age_years': np.random.uniform(1.0, 8.0),  # Established accounts
                    'risk_score': np.random.uniform(0.1, 0.4),  # Low risk (compromised)
                    'transaction_count': np.random.randint(100, 2000),
                    'business_type': 'business',
                    'geographic_risk': np.random.uniform(0.1, 0.3),
                    'verification_score': np.random.uniform(0.7, 0.9)  # Well verified
                }
            else:
                # Fraud receiver accounts (often new/suspicious)
                return {
                    'account_age_years': np.random.uniform(0.01, 0.5),  # Very new
                    'risk_score': np.random.uniform(0.6, 0.9),  # High risk
                    'transaction_count': np.random.randint(1, 50),
                    'business_type': 'individual',
                    'geographic_risk': np.random.uniform(0.5, 0.8),
                    'verification_score': np.random.uniform(0.2, 0.5)  # Poor verification
                }
        else:
            # Legitimate accounts
            return {
                'account_age_years': np.random.uniform(2.0, 15.0),
                'risk_score': np.random.uniform(0.05, 0.25),
                'transaction_count': np.random.randint(200, 5000),
                'business_type': np.random.choice(['business', 'individual']),
                'geographic_risk': np.random.uniform(0.1, 0.3),
                'verification_score': np.random.uniform(0.7, 0.95)
            }

def train_scenario_aware_models():
    """Train models that understand different knowledge scenarios"""
    
    trainer = ScenarioBasedTraining()
    
    print("🏦 TRAINING SCENARIO-AWARE FRAUD DETECTION MODELS")
    print("=" * 60)
    
    # Generate training data
    X, y, scenarios, weights = trainer.generate_scenario_data(30000)
    
    print(f"Generated {len(X)} samples with {len(X[0])} features")
    print(f"Fraud rate: {np.mean(y)*100:.3f}%")
    
    # Print scenario distribution
    for scenario in range(4):
        count = np.sum(scenarios == scenario)
        avg_weight = np.mean(weights[scenarios == scenario])
        print(f"  Scenario {scenario}: {count:,} samples, avg weight: {avg_weight:.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test, scenarios_train, scenarios_test, weights_train, weights_test = train_test_split(
        X, y, scenarios, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train each bank with their specific scenario distributions
    bank_configs = [
        {
            'id': 'A', 
            'specialty': 'wire_transfer_specialist',
            'description': 'Focuses on sender account patterns (usually knows sender)',
            'scenario_distribution': [0.1, 0.6, 0.05, 0.25],  # Mostly knows sender
            'expertise_scenarios': [0, 1]  # Best at scenarios where sender is known
        },
        {
            'id': 'B',
            'specialty': 'identity_verification_specialist', 
            'description': 'Focuses on receiver verification (usually knows receiver)',
            'scenario_distribution': [0.15, 0.1, 0.55, 0.2],  # Mostly knows receiver
            'expertise_scenarios': [0, 2]  # Best at scenarios where receiver is known
        },
        {
            'id': 'C',
            'specialty': 'network_analysis_specialist',
            'description': 'Focuses on external patterns (often knows neither)',
            'scenario_distribution': [0.2, 0.2, 0.2, 0.4],   # More external analysis
            'expertise_scenarios': [0, 1, 2, 3]  # Good at all scenarios
        }
    ]
    
    for bank_config in bank_configs:
        bank_id = bank_config['id']
        specialty = bank_config['specialty']
        description = bank_config['description']
        bank_scenario_dist = bank_config['scenario_distribution']
        expertise_scenarios = bank_config['expertise_scenarios']
        
        print(f"\n🏦 Training Bank {bank_id} - {specialty}")
        print(f"   {description}")
        print(f"   Scenario distribution: {bank_scenario_dist}")
        
        # Generate bank-specific training data
        bank_X, bank_y, bank_scenarios, bank_weights = trainer.generate_bank_specific_data(
            30000, bank_scenario_dist, bank_id
        )
        
        # Split data
        X_train_bank, X_test_bank, y_train_bank, y_test_bank, scenarios_train_bank, scenarios_test_bank, weights_train_bank, weights_test_bank = train_test_split(
            bank_X, bank_y, bank_scenarios, bank_weights, test_size=0.2, random_state=42, stratify=bank_y
        )
        
        # Boost confidence for expertise scenarios
        expertise_boost = np.ones_like(weights_train_bank)
        for scenario in expertise_scenarios:
            expertise_mask = scenarios_train_bank == scenario
            expertise_boost[expertise_mask] *= 1.2  # 20% confidence boost
        
        final_weights = weights_train_bank * expertise_boost
        # Train model with bank-specific weighted samples
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=25,
            class_weight='balanced',
            random_state=42 + ord(bank_id) - ord('A')
        )
        
        # Use bank-specific sample weights
        model.fit(X_train_bank, y_train_bank, sample_weight=final_weights)
        
        # Evaluate overall performance
        test_proba = model.predict_proba(X_test_bank)[:, 1]
        accuracy = accuracy_score(y_test_bank, model.predict(X_test_bank))
        auc = roc_auc_score(y_test_bank, test_proba)
        
        print(f"   Overall - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        # Evaluate by scenario
        for scenario in range(4):
            scenario_mask = scenarios_test_bank == scenario
            if np.sum(scenario_mask) > 0:
                scenario_acc = accuracy_score(
                    y_test_bank[scenario_mask], 
                    model.predict(X_test_bank[scenario_mask])
                )
                scenario_fraud_scores = test_proba[scenario_mask & (y_test_bank == 1)]
                scenario_legit_scores = test_proba[scenario_mask & (y_test_bank == 0)]
                
                avg_weight = np.mean(weights_test_bank[scenario_mask])
                
                print(f"   Scenario {scenario} - Acc: {scenario_acc:.3f}, Weight: {avg_weight:.2f}")
                if len(scenario_fraud_scores) > 0:
                    print(f"     Fraud scores: {np.mean(scenario_fraud_scores):.3f}")
                if len(scenario_legit_scores) > 0:
                    print(f"     Legit scores: {np.mean(scenario_legit_scores):.3f}")
        
        # Save model with bank-specific metadata
        model_path = f'models/bank_{bank_id}_model.pkl'
        metadata_path = f'models/bank_{bank_id}_metadata.json'
        
        joblib.dump(model, model_path)
        
        metadata = {
            "bank_id": f"bank_{bank_id}",
            "specialty": specialty,
            "description": description,
            "training_approach": "bank_specific_scenario_aware",
            "scenario_distribution": bank_scenario_dist,
            "expertise_scenarios": expertise_scenarios,
            "scenarios": {
                "0": "Knows both sender and receiver",
                "1": "Knows only sender",
                "2": "Knows only receiver", 
                "3": "Knows neither sender nor receiver"
            },
            "scenario_weights": {
                "0": 1.0,  # Highest confidence
                "1": 0.8,  # High confidence
                "2": 0.7,  # Medium confidence
                "3": 0.4   # Lower confidence
            },
            "performance": {
                "overall_accuracy": float(accuracy),
                "overall_auc": float(auc)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved: {model_path}")
    
    print(f"\n✅ Scenario-aware models trained successfully!")
    
    # Test scenario-based inference
    print(f"\n🧪 Testing scenario-based inference:")
    
    test_transaction = {
        'amount': 485000,
        'sender_account': 'TEST_SENDER',
        'receiver_account': 'TEST_RECEIVER',
        'transaction_type': 'wire_transfer'
    }
    
    test_email = "CEO urgent crypto investment wire transfer confidential"
    
    for scenario in range(4):
        print(f"\n   Scenario {scenario}:")
        scenario_names = [
            "Bank knows both accounts",
            "Bank knows only sender", 
            "Bank knows only receiver",
            "Bank knows neither account"
        ]
        print(f"   {scenario_names[scenario]}")
        
        features, weight = trainer._generate_scenario_features(
            test_transaction, test_email, scenario, is_fraud=True
        )
        
        print(f"   Confidence weight: {weight:.2f}")
        
        for bank_id in ['A', 'B', 'C']:
            model = joblib.load(f'models/bank_{bank_id}_model.pkl')
            score = model.predict_proba([features])[0][1]
            weighted_score = score * weight
            print(f"   Bank {bank_id}: {score:.3f} (weighted: {weighted_score:.3f})")

if __name__ == "__main__":
    train_scenario_aware_models()
