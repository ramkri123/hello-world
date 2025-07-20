"""
Enhanced Training Script for Demo-Aligned Fraud Detection Models
Creates realistic fraud patterns that align with our BEC demo scenario
"""

import pandas as pd
import numpy as np
from consortium_comparison_score_prototype import BankSimulator, ConsortiumComparisonService
import os

def generate_realistic_fraud_training_data(bank_id, n_samples=2000):
    """Generate realistic training data with patterns that align with our demo scenarios"""
    np.random.seed(42 + hash(bank_id) % 1000)  # Consistent but different per bank
    
    data = []
    
    # Generate normal transactions (70%)
    n_normal = int(n_samples * 0.7)
    for _ in range(n_normal):
        features = []
        
        # Normal transaction patterns
        features.extend([
            np.random.beta(2, 8),      # amount (lower amounts more common)
            np.random.beta(2, 5),      # urgency (usually low)
            np.random.beta(8, 2),      # sender_reputation (usually high)
            np.random.beta(8, 2),      # receiver_reputation (usually high)
            np.random.beta(8, 2),      # account_age (usually established)
            np.random.beta(2, 8),      # velocity (usually low)
            np.random.beta(1, 9),      # unusual_hour (rarely unusual)
            np.random.beta(1, 9),      # weekend_flag (rarely weekend)
            np.random.beta(8, 2),      # device_trust (usually trusted)
            np.random.beta(8, 2),      # location_trust (usually trusted)
        ])
        
        # Geographic and verification features (10 more)
        features.extend([
            np.random.beta(2, 8),      # geographic_risk (usually low)
            np.random.beta(2, 8),      # cross_border (usually domestic)
            np.random.beta(1, 9),      # high_risk_country (rarely high risk)
            np.random.beta(2, 8),      # currency_risk (usually low)
            np.random.beta(1, 9),      # sanctions_risk (rarely sanctions)
            np.random.beta(8, 2),      # identity_verification (usually verified)
            np.random.beta(8, 2),      # document_quality (usually good)
            np.random.beta(8, 2),      # biometric_match (usually matches)
            np.random.beta(2, 8),      # suspicious_patterns (usually low)
            np.random.beta(2, 8),      # ml_anomaly_score (usually low)
        ])
        
        # Behavioral and network features (10 more)
        features.extend([
            np.random.beta(2, 8),      # email_risk (usually low)
            np.random.beta(2, 8),      # communication_anomaly (usually low)
            np.random.beta(8, 2),      # business_context (usually legitimate)
            np.random.beta(2, 8),      # urgency_language (usually low)
            np.random.beta(8, 2),      # relationship_history (usually established)
            np.random.beta(8, 2),      # merchant_reputation (usually good)
            np.random.beta(8, 2),      # transaction_context (usually normal)
            np.random.beta(8, 2),      # authorization_method (usually secure)
            np.random.beta(2, 8),      # network_risk (usually low)
            np.random.beta(2, 8),      # consortium_intel (usually low)
        ])
        
        data.append(features + [0])  # 0 = not fraud
    
    # Generate suspicious but legitimate transactions (15%)
    n_suspicious = int(n_samples * 0.15)
    for _ in range(n_suspicious):
        features = []
        
        # Suspicious but legitimate patterns
        features.extend([
            np.random.beta(5, 3),      # amount (higher amounts)
            np.random.beta(4, 4),      # urgency (moderate)
            np.random.beta(6, 3),      # sender_reputation (good but not perfect)
            np.random.beta(4, 4),      # receiver_reputation (moderate)
            np.random.beta(6, 3),      # account_age (established but some new)
            np.random.beta(4, 4),      # velocity (moderate)
            np.random.beta(3, 7),      # unusual_hour (sometimes unusual)
            np.random.beta(2, 8),      # weekend_flag (sometimes weekend)
            np.random.beta(6, 3),      # device_trust (usually trusted)
            np.random.beta(6, 3),      # location_trust (usually trusted)
        ])
        
        # Geographic and verification features
        features.extend([
            np.random.beta(4, 4),      # geographic_risk (moderate)
            np.random.beta(3, 5),      # cross_border (sometimes cross-border)
            np.random.beta(2, 8),      # high_risk_country (rarely high risk)
            np.random.beta(3, 5),      # currency_risk (sometimes foreign)
            np.random.beta(1, 9),      # sanctions_risk (rarely sanctions)
            np.random.beta(6, 3),      # identity_verification (usually verified)
            np.random.beta(6, 3),      # document_quality (usually good)
            np.random.beta(6, 3),      # biometric_match (usually matches)
            np.random.beta(3, 5),      # suspicious_patterns (moderate)
            np.random.beta(3, 5),      # ml_anomaly_score (moderate)
        ])
        
        # Behavioral and network features
        features.extend([
            np.random.beta(3, 5),      # email_risk (moderate)
            np.random.beta(3, 5),      # communication_anomaly (moderate)
            np.random.beta(6, 3),      # business_context (usually legitimate)
            np.random.beta(3, 5),      # urgency_language (moderate)
            np.random.beta(6, 3),      # relationship_history (usually established)
            np.random.beta(6, 3),      # merchant_reputation (usually good)
            np.random.beta(6, 3),      # transaction_context (usually normal)
            np.random.beta(6, 3),      # authorization_method (usually secure)
            np.random.beta(3, 5),      # network_risk (moderate)
            np.random.beta(3, 5),      # consortium_intel (moderate)
        ])
        
        data.append(features + [0])  # 0 = not fraud (but suspicious)
    
    # Generate actual fraud transactions (15%) - including BEC patterns
    n_fraud = int(n_samples * 0.15)
    for i in range(n_fraud):
        features = []
        
        # Create different fraud types
        fraud_type = i % 4
        
        if fraud_type == 0:  # Business Email Compromise (BEC) - like our demo
            features.extend([
                np.random.beta(5, 3),      # amount (business-reasonable, not extreme)
                np.random.beta(6, 3),      # urgency (high but not extreme)
                np.random.beta(4, 4),      # sender_reputation (moderate - compromised but established)
                np.random.beta(2, 6),      # receiver_reputation (low - new account)
                np.random.beta(6, 2),      # account_age (sender established, mixed receiver)
                np.random.beta(5, 3),      # velocity (moderate-high)
                np.random.beta(4, 4),      # unusual_hour (moderate)
                np.random.beta(3, 5),      # weekend_flag (sometimes)
                np.random.beta(4, 4),      # device_trust (moderate - looks normal but compromised)
                np.random.beta(5, 3),      # location_trust (moderate)
            ])
            
            features.extend([
                np.random.beta(4, 4),      # geographic_risk (moderate)
                np.random.beta(3, 5),      # cross_border (sometimes)
                np.random.beta(2, 6),      # high_risk_country (low-moderate)
                np.random.beta(3, 5),      # currency_risk (moderate)
                np.random.beta(1, 9),      # sanctions_risk (low)
                np.random.beta(3, 5),      # identity_verification (moderate - poor receiver verification)
                np.random.beta(4, 4),      # document_quality (moderate)
                np.random.beta(4, 4),      # biometric_match (moderate)
                np.random.beta(6, 3),      # suspicious_patterns (moderate-high)
                np.random.beta(5, 3),      # ml_anomaly_score (moderate-high)
            ])
            
            features.extend([
                np.random.beta(7, 2),      # email_risk (high but not extreme - this is the key BEC signal)
                np.random.beta(6, 3),      # communication_anomaly (moderate-high)
                np.random.beta(4, 4),      # business_context (moderate - sounds legitimate)
                np.random.beta(6, 3),      # urgency_language (moderate-high)
                np.random.beta(4, 4),      # relationship_history (moderate)
                np.random.beta(3, 5),      # merchant_reputation (moderate-low)
                np.random.beta(4, 4),      # transaction_context (moderate)
                np.random.beta(4, 4),      # authorization_method (moderate)
                np.random.beta(5, 3),      # network_risk (moderate-high)
                np.random.beta(5, 3),      # consortium_intel (moderate-high)
            ])
            
        elif fraud_type == 1:  # Account Takeover
            features.extend([
                np.random.beta(4, 4),      # amount (moderate)
                np.random.beta(7, 2),      # urgency (high)
                np.random.beta(2, 6),      # sender_reputation (low - compromised)
                np.random.beta(2, 6),      # receiver_reputation (low)
                np.random.beta(6, 2),      # account_age (established but compromised)
                np.random.beta(8, 2),      # velocity (very high)
                np.random.beta(8, 2),      # unusual_hour (very unusual)
                np.random.beta(4, 4),      # weekend_flag (moderate)
                np.random.beta(1, 9),      # device_trust (very low - new device)
                np.random.beta(1, 9),      # location_trust (very low - new location)
            ])
        
        elif fraud_type == 2:  # Synthetic Identity
            features.extend([
                np.random.beta(3, 5),      # amount (moderate)
                np.random.beta(3, 5),      # urgency (moderate)
                np.random.beta(1, 9),      # sender_reputation (very low - fake)
                np.random.beta(1, 9),      # receiver_reputation (very low - fake)
                np.random.beta(1, 9),      # account_age (very new)
                np.random.beta(5, 3),      # velocity (high)
                np.random.beta(4, 4),      # unusual_hour (moderate)
                np.random.beta(4, 4),      # weekend_flag (moderate)
                np.random.beta(2, 6),      # device_trust (low)
                np.random.beta(3, 5),      # location_trust (moderate)
            ])
        
        else:  # Money Laundering
            features.extend([
                np.random.beta(7, 2),      # amount (high)
                np.random.beta(2, 6),      # urgency (low - structured)
                np.random.beta(4, 4),      # sender_reputation (moderate)
                np.random.beta(4, 4),      # receiver_reputation (moderate)
                np.random.beta(5, 3),      # account_age (established)
                np.random.beta(9, 1),      # velocity (extremely high)
                np.random.beta(5, 3),      # unusual_hour (high)
                np.random.beta(5, 3),      # weekend_flag (high)
                np.random.beta(5, 3),      # device_trust (moderate)
                np.random.beta(3, 5),      # location_trust (moderate)
            ])
        
        # Add remaining features for non-BEC fraud types
        if fraud_type != 0:
            features.extend([np.random.beta(6, 2) for _ in range(10)])  # Geographic/verification
            features.extend([np.random.beta(6, 2) for _ in range(10)])  # Behavioral/network
        
        data.append(features + [1])  # 1 = fraud
    
    # Create DataFrame
    feature_names = [
        'transaction_amount', 'urgency_score', 'sender_reputation', 'receiver_reputation',
        'account_age', 'velocity', 'unusual_hour', 'weekend_flag', 'device_trust', 'location_trust',
        'geographic_risk', 'cross_border', 'high_risk_country', 'currency_risk', 'sanctions_risk',
        'identity_verification', 'document_quality', 'biometric_match', 'suspicious_patterns', 'ml_anomaly_score',
        'email_risk', 'communication_anomaly', 'business_context', 'urgency_language', 'relationship_history',
        'merchant_reputation', 'transaction_context', 'authorization_method', 'network_risk', 'consortium_intel',
        'is_fraud'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    
    # Add bank-specific specializations and realistic variance
    if bank_id == 'bank_A':
        # Bank A specializes in high-value wire transfers but misses some email fraud
        mask = (df['is_fraud'] == 1) & (df['transaction_amount'] > 0.7)
        df.loc[mask, ['geographic_risk', 'cross_border']] *= 1.2  # Better at detecting international
        
        # But sometimes misses email-based fraud (more conservative on communication signals)
        mask = (df['is_fraud'] == 1) & (df['email_risk'] > 0.8)
        df.loc[mask, ['email_risk', 'communication_anomaly']] *= 0.7  # Less sensitive to email fraud
        
    elif bank_id == 'bank_B':
        # Bank B specializes in identity verification but sometimes over-cautious
        mask = (df['is_fraud'] == 1) & (df['identity_verification'] < 0.3)
        df.loc[mask, ['document_quality', 'biometric_match']] *= 1.2  # Better at identity fraud
        
        # Add some false positives for legitimate high-value transactions
        mask = (df['is_fraud'] == 0) & (df['transaction_amount'] > 0.8)
        df.loc[mask, ['suspicious_patterns']] *= 1.3  # Sometimes flags large legitimate transactions
        
    elif bank_id == 'bank_C':
        # Bank C specializes in network patterns but weaker on traditional fraud indicators
        mask = (df['is_fraud'] == 1) & (df['network_risk'] > 0.6)
        df.loc[mask, ['email_risk', 'communication_anomaly', 'consortium_intel']] *= 1.2  # Better at BEC/network fraud
        
        # But sometimes misses traditional financial fraud patterns
        mask = (df['is_fraud'] == 1) & (df['transaction_amount'] > 0.9)
        df.loc[mask, ['transaction_amount', 'velocity']] *= 0.8  # Less sensitive to amount-based patterns
    
    # Add realistic noise and variance to make banks more diverse
    np.random.seed(42 + hash(bank_id) % 1000)
    noise_factor = 0.1  # 10% noise
    
    # Add bank-specific biases
    for col in feature_cols:
        if bank_id == 'bank_A' and 'geo' in col.lower():
            df[col] *= (1 + np.random.normal(0.1, 0.05))  # Slightly better at geographic
        elif bank_id == 'bank_B' and 'identity' in col.lower():
            df[col] *= (1 + np.random.normal(0.1, 0.05))  # Slightly better at identity
        elif bank_id == 'bank_C' and ('email' in col.lower() or 'network' in col.lower()):
            df[col] *= (1 + np.random.normal(0.1, 0.05))  # Slightly better at network/email
        else:
            # Add small random variations for other features
            df[col] *= (1 + np.random.normal(0, noise_factor * 0.5))
    
    # Ensure all values are in [0,1] range
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    df[feature_cols] = df[feature_cols].clip(0, 1)
    
    return df

def retrain_consortium_models():
    """Retrain all consortium models with realistic fraud patterns"""
    print("🔄 Retraining Consortium Models with Demo-Aligned Patterns")
    print("=" * 60)
    
    # Clean up old models
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        model_path = f'models/{bank_id}_model.pkl'
        metadata_path = f'models/{bank_id}_metadata.json'
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"🗑️  Removed old model: {model_path}")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"🗑️  Removed old metadata: {metadata_path}")
    
    banks = []
    
    # Generate and train each bank
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        print(f"\n🏦 Training {bank_id}...")
        
        # Generate realistic training data
        print(f"  📊 Generating realistic fraud patterns...")
        training_data = generate_realistic_fraud_training_data(bank_id, 2000)
        
        # Save training data
        data_path = f'{bank_id}_data.csv'
        training_data.to_csv(data_path, index=False)
        print(f"  💾 Saved training data: {data_path}")
        print(f"  📈 Fraud rate: {training_data['is_fraud'].mean():.1%}")
        
        # Create and train bank simulator
        bank = BankSimulator(bank_id, data_path)
        print(f"  🤖 Training XGBoost model...")
        bank.train_local_model('xgboost')
        
        banks.append(bank)
        print(f"  ✅ {bank_id} trained successfully (confidence: {bank.model_confidence:.3f})")
    
    print(f"\n🎯 Testing Demo BEC Scenario...")
    
    # Test the demo case
    consortium = ConsortiumComparisonService()
    for bank in banks:
        consortium.register_bank(bank.bank_id, bank)
    
    # Our demo BEC features - Test 2 from variance testing (realistic disagreement)
    # Bank A: 0.002 (very low - would approve - misses email sophistication)
    # Bank B: 0.153 (moderate - flags for review - notices identity issues)  
    # Bank C: 0.046 (low-moderate - slight concern - sees some email patterns)
    bec_demo = [0.35, 0.45, 0.75, 0.40, 0.85, 0.35, 0.40, 0.70, 0.80, 0.90,  # Low amounts, high trust
                0.25, 0.35, 0.15, 0.30, 0.10, 0.70, 0.85, 0.90, 0.40, 0.35,  # Low geo risk, good identity  
                0.75, 0.35, 0.65, 0.55, 0.85, 0.75, 0.70, 0.75, 0.45, 0.40]   # Moderate email, low network
    
    result = consortium.generate_comparison_score(bec_demo)
    
    print(f"\n📊 Demo BEC Results:")
    if 'final_comparison_score' in result:
        print(f"  🎯 Final Score: {result['final_comparison_score']:.3f}")
    if 'individual_scores' in result:
        individual_scores = result['individual_scores']
        if isinstance(individual_scores, list):
            print(f"  🏦 Individual Scores: {[f'{float(s):.3f}' for s in individual_scores]}")
        else:
            print(f"  🏦 Individual Scores: {individual_scores}")
    if 'consensus_score' in result:
        print(f"  🤝 Consensus: {result['consensus_score']:.3f}")
    if 'variance_score' in result:
        print(f"  📊 Variance: {result['variance_score']:.3f}")
    if 'network_anomaly_score' in result:
        print(f"  🌐 Network Anomaly: {result['network_anomaly_score']:.3f}")
    if 'recommendation' in result:
        print(f"  ⚖️  Recommendation: {result['recommendation']}")
    if 'flagging_banks' in result:
        print(f"  🚩 Flagging Banks: {result['flagging_banks']}")
    
    print(f"\n✅ Model retraining complete! Demo scenario should now show proper fraud detection.")
    
    return consortium

if __name__ == "__main__":
    retrain_consortium_models()
