"""
Consortium Comparison Score Prototype
Enhanced version with comprehensive comparison scoring leveraging diverse data perspectives
across financial institutions using consistent XGBoost methodology
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import hashlib
from typing import Dict, List, Any
import json
import pickle
import os

# Optional: Import XGBoost (uncomment if installed)
# import xgboost as xgb

class BankSimulator:
    def __init__(self, bank_id: str, data_path: str, model_path: str = None):
        self.bank_id = bank_id
        self.data = pd.read_csv(data_path) if data_path else None
        self.model = None
        self.model_confidence = 0.0
        self.model_path = model_path or f"models/{bank_id}_model.pkl"
        self.metadata_path = model_path.replace('.pkl', '_metadata.json') if model_path else f"models/{bank_id}_metadata.json"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def train_local_model(self, model_type: str = 'xgboost'):
        """Train fraud detection model on local data"""
        if self.data is None:
            raise ValueError("No training data available")
            
        # Prepare features (assuming BAF dataset structure)
        feature_cols = [col for col in self.data.columns 
                       if col not in ['is_fraud', 'ENTITY_ID', 'EVENT_ID', 'assigned_bank']]
        
        X = self.data[feature_cols]
        y = self.data['is_fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
                # Calculate scale_pos_weight for class imbalance
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1
                )
            except ImportError:
                print(f"XGBoost not available for {self.bank_id}, falling back to Random Forest")
                return self.train_local_model('random_forest')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Calculate model confidence based on test accuracy
        self.model_confidence = self.model.score(X_test, y_test)
        
        # Save the trained model and metadata
        self.save_model(model_type)
        
        return self.model_confidence
    
    def save_model(self, model_type: str):
        """Save trained model and metadata to disk"""
        try:
            # Save the model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save metadata
            metadata = {
                'bank_id': self.bank_id,
                'model_type': model_type,
                'model_confidence': self.model_confidence,
                'model_class': type(self.model).__name__,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"  Model saved: {self.model_path}")
            print(f"  Metadata saved: {self.metadata_path}")
            
        except Exception as e:
            print(f"  Warning: Failed to save model for {self.bank_id}: {e}")
    
    def load_model(self):
        """Load trained model and metadata from disk"""
        try:
            # Load the model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_confidence = metadata['model_confidence']
            
            print(f"  Model loaded: {self.model_path}")
            print(f"  Model confidence: {self.model_confidence:.3f}")
            return True
            
        except FileNotFoundError:
            print(f"  No saved model found for {self.bank_id}")
            return False
        except Exception as e:
            print(f"  Error loading model for {self.bank_id}: {e}")
            return False
    
    def model_exists(self) -> bool:
        """Check if saved model exists"""
        return os.path.exists(self.model_path) and os.path.exists(self.metadata_path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about saved model"""
        if not self.model_exists():
            return {'exists': False}
        
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add file information
            model_size = os.path.getsize(self.model_path)
            metadata.update({
                'exists': True,
                'model_file_size_kb': round(model_size / 1024, 2),
                'model_path': self.model_path
            })
            
            return metadata
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    
    def predict_risk_score(self, transaction: List[float]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment for a transaction"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get fraud probability
        fraud_prob = self.model.predict_proba([transaction])[0][1]
        
        # Get feature importance for explanation
        feature_importance = self.model.feature_importances_
        
        return {
            'bank_id': self.bank_id,
            'risk_score': fraud_prob,
            'model_confidence': self.model_confidence,
            'risk_bucket': self._get_risk_bucket(fraud_prob),
            'top_features': self._get_top_features(transaction, feature_importance)
        }
    
    def _get_risk_bucket(self, score: float) -> str:
        """Convert numeric score to risk bucket"""
        if score < 0.3:
            return 'low'
        elif score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _get_top_features(self, transaction: List[float], importance: np.ndarray) -> List[Dict]:
        """Get top contributing features for explanation"""
        top_indices = np.argsort(importance)[-3:][::-1]
        return [
            {
                'feature_index': int(idx),
                'importance': float(importance[idx]),
                'value': float(transaction[idx])
            }
            for idx in top_indices
        ]

class ConsortiumComparisonService:
    def __init__(self):
        self.participating_banks = {}
        self.transaction_history = []
        
    def register_bank(self, bank_id: str, bank_simulator: BankSimulator):
        """Register a bank for consortium scoring with its unique data perspective"""
        self.participating_banks[bank_id] = bank_simulator
        print(f"Registered {bank_id} with model confidence: {bank_simulator.model_confidence:.3f} (unique data perspective)")
        
    def generate_comparison_score(self, transaction_features: List[float]) -> Dict[str, Any]:
        """Generate comprehensive comparison score leveraging diverse institutional data perspectives"""
        individual_assessments = {}
        
        # Collect assessments from banks with diverse data perspectives
        for bank_id, bank in self.participating_banks.items():
            try:
                assessment = bank.predict_risk_score(transaction_features)
                individual_assessments[bank_id] = assessment
            except Exception as e:
                print(f"Bank {bank_id} scoring failed: {e}")
                
        if not individual_assessments:
            return self._empty_comparison_score()
            
        # Calculate comparison score components
        comparison_score = self._calculate_comparison_score(individual_assessments)
        
        # Store for historical analysis
        self.transaction_history.append({
            'transaction_features': transaction_features,
            'individual_assessments': individual_assessments,
            'comparison_score': comparison_score
        })
        
        return comparison_score
    
    def _calculate_comparison_score(self, assessments: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate comprehensive comparison score leveraging diverse data perspectives"""
        scores = [assessment['risk_score'] for assessment in assessments.values()]
        confidences = [assessment['model_confidence'] for assessment in assessments.values()]
        
        # Weighted consensus from diverse institutional data sources
        consensus_score = np.average(scores, weights=confidences)
        variance_score = np.var(scores)
        
        # Network anomaly detection across data perspectives
        network_anomaly_score = self._detect_network_anomaly(scores)
        
        # Final comparison score combines multiple factors
        final_comparison_score = self._calculate_final_score(
            consensus_score, variance_score, network_anomaly_score
        )
        
        # Determine confidence and recommendation
        confidence_level = self._get_confidence_level(variance_score, len(scores))
        recommendation = self._get_recommendation(final_comparison_score, confidence_level)
        
        # Count flagging banks
        flagging_banks = [
            bank_id for bank_id, assessment in assessments.items() 
            if assessment['risk_score'] > 0.5
        ]
        
        return {
            'individual_scores': {
                bank_id: assessment['risk_score'] 
                for bank_id, assessment in assessments.items()
            },
            'individual_assessments': assessments,
            'consensus_score': consensus_score,
            'variance_score': variance_score,
            'network_anomaly_score': network_anomaly_score,
            'final_comparison_score': final_comparison_score,
            'confidence_level': confidence_level,
            'flagging_banks_count': len(flagging_banks),
            'flagging_banks': flagging_banks,
            'recommendation': recommendation,
            'participating_banks': len(assessments)
        }
    
    def _detect_network_anomaly(self, scores: List[float]) -> float:
        """Detect cross-institutional anomaly patterns using diverse data perspectives"""
        if len(scores) < 2:
            return 0.0
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        # Cross-institutional fraud detection scenarios
        
        # High consensus on high risk = strong cross-institutional fraud signal
        if mean_score > 0.8 and std_score < 0.1:
            return 0.95  # All banks agree on high risk - likely cross-institutional fraud
        
        # Medium-high consensus = good network intelligence
        elif mean_score > 0.6 and std_score < 0.15:
            return 0.75  # Strong consensus across diverse data sources
        
        # High variance with high max = one bank detects specialized fraud
        elif max_score > 0.8 and std_score > 0.3:
            return 0.85  # One bank's specialty detected something others missed
        
        # Moderate consensus = standard network benefit
        elif mean_score > 0.4 and std_score < 0.2:
            return 0.6   # Moderate consensus across data perspectives
        
        # Low variance, low mean = likely legitimate transaction
        elif mean_score < 0.3 and std_score < 0.1:
            return 0.1   # All banks agree it's low risk
        
        else:
            return 0.3   # Mixed signals, moderate network value
    
    def _calculate_final_score(self, consensus: float, variance: float, anomaly: float) -> float:
        """Calculate final comparison score"""
        # Weight factors based on importance
        consensus_weight = 0.6
        anomaly_weight = 0.3
        variance_penalty = 0.1
        
        # Higher variance reduces confidence
        variance_factor = max(0, 1 - variance * 2)
        
        final_score = (
            consensus * consensus_weight + 
            anomaly * anomaly_weight - 
            variance * variance_penalty
        ) * variance_factor
        
        return np.clip(final_score, 0.0, 1.0)
    
    def _get_confidence_level(self, variance: float, num_banks: int) -> str:
        """Determine confidence level based on agreement across diverse data perspectives"""
        if num_banks >= 3 and variance < 0.1:
            return 'high'  # Strong consensus across diverse data sources
        elif num_banks >= 2 and variance < 0.2:
            return 'medium'  # Moderate agreement between data perspectives
        else:
            return 'low'
    
    def _get_recommendation(self, score: float, confidence: str) -> str:
        """Generate recommendation based on score and confidence"""
        if confidence == 'high':
            if score > 0.7:
                return 'block'
            elif score > 0.4:
                return 'review'
            else:
                return 'approve'
        elif confidence == 'medium':
            if score > 0.8:
                return 'block'
            elif score > 0.3:
                return 'review'
            else:
                return 'approve'
        else:  # low confidence
            if score > 0.9:
                return 'block'
            else:
                return 'review'
    
    def _empty_comparison_score(self) -> Dict[str, Any]:
        """Return empty comparison score when no banks available"""
        return {
            'individual_scores': {},
            'individual_assessments': {},
            'consensus_score': 0.0,
            'variance_score': 0.0,
            'network_anomaly_score': 0.0,
            'final_comparison_score': 0.0,
            'confidence_level': 'low',
            'flagging_banks_count': 0,
            'flagging_banks': [],
            'recommendation': 'review',
            'participating_banks': 0
        }
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Generate analytics summary from transaction history"""
        if not self.transaction_history:
            return {'message': 'No transaction history available'}
            
        total_transactions = len(self.transaction_history)
        high_risk_transactions = sum(
            1 for t in self.transaction_history 
            if t['comparison_score']['final_comparison_score'] > 0.7
        )
        
        blocked_transactions = sum(
            1 for t in self.transaction_history 
            if t['comparison_score']['recommendation'] == 'block'
        )
        
        avg_consensus = np.mean([
            t['comparison_score']['consensus_score'] 
            for t in self.transaction_history
        ])
        
        return {
            'total_transactions': total_transactions,
            'high_risk_transactions': high_risk_transactions,
            'blocked_transactions': blocked_transactions,
            'block_rate': blocked_transactions / total_transactions,
            'average_consensus_score': avg_consensus,
            'average_participating_banks': np.mean([
                t['comparison_score']['participating_banks'] 
                for t in self.transaction_history
            ])
        }

def generate_synthetic_transaction() -> List[float]:
    """Generate a synthetic transaction for testing"""
    # Generate 30 features similar to BAF dataset
    np.random.seed()
    return np.random.normal(0.5, 0.2, 30).tolist()

def generate_cross_institutional_fraud_scenarios():
    """
    Generate specific fraud scenarios that demonstrate consortium value
    Each scenario represents a real-world fraud type where consortium collaboration
    provides significant advantages over individual bank detection
    """
    scenarios = []
    
    # DEMO SCENARIO: Business Email Compromise - ABC Manufacturing Case
    # Real-world example: CEO impersonation for $485K wire to fake supplier
    bec_demo = np.random.normal(0.5, 0.1, 30).tolist()
    bec_demo[0] = 0.75    # High amount ($485K - significant but not extreme)
    bec_demo[5] = 0.85    # High urgency/velocity (Friday 4:47 PM rush)
    bec_demo[10] = 0.35   # Low geographic risk (USA to USA)
    bec_demo[15] = 0.25   # Low sender identity risk (legitimate business)
    bec_demo[16] = 0.95   # Very high receiver identity risk (3-day-old account)
    bec_demo[18] = 0.80   # Business account patterns
    bec_demo[22] = 0.90   # Email communication anomalies (spoofed CEO)
    bec_demo[25] = 0.85   # Network patterns (same scheme at other banks)
    bec_demo[28] = 0.95   # Urgency indicators (rush payment, confidential)
    bec_demo[29] = 0.70   # Timing patterns (end of business day)
    scenarios.append((
        "🎯 DEMO: CEO Fraud - ABC Manufacturing ($485K)", 
        bec_demo,
        "ABC Manufacturing's CFO receives urgent email from 'CEO' requesting $485K wire to "
        "'new strategic supplier' for confidential acquisition. Bank A sees normal business "
        "transaction. Bank B notices recipient account opened 3 days ago. Bank C recognizes "
        "identical pattern affecting 5 other companies this week. Classic BEC fraud caught "
        "through consortium intelligence that no single bank could detect alone."
    ))
    
    # Scenario 1: High-Value Wire Transfer Fraud
    # Why consortium helps: Bank A specializes in high-value transaction analysis,
    # but fraudsters use legitimate accounts from other banks as intermediaries
    wire_fraud = np.random.normal(0.3, 0.1, 30).tolist()
    wire_fraud[0] = 0.95   # Very high transaction amount ($10M+)
    wire_fraud[10] = 0.88  # High geographic risk (offshore destination)
    wire_fraud[15] = 0.25  # Low identity risk (legitimate accounts compromised)
    wire_fraud[18] = 0.85  # Business account patterns (corporate wire)
    wire_fraud[22] = 0.70  # Communication anomalies (urgent requests)
    scenarios.append((
        "High-Value Wire Fraud ($50M Corporate)", 
        wire_fraud,
        "Large wire transfer using compromised corporate credentials. Bank A's high-value "
        "detection algorithms excel here, while other banks might miss the pattern if they "
        "don't typically handle such large amounts. Cross-bank validation confirms legitimacy."
    ))
    
    # Scenario 2: Synthetic Identity Network
    # Why consortium helps: Bank B has advanced identity verification, but synthetic
    # identities span multiple institutions using different data combinations
    synthetic_identity = np.random.normal(0.4, 0.1, 30).tolist()
    synthetic_identity[0] = 0.45   # Moderate amount (not suspicious by amount alone)
    synthetic_identity[15] = 0.92  # Very high identity inconsistency
    synthetic_identity[20] = 0.85  # Suspicious account patterns (new accounts)
    synthetic_identity[10] = 0.35  # Low geographic risk (domestic)
    synthetic_identity[25] = 0.75  # Network patterns (coordinated applications)
    scenarios.append((
        "Synthetic Identity Fraud (Coordinated Ring)", 
        synthetic_identity,
        "Criminal network creates fake identities using real SSNs + fake names/addresses. "
        "Bank B's identity verification systems detect inconsistencies, but individual banks "
        "only see fragments. Consortium reveals the full network pattern across institutions."
    ))
    
    # Scenario 3: Money Mule Network
    # Why consortium helps: Bank C specializes in velocity analysis, but mule
    # networks deliberately spread transactions across multiple banks
    mule_network = np.random.normal(0.35, 0.1, 30).tolist()
    mule_network[5] = 0.95    # Extreme velocity pattern (rapid consecutive transfers)
    mule_network[25] = 0.90   # Strong network connectivity (same recipients)
    mule_network[0] = 0.55    # Moderate individual amounts (under reporting thresholds)
    mule_network[15] = 0.40   # Legitimate-looking identities (recruited victims)
    mule_network[12] = 0.80   # Structured timing patterns
    scenarios.append((
        "Money Mule Network (Human Trafficking)", 
        mule_network,
        "Human trafficking ring recruits victims to move money rapidly across banks. "
        "Bank C's velocity algorithms detect the speed, but individual banks can't see "
        "the full network. Consortium analysis reveals coordinated timing and recipients."
    ))
    
    # Scenario 4: Business Email Compromise (BEC)
    # Why consortium helps: Sophisticated social engineering affects multiple banks
    # simultaneously, requiring cross-institutional intelligence
    bec_fraud = np.random.normal(0.6, 0.1, 30).tolist()
    bec_fraud[0] = 0.78    # High but not extreme amount ($500K)
    bec_fraud[10] = 0.72   # International component (fake supplier overseas)
    bec_fraud[18] = 0.85   # Business account patterns (corporate accounts)
    bec_fraud[22] = 0.80   # Email/communication anomalies (spoofed emails)
    bec_fraud[28] = 0.88   # Urgency indicators (rush payment requests)
    scenarios.append((
        "Business Email Compromise (CEO Fraud)", 
        bec_fraud,
        "Criminals impersonate CEO requesting urgent wire to 'confidential supplier'. "
        "Affects multiple banks as companies bank with different institutions. "
        "Consortium detects similar patterns across institutions simultaneously."
    ))
    
    # Scenario 5: Cross-Border Laundering Ring
    # Why consortium helps: Sophisticated laundering operations use multiple
    # jurisdictions and institutions to obscure money trails
    laundering_ring = np.random.normal(0.5, 0.1, 30).tolist()
    laundering_ring[10] = 0.95  # Multiple countries (complex routing)
    laundering_ring[5] = 0.75   # Structured timing (coordinated transfers)
    laundering_ring[25] = 0.88  # Network patterns (shell company connections)
    laundering_ring[12] = 0.82  # Currency patterns (conversion indicators)
    laundering_ring[27] = 0.85  # Trade-based patterns (fake invoices)
    scenarios.append((
        "Cross-Border Laundering (Trade-Based)", 
        laundering_ring,
        "International drug cartel uses fake trade invoices to move money through "
        "multiple banks and countries. Individual banks see legitimate-looking trade "
        "finance. Consortium reveals coordinated timing and inflated invoice patterns."
    ))
    
    # Scenario 6: Cryptocurrency Conversion Fraud
    # Why consortium helps: Criminals move through traditional banking before
    # converting to crypto, requiring cross-institutional tracking
    crypto_fraud = np.random.normal(0.45, 0.1, 30).tolist()
    crypto_fraud[0] = 0.70     # High amount heading to crypto
    crypto_fraud[5] = 0.85     # High velocity (quick conversion)
    crypto_fraud[14] = 0.90    # Crypto exchange indicators
    crypto_fraud[26] = 0.80    # Privacy coin patterns
    crypto_fraud[29] = 0.88    # Mixing service indicators
    scenarios.append((
        "Cryptocurrency Laundering (Ransomware)", 
        crypto_fraud,
        "Ransomware group converts stolen funds to privacy coins through multiple banks. "
        "Individual banks see normal crypto exchange activity. Consortium tracking "
        "reveals coordinated conversion patterns from crime proceeds."
    ))
    
    return scenarios

def create_synthetic_bank_data(bank_id: str, num_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic bank data with unique characteristics for diverse data perspectives"""
    np.random.seed(hash(bank_id) % 2**32)  # Each bank gets unique data distribution
    
    # Generate features with bank-specific patterns for data diversity
    data = []
    for i in range(num_samples):
        # Create feature vector with bank-specific distribution
        features = np.random.normal(0.5, 0.3, 30)
        
        # Bank-specific specialization patterns (each bank sees different fraud types better)
        if bank_id == 'bank_A':
            # Bank A better at detecting high-value wire fraud
            if np.random.random() < 0.15:  # 15% high-value scenarios
                features[0] = np.random.uniform(0.8, 1.0)  # High amount feature
                features[10] = np.random.uniform(0.7, 0.9)  # Geographic risk
        elif bank_id == 'bank_B':
            # Bank B better at detecting synthetic identity fraud
            if np.random.random() < 0.12:  # 12% synthetic identity scenarios
                features[15] = np.random.uniform(0.8, 1.0)  # Identity inconsistency
                features[20] = np.random.uniform(0.7, 0.9)  # Account age suspicion
        elif bank_id == 'bank_C':
            # Bank C better at detecting mule network patterns
            if np.random.random() < 0.18:  # 18% mule network scenarios
                features[5] = np.random.uniform(0.8, 1.0)   # Velocity feature
                features[25] = np.random.uniform(0.7, 0.9)  # Network pattern
        
        # Create fraud label with bank-specific patterns for data diversity
        fraud_score = np.sum(features[:5]) / 5  # Use first 5 features
        is_fraud = 1 if fraud_score > 0.7 or np.random.random() < 0.05 else 0
        
        row = {f'feature_{j}': features[j] for j in range(30)}
        row.update({
            'ENTITY_ID': f'{bank_id}_entity_{i}',
            'EVENT_ID': f'{bank_id}_event_{i}',
            'is_fraud': is_fraud,
            'assigned_bank': bank_id
        })
        data.append(row)
    
    return pd.DataFrame(data)

def compare_models_performance():
    """Compare model algorithms on identical data (for validation purposes)"""
    print("Model Algorithm Comparison")
    print("=" * 40)
    
    # Create test dataset for algorithm comparison
    bank_data = create_synthetic_bank_data('test_bank', 2000)
    bank_data.to_csv('test_bank_data.csv', index=False)
    
    results = {}
    
    for model_type in ['xgboost', 'random_forest']:
        print(f"\nTesting {model_type}...")
        
        bank = BankSimulator('test_bank', 'test_bank_data.csv')
        try:
            accuracy = bank.train_local_model(model_type)
            results[model_type] = {
                'accuracy': accuracy,
                'model_confidence': bank.model_confidence,
                'model_type': type(bank.model).__name__
            }
            print(f"  {model_type} accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"  {model_type} failed: {e}")
            results[model_type] = {'error': str(e)}
    
    return results

def train_consortium_models(bank_data: Dict[str, pd.DataFrame], model_types: List[str], force_retrain: bool = False):
    """Train and save models for all banks"""
    print("\n3. Training bank models on diverse data...")
    banks = {}
    
    for i, bank_id in enumerate(['bank_A', 'bank_B', 'bank_C']):
        model_type = model_types[i]
        bank = BankSimulator(bank_id, f'{bank_id}_data.csv')
        
        # Check if model already exists
        if bank.model_exists() and not force_retrain:
            print(f"{bank_id} ({model_type}): Loading existing model...")
            bank.load_model()
        else:
            print(f"{bank_id} ({model_type}): Training new model...")
            accuracy = bank.train_local_model(model_type)
            print(f"  Training accuracy: {accuracy:.3f} (trained on unique data perspective)")
        
        banks[bank_id] = bank
    
    return banks

def run_consortium_inference(banks: Dict[str, BankSimulator]):
    """Run inference using pre-trained models"""
    print("\n4. Initializing consortium comparison service...")
    consortium = ConsortiumComparisonService()
    for bank_id, bank in banks.items():
        consortium.register_bank(bank_id, bank)
    
    # Test with multiple transactions using diverse data perspectives
    print("\n5. Testing consortium comparison scoring with diverse data perspectives...")
    test_transactions = [generate_synthetic_transaction() for _ in range(5)]
    
    for i, transaction in enumerate(test_transactions):
        print(f"\nTransaction {i+1}:")
        result = consortium.generate_comparison_score(transaction)
        
        print(f"  Individual Scores: {result['individual_scores']}")
        print(f"  Consensus Score: {result['consensus_score']:.3f}")
        print(f"  Variance Score: {result['variance_score']:.3f}")
        print(f"  Network Anomaly: {result['network_anomaly_score']:.3f}")
        print(f"  Final Comparison Score: {result['final_comparison_score']:.3f}")
        print(f"  Confidence: {result['confidence_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Flagging Banks: {result['flagging_banks']}")
    
    return consortium

def main():
    """Main demonstration function showcasing consortium fraud detection with diverse data perspectives"""
    print("Consortium Comparison Score Prototype")
    print("=" * 40)
    
    # First validate algorithm choice with identical data
    model_comparison = compare_models_performance()
    
    # Create synthetic data for 3 banks with diverse data characteristics
    print("\n1. Creating synthetic bank data with diverse perspectives...")
    bank_data = {}
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        bank_data[bank_id] = create_synthetic_bank_data(bank_id, 1000)
        fraud_rate = bank_data[bank_id]['is_fraud'].mean()
        print(f"{bank_id}: {len(bank_data[bank_id])} transactions, {fraud_rate:.1%} fraud rate (unique data perspective)")
    
    # Save data to files
    for bank_id, data in bank_data.items():
        data.to_csv(f'{bank_id}_data.csv', index=False)
    
    # Initialize banks with same models but diverse data
    print("\n2. Initializing bank simulators with model persistence...")
    model_types = ['xgboost', 'xgboost', 'xgboost']  # All banks use XGBoost for consistent methodology
    
    for i, bank_id in enumerate(['bank_A', 'bank_B', 'bank_C']):
        model_type = model_types[i]
        print(f"  {bank_id} configured for {model_type} (diverse data perspective)")
    
    # Training Phase: Train and save models
    banks = train_consortium_models(bank_data, model_types, force_retrain=False)
    
    # Inference Phase: Load models and run consortium scoring
    consortium = run_consortium_inference(banks)
    
    # Analytics summary
    print("\n6. Analytics Summary:")
    summary = consortium.get_analytics_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Model comparison summary
    print("\n7. Algorithm Performance Summary (for validation):")
    for model_type, results in model_comparison.items():
        if 'accuracy' in results:
            print(f"  {model_type}: {results['accuracy']:.4f} accuracy ({results['model_type']}) - validates algorithm choice")
        else:
            print(f"  {model_type}: {results.get('error', 'Failed')}")

def main_training_only():
    """Training-only mode for model preparation"""
    print("Consortium Model Training Mode")
    print("=" * 30)
    
    # Create synthetic data for 3 banks with diverse data characteristics
    print("\n1. Creating synthetic bank data with diverse perspectives...")
    bank_data = {}
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        bank_data[bank_id] = create_synthetic_bank_data(bank_id, 1000)
        fraud_rate = bank_data[bank_id]['is_fraud'].mean()
        print(f"{bank_id}: {len(bank_data[bank_id])} transactions, {fraud_rate:.1%} fraud rate")
    
    # Save data to files
    for bank_id, data in bank_data.items():
        data.to_csv(f'{bank_id}_data.csv', index=False)
    
    # Training Phase: Train and save models
    model_types = ['xgboost', 'xgboost', 'xgboost']
    banks = train_consortium_models(bank_data, model_types, force_retrain=True)
    
    print("\nTraining complete! Models saved and ready for inference.")

def main_inference_only():
    """Inference-only mode using pre-trained models"""
    print("Consortium Inference Mode")
    print("=" * 25)
    
    # Load pre-trained models
    print("\n1. Loading pre-trained models...")
    banks = {}
    
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        bank = BankSimulator(bank_id, f'{bank_id}_data.csv')
        if bank.load_model():
            banks[bank_id] = bank
        else:
            print(f"Error: No trained model found for {bank_id}. Please run training first.")
            return
    
    # Inference Phase: Run consortium scoring with enhanced scenarios
    consortium = run_consortium_inference_enhanced(banks)
    
    # Analytics summary
    print("\n6. Analytics Summary:")
    summary = consortium.get_analytics_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

def run_consortium_inference_enhanced(banks):
    """Enhanced consortium inference with cross-institutional fraud scenarios"""
    # Initialize consortium
    consortium = ConsortiumComparisonService()
    
    for bank_id, bank in banks.items():
        consortium.register_bank(bank_id, bank)
    
    # Test with cross-institutional fraud scenarios
    print("\n4. Testing consortium comparison scoring with cross-institutional fraud scenarios...")
    
    # Test specific fraud scenarios that demonstrate consortium value
    fraud_scenarios = generate_cross_institutional_fraud_scenarios()
    
    print("\n=== Cross-Institutional Fraud Detection Analysis ===")
    
    for scenario_data in fraud_scenarios:
        if len(scenario_data) == 3:
            scenario_name, transaction_features, explanation = scenario_data
        else:
            scenario_name, transaction_features = scenario_data
            explanation = "No detailed explanation available"
            
        print(f"\n🚨 {scenario_name}:")
        print(f"📝 Scenario: {explanation}")
        
        result = consortium.generate_comparison_score(transaction_features)
        
        print(f"\n📊 Analysis Results:")
        print(f"  Individual Scores: {result['individual_scores']}")
        print(f"  Consensus Score: {result['consensus_score']:.3f}")
        print(f"  Variance Score: {result['variance_score']:.3f}")
        print(f"  Network Anomaly: {result['network_anomaly_score']:.3f}")
        print(f"  Final Comparison Score: {result['final_comparison_score']:.3f}")
        print(f"  Confidence: {result['confidence_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        
        consortium_value = 'HIGH' if result['network_anomaly_score'] > 0.7 else 'MEDIUM' if result['network_anomaly_score'] > 0.4 else 'LOW'
        print(f"  🎯 Consortium Value: {consortium_value}")
        print(f"  💡 Cross-Institutional Insight: {get_consortium_insight(result, scenario_name)}")
        
        # Add interpretation
        if result['variance_score'] > 0.3:
            print(f"  🔍 Pattern: High variance suggests specialized bank detection")
        elif result['consensus_score'] > 0.7:
            print(f"  🔍 Pattern: Strong consensus indicates clear fraud signature")
        elif result['network_anomaly_score'] > 0.8:
            print(f"  🔍 Pattern: Network anomaly suggests cross-institutional coordination")
    
    return consortium

def get_consortium_insight(result, scenario_name):
    """Generate detailed insight about consortium value for specific scenarios"""
    consensus = result['consensus_score']
    variance = result['variance_score']
    network_anomaly = result['network_anomaly_score']
    individual_scores = result['individual_scores']
    
    # Analyze score patterns
    max_score = max(individual_scores.values()) if individual_scores else 0
    min_score = min(individual_scores.values()) if individual_scores else 0
    score_range = max_score - min_score
    
    # Scenario-specific insights
    if "Wire Fraud" in scenario_name:
        if network_anomaly > 0.7:
            return f"Bank A's high-value algorithms detected {max_score:.2f} risk while others saw {min_score:.2f} - specialist knowledge crucial"
        else:
            return "High-value transaction patterns require specialized detection capabilities"
    
    elif "Synthetic Identity" in scenario_name:
        if variance > 0.3:
            return f"Bank B's identity verification caught what others missed (variance: {variance:.2f}) - unique data perspective valuable"
        else:
            return "Identity fraud detection benefits from diverse verification approaches"
    
    elif "Mule Network" in scenario_name:
        if consensus > 0.7:
            return f"Strong consensus ({consensus:.2f}) confirms velocity patterns across all banks - network effect clear"
        else:
            return "Velocity-based detection enhanced through cross-bank pattern analysis"
    
    elif "Email Compromise" in scenario_name:
        if consensus > 0.6:
            return f"Business fraud pattern recognized across {len(individual_scores)} institutions - shared intelligence valuable"
        else:
            return "Social engineering attacks benefit from cross-institutional pattern recognition"
    
    elif "Laundering" in scenario_name:
        if network_anomaly > 0.8:
            return f"Complex multi-bank patterns detected (anomaly: {network_anomaly:.2f}) - individual banks would miss connections"
        else:
            return "Money laundering schemes require consortium-level intelligence to detect full patterns"
    
    elif "Cryptocurrency" in scenario_name:
        if consensus > 0.6:
            return f"Crypto conversion patterns visible across traditional banking network - consortium tracking essential"
        else:
            return "Digital asset fraud requires traditional banking surveillance coordination"
    
    # General insights based on scoring patterns
    elif consensus > 0.8 and variance < 0.1:
        return f"Unanimous high risk ({consensus:.2f}) - clear fraud signature recognized by all banks"
    elif variance > 0.4:
        return f"Specialized detection advantage (range: {score_range:.2f}) - one bank's expertise proves valuable"
    elif network_anomaly > 0.7:
        return f"Cross-institutional coordination detected - pattern invisible to individual banks"
    elif consensus > 0.6:
        return f"Strong multi-bank agreement ({consensus:.2f}) increases confidence in decision"
    else:
        return "Consortium analysis provides enhanced risk assessment through diverse perspectives"
    """Main demonstration function showcasing consortium fraud detection with diverse data perspectives"""
    print("Consortium Comparison Score Prototype")
    print("=" * 40)
    
    # First validate algorithm choice with identical data
    model_comparison = compare_models_performance()
    
    # Create synthetic data for 3 banks with diverse data characteristics
    print("\n1. Creating synthetic bank data with diverse perspectives...")
    bank_data = {}
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        bank_data[bank_id] = create_synthetic_bank_data(bank_id, 1000)
        fraud_rate = bank_data[bank_id]['is_fraud'].mean()
        print(f"{bank_id}: {len(bank_data[bank_id])} transactions, {fraud_rate:.1%} fraud rate (unique data perspective)")
    
    # Save data to files
    for bank_id, data in bank_data.items():
        data.to_csv(f'{bank_id}_data.csv', index=False)
    
    # Initialize banks with same models but diverse data
    print("\n2. Initializing bank simulators...")
    banks = {}
    model_types = ['xgboost', 'xgboost', 'xgboost']  # All banks use XGBoost for consistent methodology
    
    for i, bank_id in enumerate(['bank_A', 'bank_B', 'bank_C']):
        banks[bank_id] = BankSimulator(bank_id, f'{bank_id}_data.csv')
        model_type = model_types[i]
        print(f"  {bank_id} using {model_type} (diverse data perspective)")
    
    # Train models on diverse data perspectives
    print("\n3. Training bank models on diverse data...")
    for i, (bank_id, bank) in enumerate(banks.items()):
        model_type = model_types[i]
        accuracy = bank.train_local_model(model_type)
        print(f"{bank_id} ({model_type}) accuracy: {accuracy:.3f} (trained on unique data perspective)")
    
    # Initialize Consortium service
    print("\n4. Initializing consortium comparison service...")
    consortium = ConsortiumComparisonService()
    for bank_id, bank in banks.items():
        consortium.register_bank(bank_id, bank)
    
    # Test with multiple transactions using diverse data perspectives
    print("\n5. Testing consortium comparison scoring with diverse data perspectives...")
    test_transactions = [generate_synthetic_transaction() for _ in range(5)]
    
    for i, transaction in enumerate(test_transactions):
        print(f"\nTransaction {i+1}:")
        result = consortium.generate_comparison_score(transaction)
        
        print(f"  Individual Scores: {result['individual_scores']}")
        print(f"  Consensus Score: {result['consensus_score']:.3f}")
        print(f"  Variance Score: {result['variance_score']:.3f}")
        print(f"  Network Anomaly: {result['network_anomaly_score']:.3f}")
        print(f"  Final Comparison Score: {result['final_comparison_score']:.3f}")
        print(f"  Confidence: {result['confidence_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Flagging Banks: {result['flagging_banks']}")
    
    # Analytics summary
    print("\n6. Analytics Summary:")
    summary = consortium.get_analytics_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Model comparison summary
    print("\n7. Algorithm Performance Summary (for validation):")
    for model_type, results in model_comparison.items():
        if 'accuracy' in results:
            print(f"  {model_type}: {results['accuracy']:.4f} accuracy ({results['model_type']}) - validates algorithm choice")
        else:
            print(f"  {model_type}: {results.get('error', 'Failed')}")

def list_saved_models():
    """List all saved models and their information"""
    print("Saved Consortium Models")
    print("=" * 23)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found. No models have been trained yet.")
        return
    
    found_models = False
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        bank = BankSimulator(bank_id, f'{bank_id}_data.csv')
        model_info = bank.get_model_info()
        
        if model_info['exists']:
            found_models = True
            print(f"\n{bank_id.upper()}:")
            if 'error' in model_info:
                print(f"  Error: {model_info['error']}")
            else:
                print(f"  Model Type: {model_info.get('model_class', 'Unknown')}")
                print(f"  Algorithm: {model_info.get('model_type', 'Unknown')}")
                print(f"  Confidence: {model_info.get('model_confidence', 0):.3f}")
                print(f"  File Size: {model_info.get('model_file_size_kb', 0)} KB")
                print(f"  Trained: {model_info.get('timestamp', 'Unknown')}")
    
    if not found_models:
        print("No trained models found. Run training mode first.")

def main_list_models():
    """List saved models mode"""
    list_saved_models()

if __name__ == "__main__":
    import sys
    
    # Support different execution modes
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "train":
            main_training_only()
        elif mode == "inference":
            main_inference_only()
        elif mode == "list":
            main_list_models()
        elif mode == "full":
            main()
        else:
            print("Usage: python script.py [train|inference|list|full]")
            print("  train: Train and save models only")
            print("  inference: Use pre-trained models for inference only")
            print("  list: List all saved models and their information")
            print("  full: Complete demo with training and inference (default)")
    else:
        # Default: run full demo
        main()
