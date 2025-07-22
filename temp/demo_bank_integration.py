#!/usr/bin/env python3
"""
Example: How a bank uses the consortium anonymization system
"""
import sys
sys.path.append('src')

from src.consortium.account_anonymizer import AccountAnonymizer

class BankFraudDetector:
    """Example bank fraud detection system that integrates with consortium"""
    
    def __init__(self, bank_id: str, bank_accounts: list):
        self.bank_id = bank_id
        self.bank_accounts = bank_accounts  # List of accounts this bank owns
        
    def analyze_consortium_inference(self, features: list, anonymized_accounts: dict) -> dict:
        """
        Analyze a consortium inference request
        
        Args:
            features: Anonymous behavioral features (35 features)
            anonymized_accounts: {'sender_anonymous': 'ANON_...', 'receiver_anonymous': 'ANON_...'}
            
        Returns:
            dict: Bank's risk assessment with scenario-aware confidence
        """
        # Step 1: Determine knowledge scenario using shared hash function
        scenario = AccountAnonymizer.bank_can_determine_ownership(
            self.bank_accounts, 
            anonymized_accounts
        )
        
        # Step 2: Get confidence weight for this scenario
        confidence_weight = AccountAnonymizer.get_scenario_confidence_weight(scenario)
        
        # Step 3: Run fraud detection model on anonymous features
        # (In practice, this would use the bank's trained ML model)
        base_risk_score = self._run_fraud_model(features)
        
        # Step 4: Apply scenario-aware confidence weighting
        final_risk_score = base_risk_score
        final_confidence = confidence_weight
        
        print(f"🏦 {self.bank_id} ANALYSIS:")
        print(f"   Knowledge scenario: {scenario}")
        print(f"   Confidence weight: {confidence_weight:.2f}")
        print(f"   Base risk score: {base_risk_score:.3f}")
        print(f"   Final risk score: {final_risk_score:.3f}")
        print(f"   Final confidence: {final_confidence:.2f}")
        
        return {
            'risk_score': final_risk_score,
            'confidence': final_confidence,
            'scenario': scenario,
            'bank_id': self.bank_id
        }
    
    def _run_fraud_model(self, features: list) -> float:
        """Mock fraud detection model - returns risk score 0.0-1.0"""
        # In practice, this would be the bank's trained ML model
        # For demo, simulate different bank specializations
        if self.bank_id == 'bank_a':
            # Wire transfer specialist - focuses on amount patterns
            return min(0.8 if features[0] > 0.5 else 0.2, 1.0)  # Amount ratio feature
        elif self.bank_id == 'bank_b':
            # Identity verification specialist - focuses on account patterns  
            return min(0.9 if features[25] < 0.3 else 0.3, 1.0)  # Account age feature
        else:
            # Network analysis specialist - general pattern detection
            return min(sum(features[:5]) / 5 + 0.2, 1.0)  # Transaction timing patterns

def demo_bank_consortium_integration():
    """Demonstrate how banks integrate with consortium using one-way hash"""
    print("🌐 BANK CONSORTIUM INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize banks with their account lists
    banks = {
        'bank_a': BankFraudDetector('bank_a', ['ACCA12345', 'ACCA67890', 'ACCA11111']),
        'bank_b': BankFraudDetector('bank_b', ['ACCB67890', 'ACCB12345', 'ACCB22222']),
        'bank_c': BankFraudDetector('bank_c', ['ACCC99999', 'ACCC12345', 'ACCC55555'])
    }
    
    # Simulate consortium inference request
    print("\n📨 CONSORTIUM SENDS INFERENCE REQUEST:")
    
    # Transaction: ACCA12345 → ACCB67890 (Bank A knows sender, Bank B knows receiver)
    anonymized_accounts = AccountAnonymizer.anonymize_transaction_accounts('ACCA12345', 'ACCB67890')
    print(f"   Original transaction: ACCA12345 → ACCB67890")
    print(f"   Anonymized: {anonymized_accounts['sender_anonymous']} → {anonymized_accounts['receiver_anonymous']}")
    
    # Mock features (35 anonymous behavioral features)
    features = [0.8, 0.9, 0.7, 0.8, 0.6] + [0.5] * 30  # High-risk transaction features
    print(f"   Features: {len(features)} anonymous behavioral features")
    
    print("\n🏦 BANK RESPONSES:")
    responses = {}
    for bank_id, bank in banks.items():
        response = bank.analyze_consortium_inference(features, anonymized_accounts)
        responses[bank_id] = response
        print()
    
    print("📊 CONSORTIUM AGGREGATION:")
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for bank_id, response in responses.items():
        score = response['risk_score']
        confidence = response['confidence']
        weighted_score = score * confidence
        total_weighted_score += weighted_score
        total_weight += confidence
        print(f"   {bank_id}: {score:.3f} * {confidence:.2f} = {weighted_score:.3f}")
    
    final_consensus = total_weighted_score / total_weight if total_weight > 0 else 0.5
    print(f"   Final consensus: {final_consensus:.3f}")
    
    recommendation = "BLOCK" if final_consensus > 0.7 else "REVIEW" if final_consensus > 0.3 else "APPROVE"
    print(f"   Recommendation: {recommendation}")
    
    print("\n✅ INTEGRATION BENEFITS:")
    print("   • Banks determine their own knowledge scenarios using one-way hash")
    print("   • Banks apply appropriate confidence weights to their risk scores")
    print("   • Consortium gets meaningful scenario-aware consensus")
    print("   • Privacy fully preserved with one-way hash function")
    print("   • No central authority needs to know bank account mappings")

if __name__ == "__main__":
    demo_bank_consortium_integration()
