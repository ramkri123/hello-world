#!/usr/bin/env python3
"""
Consortium Anonymization Standard
Shared utility for consistent account anonymization across all consortium members
"""
import hashlib

def anonymize_account(account: str) -> str:
    """
    Standard consortium account anonymization function
    
    This function is shared across all consortium members to ensure
    consistent anonymization. Banks can use this to:
    1. Anonymize their own accounts for comparison
    2. Determine which accounts in inference requests belong to them
    3. Calculate appropriate confidence weights for their scenarios
    
    Args:
        account: Original account identifier (e.g., "ACCA12345")
        
    Returns:
        Anonymized identifier (e.g., "ANON_a1b2c3d4e5f6g7h8")
    """
    # Standard SHA256 hash without salt - allows banks to compute same hash
    account_hash = hashlib.sha256(account.encode()).hexdigest()[:16]
    return f"ANON_{account_hash}"

class BankScenarioDetector:
    """Helper class for banks to determine their knowledge scenario"""
    
    def __init__(self, bank_accounts: list):
        """
        Initialize with bank's account list
        
        Args:
            bank_accounts: List of account identifiers this bank owns
        """
        self.bank_accounts = bank_accounts
        self.anonymized_accounts = {
            account: anonymize_account(account) 
            for account in bank_accounts
        }
    
    def determine_scenario(self, sender_anonymous: str, receiver_anonymous: str) -> dict:
        """
        Determine this bank's knowledge scenario for a transaction
        
        Args:
            sender_anonymous: Anonymized sender account from inference request
            receiver_anonymous: Anonymized receiver account from inference request
            
        Returns:
            Dict with scenario info and confidence weight
        """
        knows_sender = sender_anonymous in self.anonymized_accounts.values()
        knows_receiver = receiver_anonymous in self.anonymized_accounts.values()
        
        if knows_sender and knows_receiver:
            scenario = "knows_both"
            confidence = 1.0
        elif knows_sender:
            scenario = "knows_sender"
            confidence = 0.8
        elif knows_receiver:
            scenario = "knows_receiver" 
            confidence = 0.7
        else:
            scenario = "knows_neither"
            confidence = 0.4
            
        return {
            'scenario': scenario,
            'confidence_weight': confidence,
            'knows_sender': knows_sender,
            'knows_receiver': knows_receiver
        }
    
    def get_owned_accounts(self, sender_anonymous: str, receiver_anonymous: str) -> dict:
        """
        Get which specific accounts this bank owns in the transaction
        
        Returns:
            Dict mapping anonymous IDs to original account IDs (for owned accounts only)
        """
        owned = {}
        
        # Reverse lookup for owned accounts
        for original, anonymous in self.anonymized_accounts.items():
            if anonymous == sender_anonymous:
                owned['sender'] = original
            elif anonymous == receiver_anonymous:
                owned['receiver'] = original
                
        return owned

# Example usage for banks:
if __name__ == "__main__":
    # Bank A would initialize with their accounts
    bank_a_accounts = ['ACCA12345', 'ACCA67890', 'ACCA11111']
    detector_a = BankScenarioDetector(bank_a_accounts)
    
    # When receiving inference request with anonymized accounts
    sender_anon = "ANON_a665a45920422f9d"  # From inference request
    receiver_anon = "ANON_b1c2d3e4f5a6b7c8"  # From inference request
    
    scenario_info = detector_a.determine_scenario(sender_anon, receiver_anon)
    print(f"Bank A scenario: {scenario_info['scenario']}")
    print(f"Confidence weight: {scenario_info['confidence_weight']}")
    
    owned_accounts = detector_a.get_owned_accounts(sender_anon, receiver_anon)
    print(f"Owned accounts: {owned_accounts}")
