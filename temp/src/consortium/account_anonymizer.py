#!/usr/bin/env python3
"""
Consortium Account Anonymizer - Shared one-way hash function
All participants (banks and hub) use this same algorithm for consistent anonymization
"""

import hashlib

class AccountAnonymizer:
    """Shared anonymization utility using one-way hash function"""
    
    # Consortium-wide agreed hash parameters
    HASH_ALGORITHM = 'sha256'
    HASH_LENGTH = 16  # First 16 characters of hash
    PREFIX = 'ANON_'
    
    @staticmethod
    def anonymize_account(account_number: str) -> str:
        """
        One-way hash function for account anonymization
        
        Args:
            account_number: Original account number (e.g., 'ACCA12345')
            
        Returns:
            Anonymized identifier (e.g., 'ANON_a57b927481e29a3f')
            
        Properties:
        - One-way: Cannot reverse to get original account
        - Deterministic: Same account always produces same hash
        - Collision-resistant: Different accounts produce different hashes
        - No patterns: Original bank prefixes not visible in output
        """
        if not account_number:
            return f"{AccountAnonymizer.PREFIX}{'0' * AccountAnonymizer.HASH_LENGTH}"
        
        # Create one-way hash of account number
        hash_input = account_number.encode('utf-8')
        hash_object = hashlib.new(AccountAnonymizer.HASH_ALGORITHM, hash_input)
        hash_hex = hash_object.hexdigest()
        
        # Return first N characters with prefix
        return f"{AccountAnonymizer.PREFIX}{hash_hex[:AccountAnonymizer.HASH_LENGTH]}"
    
    @staticmethod
    def anonymize_transaction_accounts(sender_account: str, receiver_account: str) -> dict:
        """
        Anonymize both sender and receiver accounts for a transaction
        
        Returns:
            dict: {
                'sender_anonymous': 'ANON_...',
                'receiver_anonymous': 'ANON_...'
            }
        """
        return {
            'sender_anonymous': AccountAnonymizer.anonymize_account(sender_account),
            'receiver_anonymous': AccountAnonymizer.anonymize_account(receiver_account)
        }
    
    @staticmethod
    def bank_can_determine_ownership(bank_accounts: list, anonymized_accounts: dict) -> str:
        """
        Allow banks to determine their knowledge scenario by comparing
        their own account hashes to the anonymized identifiers
        
        Args:
            bank_accounts: List of account numbers this bank owns
            anonymized_accounts: Dict with 'sender_anonymous' and 'receiver_anonymous'
            
        Returns:
            str: 'knows_both', 'knows_sender', 'knows_receiver', 'knows_neither'
        """
        # Bank anonymizes its own accounts using the same one-way function
        bank_anonymous = [AccountAnonymizer.anonymize_account(acc) for acc in bank_accounts]
        
        sender_anon = anonymized_accounts.get('sender_anonymous', '')
        receiver_anon = anonymized_accounts.get('receiver_anonymous', '')
        
        knows_sender = sender_anon in bank_anonymous
        knows_receiver = receiver_anon in bank_anonymous
        
        if knows_sender and knows_receiver:
            return 'knows_both'
        elif knows_sender:
            return 'knows_sender'
        elif knows_receiver:
            return 'knows_receiver'
        else:
            return 'knows_neither'
    
    @staticmethod
    def get_scenario_confidence_weight(scenario: str) -> float:
        """
        Get confidence weight for a knowledge scenario
        
        Args:
            scenario: 'knows_both', 'knows_sender', 'knows_receiver', 'knows_neither'
            
        Returns:
            float: Confidence weight (0.0 to 1.0)
        """
        weights = {
            'knows_both': 1.0,     # Highest confidence - bank knows both accounts
            'knows_sender': 0.8,   # High confidence - bank knows sender (their customer)
            'knows_receiver': 0.7, # Medium-high confidence - bank knows receiver
            'knows_neither': 0.4   # Lower confidence but still valuable for patterns
        }
        return weights.get(scenario, 0.5)  # Default if unknown scenario
