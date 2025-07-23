#!/usr/bin/env python3
"""
Bank B - Corporate Banking Specialist
Outbound-only communication to consortium hub (no inbound ports)
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import requests
import time
import threading
import json

# ramki
# receiver
anonymized_accounts = ['ANON_beea8ae9fe948ea5', 'ANON_a5058418f8ce514f']
anonymized_accounts_age = ["new", "old"]

class BankBProcessor:
    def __init__(self):
        self.bank_id = 'bank_B'
        self.specialty = 'Corporate Banking Analysis'
        self.consortium_url = 'http://localhost:8080'
        self.model = None
        self.running = True
        
        '''
        # Load the model
        try:
            # Use path relative to project root
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'bank_B_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Bank B model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}, using mock scoring")
            self.model = None
        '''
    
    def analyze_transaction(self, transaction_data):
        """Analyze transaction using Bank B's corporate banking expertise"""
        try:
            # ramki
            print("\n*** transaction_data start ***\n")
            print(transaction_data)
            sender_account_match = False
            receiver_account_match = False
            sender_anonymous = transaction_data['transaction_data']['sender_anonymous']
            receiver_anonymous = transaction_data['transaction_data']['receiver_anonymous']
            sender_account_age = "old"
            receiver_account_age = "new"
            i = 0
            for a in anonymized_accounts:
                if a == sender_anonymous:
                    sender_account_match = True
                    sender_account_age = anonymized_accounts_age[i]
                if a == receiver_anonymous:
                    receiver_account_match = True
                    receiver_account_age = anonymized_accounts_age[i]
                i = i + 1

            # Extract transaction amount from the raw data structure
            # transaction_data contains the entire original request
            if 'transaction_data' in transaction_data:
                # New format: extract from nested transaction_data
                inner_tx = transaction_data['transaction_data']
                amount = float(inner_tx.get('amount', 0))
                tx_type = inner_tx.get('transaction_type', '')
            else:
                # Fallback: try direct access
                amount = float(transaction_data.get('transaction_amount', 0))
                tx_type = transaction_data.get('transaction_type', '')
                
            print(f"🔍 Bank B analyzing transaction amount: ${amount:,.2f}")
            
            # Bank B specializes in corporate banking patterns
            base_risk = min(0.35, amount / 2000000)  # Different threshold for corporate
            
            # Add corporate-specific risk factors
            if amount > 1000000:
                base_risk += 0.05  # Large corporate transactions
                print(f"   📈 Large corporate amount risk: +0.15")
            if transaction_data.get('business_type', '').lower() in ['corporation', 'llc', 'inc']:
                base_risk += 0.03
                print(f"   📈 Corporate entity risk: +0.03")
            if tx_type.lower() in ['ach', 'corporate']:
                base_risk += 0.02
                print(f"   📈 Corporate transaction type risk: +0.02")

            if (sender_account_age == 'new'):
                print(f"   New sender account risk: +0.3")
                base_risk += 0.3
            if (receiver_account_age == 'new'):
                print(f"   New receiver account risk: +0.3")
                base_risk += 0.3

            print("\n*** transaction_data end ***\n")
                
            print(f"   💯 Bank B final risk score: {base_risk:.3f}")
                
            return {
                'bank_id': self.bank_id,
                'fraud_score': base_risk,
                'confidence': 0.88,
                'specialty': self.specialty,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            print(f"❌ Bank B analysis error: {e}")
            return {
                'bank_id': self.bank_id,
                'fraud_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def register_with_consortium(self):
        """Register this bank with the consortium hub"""
        try:
            response = requests.post(f"{self.consortium_url}/register_bank", 
                                   json={
                                       'bank_id': self.bank_id,
                                       'specialty': self.specialty,
                                       'status': 'online'
                                   }, timeout=5)
            if response.status_code == 200:
                print(f"✅ Bank B registered with consortium hub")
                return True
        except Exception as e:
            print(f"❌ Failed to register with consortium: {e}")
        return False
    
    def listen_for_analysis_requests(self):
        """Listen for analysis requests from consortium (polling)"""
        while self.running:
            try:
                # Poll consortium for analysis requests
                response = requests.get(f"{self.consortium_url}/get_analysis_request/{self.bank_id}", 
                                      timeout=2)
                
                if response.status_code == 200:
                    request_data = response.json()
                    if request_data.get('has_request'):
                        # Process the transaction
                        transaction = request_data.get('transaction_data')
                        analysis = self.analyze_transaction(transaction)
                        
                        # Send analysis back to consortium
                        requests.post(f"{self.consortium_url}/submit_analysis",
                                    json={
                                        'request_id': request_data.get('request_id'),
                                        'analysis': analysis
                                    }, timeout=5)
                        
                        print(f"📊 Bank B analyzed transaction {request_data.get('request_id')}")
                
            except Exception as e:
                if "Connection refused" not in str(e):
                    print(f"⚠️ Bank B polling error: {e}")
            
            time.sleep(1)  # Poll every second
    
    def run(self):
        """Main run method - outbound only communication"""
        print("🏦 Starting Bank B - Corporate Banking Specialist (Outbound Only)")
        print("🔒 Security: No inbound ports exposed")
        
        # Register with consortium
        while not self.register_with_consortium():
            print("⏳ Waiting for consortium hub...")
            time.sleep(5)
        
        # Start listening for analysis requests
        print("👂 Bank B listening for analysis requests...")
        self.listen_for_analysis_requests()

def main():
    """Main function for Bank B process"""
    bank = BankBProcessor()
    try:
        bank.run()
    except KeyboardInterrupt:
        print("\n🛑 Bank B shutting down...")
        bank.running = False

if __name__ == "__main__":
    main()
