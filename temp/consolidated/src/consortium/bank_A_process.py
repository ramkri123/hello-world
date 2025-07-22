#!/usr/bin/env python3
"""
Bank A - Wire Transfer Specialist
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

class BankAProcessor:
    def __init__(self):
        self.bank_id = 'bank_A'
        self.specialty = 'Wire Transfer Analysis'
        self.consortium_url = 'http://localhost:8080'
        self.model = None
        self.running = True
        
        # Load the model
        try:
            # Use path relative to project root
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'bank_A_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Bank A model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}, using mock scoring")
            self.model = None
    
    def analyze_transaction(self, transaction_data):
        """Analyze transaction using Bank A's wire transfer expertise"""
        try:
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
            
            print(f"🔍 Bank A analyzing transaction amount: ${amount:,.2f}")
            
            # Bank A specializes in wire transfer patterns
            base_risk = min(0.4, amount / 1000000)  # Higher amounts = higher risk
            
            # Add some wire transfer specific risk factors
            if amount > 500000:
                base_risk += 0.1
                print(f"   📈 Large amount risk: +0.1")
            if tx_type.lower() in ['wire', 'wire_transfer', 'international']:
                base_risk += 0.05
                print(f"   📈 Wire transfer risk: +0.05")
                
            print(f"   💯 Bank A final risk score: {base_risk:.3f}")
                
            return {
                'bank_id': self.bank_id,
                'fraud_score': base_risk,
                'confidence': 0.85,
                'specialty': self.specialty,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            print(f"❌ Bank A analysis error: {e}")
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
                print(f"✅ Bank A registered with consortium hub")
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
                        
                        print(f"📊 Bank A analyzed transaction {request_data.get('request_id')}")
                
            except Exception as e:
                if "Connection refused" not in str(e):
                    print(f"⚠️ Bank A polling error: {e}")
            
            time.sleep(1)  # Poll every second
    
    def run(self):
        """Main run method - outbound only communication"""
        print("🏦 Starting Bank A - Wire Transfer Specialist (Outbound Only)")
        print("🔒 Security: No inbound ports exposed")
        
        # Register with consortium
        while not self.register_with_consortium():
            print("⏳ Waiting for consortium hub...")
            time.sleep(5)
        
        # Start listening for analysis requests
        print("👂 Bank A listening for analysis requests...")
        self.listen_for_analysis_requests()

def main():
    """Main function for Bank A process"""
    bank = BankAProcessor()
    try:
        bank.run()
    except KeyboardInterrupt:
        print("\n🛑 Bank A shutting down...")
        bank.running = False

if __name__ == "__main__":
    main()
