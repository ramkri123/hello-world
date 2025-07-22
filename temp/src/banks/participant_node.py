#!/usr/bin/env python3
"""
Participant Node - Individual bank/organization service
Runs as separate HTTP client process
"""

import requests
import time
import threading
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pickle
import os
import sys
import joblib
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from consortium.account_anonymizer import AccountAnonymizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    node_id: str
    specialty: str
    consortium_url: str
    model_path: str
    bank_accounts: List[str]  # List of account numbers this bank owns
    geolocation: str = "US-East"
    heartbeat_interval: int = 30

class ParticipantNode:
    def __init__(self, config: NodeConfig):
        self.config = config
        self.session_token = None
        self.is_registered = False
        self.is_running = False
        
        # Load local model and scaler
        self.model = self._load_local_model()
        self.scaler = self._load_scaler()
        
        # Store bank account list for scenario determination
        self.bank_accounts = config.bank_accounts
        logger.info(f"🏦 Bank {config.node_id} initialized with {len(self.bank_accounts)} accounts")
        logger.info(f"   📋 Account sample: {self.bank_accounts[:2]}..." if len(self.bank_accounts) > 2 else f"   📋 Accounts: {self.bank_accounts}")
        
    def _load_local_model(self):
        """Load the local AI/ML model"""
        try:
            if os.path.exists(self.config.model_path):
                model = joblib.load(self.config.model_path)
                logger.info(f"✅ Loaded model: {self.config.model_path}")
                return model
            else:
                logger.warning(f"⚠️ Model file not found: {self.config.model_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None
    
    def _load_scaler(self):
        """Load the feature scaler"""
        scaler_path = self.config.model_path.replace('_model.pkl', '_scaler.pkl')
        try:
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Loaded scaler: {scaler_path}")
                return scaler
            else:
                logger.warning(f"⚠️ Scaler file not found: {scaler_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading scaler: {e}")
            return None
    
    def register_with_consortium(self) -> bool:
        """Register this node with the consortium hub"""
        try:
            logger.info(f"🔄 REGISTERING {self.config.node_id} with consortium")
            logger.info(f"   🎯 Specialty: {self.config.specialty}")
            logger.info(f"   🌐 Consortium URL: {self.config.consortium_url}")
            logger.info(f"   📍 Location: {self.config.geolocation}")
            
            registration_data = {
                "node_id": self.config.node_id,
                "specialty": self.config.specialty,
                "endpoint": None,  # No inbound endpoint - outbound-only connection
                "geolocation": self.config.geolocation
            }
            
            logger.info(f"   📤 Sending registration request...")
            
            response = requests.post(
                f"{self.config.consortium_url}/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_token = result.get('session_token')
                self.is_registered = True
                logger.info(f"✅ REGISTRATION SUCCESSFUL: {self.config.node_id}")
                logger.info(f"   🔑 Session token: {self.session_token[:20]}...")
                logger.info(f"   🎯 Status: {result.get('status')}")
                return True
            else:
                logger.error(f"❌ REGISTRATION FAILED: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            return False
    
    def process_inference(self, features: List[float], anonymized_accounts: Dict[str, str] = None) -> Dict[str, Any]:
        """Process inference request using local model with scenario-aware confidence"""
        try:
            logger.info(f"🔄 PROCESSING INFERENCE at {self.config.node_id}")
            logger.info(f"   📊 Specialty: {self.config.specialty}")
            logger.info(f"   📈 Features received: {len(features)} values")
            logger.info(f"   🔍 First 10 features: {features[:10]}")
            
            # Determine knowledge scenario if account data provided
            scenario = "knows_neither"
            confidence_weight = AccountAnonymizer.get_scenario_confidence_weight(scenario)
            
            if anonymized_accounts:
                scenario = AccountAnonymizer.bank_can_determine_ownership(
                    self.bank_accounts, anonymized_accounts
                )
                confidence_weight = AccountAnonymizer.get_scenario_confidence_weight(scenario)
                logger.info(f"   🎯 Knowledge scenario: {scenario} (confidence weight: {confidence_weight:.2f})")
                logger.info(f"   🔐 Anonymized accounts: {anonymized_accounts}")
            
            start_time = time.time()
            
            if self.model is None:
                logger.error(f"   ❌ No model loaded - cannot process inference")
                raise ValueError(f"Model not loaded for {self.config.node_id}")
            
            logger.info(f"   🤖 Using trained model: {type(self.model).__name__}")
            # Use actual model only
            features_array = np.array([features])
            # Get probability of fraud (class 1)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_array)
                base_score = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])
                logger.info(f"   📊 Model probabilities: {probabilities[0]}")
            else:
                base_score = float(self.model.predict(features_array)[0])
            
            # Ensure base score is between 0 and 1
            base_score = max(0.0, min(1.0, base_score))
            
            # Apply scenario-aware confidence weighting
            final_score = base_score  # Keep original risk score
            scenario_confidence = confidence_weight  # Use scenario confidence as final confidence
            
            logger.info(f"🎯 Processed inference:")
            logger.info(f"   📊 Base risk score: {base_score:.3f}")
            logger.info(f"   🎭 Scenario: {scenario}")
            logger.info(f"   🎯 Final confidence: {scenario_confidence:.2f}")
            
            return {
                "risk_score": final_score,
                "base_risk_score": base_score,
                "confidence": scenario_confidence,
                "scenario": scenario,
                "processed_at": datetime.now().isoformat(),
                "model_version": "1.0"
            }
            
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
            return {
                "risk_score": 0.5,  # Default neutral score
                "confidence": 0.1,
                "scenario": "error",
                "error": str(e)
            }

    def submit_score(self, session_id: str, inference_result: Dict[str, Any]) -> bool:
        """Submit inference score to consortium hub"""
        try:
            score_data = {
                "session_id": session_id,
                "participant_id": self.config.node_id,
                "risk_score": inference_result["risk_score"],
                "confidence": inference_result["confidence"]
            }
            
            response = requests.post(
                f"{self.config.consortium_url}/score",
                json=score_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Submitted score for session {session_id}")
                return True
            else:
                logger.error(f"❌ Score submission failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Score submission error: {e}")
            return False
    
    def start_polling(self):
        """Start polling for inference requests from consortium"""
        self.is_running = True
        
        def poll_loop():
            while self.is_running:
                try:
                    # Poll for pending inference requests
                    response = requests.get(
                        f"{self.config.consortium_url}/poll_inference", 
                        params={'participant_id': self.config.node_id},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        # Received inference request
                        inference_data = response.json()
                        session_id = inference_data['session_id']
                        features = inference_data['features']
                        
                        logger.info(f"📨 {self.config.node_id} received inference request for session {session_id}")
                        logger.info(f"   📊 Processing {len(features)} features")
                        
                        # Process the inference with real model
                        result = self.process_inference(features)
                        
                        # Submit the score back to consortium
                        success = self.submit_score(session_id, result)
                        
                        if success:
                            logger.info(f"✅ {self.config.node_id} completed inference for session {session_id}")
                        else:
                            logger.error(f"❌ {self.config.node_id} failed to submit score for session {session_id}")
                            
                    elif response.status_code == 204:
                        # No pending inference - this is normal
                        logger.debug(f"🔄 {self.config.node_id} - no pending inference")
                        
                    else:
                        logger.debug(f"Poll response: {response.status_code}")
                    
                    time.sleep(5)  # Poll every 5 seconds for real work
                    
                except Exception as e:
                    logger.debug(f"Polling error: {e}")
                    time.sleep(15)  # Back off on error
        
        threading.Thread(target=poll_loop, daemon=True).start()
        logger.info(f"🔄 Started polling for {self.config.node_id}")
    
    def stop(self):
        """Stop the participant node"""
        self.is_running = False
        logger.info(f"🛑 Stopped {self.config.node_id}")

def create_bank_nodes(consortium_url: str) -> List[ParticipantNode]:
    """Create the three bank participant nodes with their account lists"""
    nodes = []
    
    # Bank A - Wire Transfer Specialist
    config_a = NodeConfig(
        node_id="bank_A",
        specialty="wire_transfer_specialist",
        consortium_url=consortium_url,
        model_path="models/bank_A_model.pkl",
        bank_accounts=['ACCA12345', 'ACCA67890', 'ACCA11111', 'ACCA22222', 'ACCA33333']
    )
    nodes.append(ParticipantNode(config_a))
    
    # Bank B - Identity Verification Expert
    config_b = NodeConfig(
        node_id="bank_B", 
        specialty="identity_verification_expert",
        consortium_url=consortium_url,
        model_path="models/bank_B_model.pkl",
        bank_accounts=['ACCB67890', 'ACCB12345', 'ACCB22222', 'ACCB44444', 'ACCB55555']
    )
    nodes.append(ParticipantNode(config_b))
    
    # Bank C - Network Pattern Analyst
    config_c = NodeConfig(
        node_id="bank_C",
        specialty="network_pattern_analyst", 
        consortium_url=consortium_url,
        model_path="models/bank_C_model.pkl",
        bank_accounts=['ACCC99999', 'ACCC12345', 'ACCC55555', 'ACCC66666', 'ACCC77777']
    )
    nodes.append(ParticipantNode(config_c))
    
    return nodes

def main():
    """Main function to start participant nodes"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Participant Node')
    parser.add_argument('--consortium-url', default='http://localhost:8080', help='Consortium hub URL')
    parser.add_argument('--node-id', help='Specific node to run (bank_A, bank_B, bank_C)')
    parser.add_argument('--all', action='store_true', help='Run all bank nodes')
    
    args = parser.parse_args()
    
    if args.all:
        # Run all bank nodes
        nodes = create_bank_nodes(args.consortium_url)
        
        # Register all nodes
        for node in nodes:
            if node.register_with_consortium():
                node.start_polling()
            else:
                logger.error(f"Failed to register {node.config.node_id}")
        
        logger.info("🚀 All bank nodes started. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Stopping all nodes...")
            for node in nodes:
                node.stop()
    
    elif args.node_id:
        # Run specific node
        nodes = create_bank_nodes(args.consortium_url)
        node = next((n for n in nodes if n.config.node_id == args.node_id), None)
        
        if node:
            if node.register_with_consortium():
                node.start_polling()
                logger.info(f"🚀 {args.node_id} started. Press Ctrl+C to stop.")
                
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info(f"🛑 Stopping {args.node_id}...")
                    node.stop()
            else:
                logger.error(f"Failed to register {args.node_id}")
        else:
            logger.error(f"Unknown node ID: {args.node_id}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
