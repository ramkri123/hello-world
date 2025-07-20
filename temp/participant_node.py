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
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    node_id: str
    specialty: str
    consortium_url: str
    model_path: str
    geolocation: str = "US-East"
    heartbeat_interval: int = 30

class ParticipantNode:
    def __init__(self, config: NodeConfig):
        self.config = config
        self.session_token = None
        self.is_registered = False
        self.is_running = False
        
        # Load local model
        self.model = self._load_local_model()
        
    def _load_local_model(self):
        """Load the local AI/ML model"""
        try:
            if os.path.exists(self.config.model_path):
                with open(self.config.model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"✅ Loaded model: {self.config.model_path}")
                return model
            else:
                logger.warning(f"⚠️ Model file not found: {self.config.model_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None
    
    def register_with_consortium(self) -> bool:
        """Register this node with the consortium hub"""
        try:
            registration_data = {
                "node_id": self.config.node_id,
                "specialty": self.config.specialty,
                "endpoint": f"http://localhost:808{self.config.node_id[-1]}",  # Mock endpoint
                "geolocation": self.config.geolocation
            }
            
            response = requests.post(
                f"{self.config.consortium_url}/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_token = result.get('session_token')
                self.is_registered = True
                logger.info(f"✅ Registered with consortium: {self.config.node_id}")
                return True
            else:
                logger.error(f"❌ Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            return False
    
    def process_inference(self, features: List[float]) -> Dict[str, Any]:
        """Process inference request using local model"""
        try:
            if self.model is None:
                # Fallback to simulated scoring based on specialty
                score = self._simulate_scoring(features)
            else:
                # Use actual model
                features_array = np.array([features])
                # Get probability of fraud (class 1)
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features_array)
                    score = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])
                else:
                    score = float(self.model.predict(features_array)[0])
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            confidence = 0.95 if self.model else 0.75  # Lower confidence for simulated
            
            logger.info(f"🎯 Processed inference: score={score:.3f}, confidence={confidence:.2f}")
            
            return {
                "risk_score": score,
                "confidence": confidence,
                "processed_at": datetime.now().isoformat(),
                "model_version": "1.0"
            }
            
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
            return {
                "risk_score": 0.5,  # Default neutral score
                "confidence": 0.1,
                "error": str(e)
            }
    
    def _simulate_scoring(self, features: List[float]) -> float:
        """Simulate model scoring based on specialty"""
        try:
            # Different specialties focus on different feature ranges
            if "wire_transfer" in self.config.specialty.lower():
                # Focus on amount-related features (first few features)
                amount_features = features[:5]
                score = sum(amount_features) / len(amount_features)
            elif "identity" in self.config.specialty.lower():
                # Focus on identity/account features (middle features)
                identity_features = features[8:15]
                score = sum(identity_features) / len(identity_features)
            elif "network" in self.config.specialty.lower():
                # Focus on network/behavioral features (last features)
                network_features = features[20:]
                score = sum(network_features) / len(network_features)
            else:
                # General scoring
                score = sum(features) / len(features)
            
            # Add some randomness to make it realistic
            noise = np.random.normal(0, 0.1)
            score = max(0.0, min(1.0, score + noise))
            
            return score
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return 0.5
    
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
        """Start polling for inference requests (simulation)"""
        self.is_running = True
        
        def poll_loop():
            while self.is_running:
                try:
                    # In a real implementation, this would be a webhook or long-polling
                    # For now, we'll just check consortium health
                    response = requests.get(f"{self.config.consortium_url}/health", timeout=5)
                    if response.status_code == 200:
                        health = response.json()
                        logger.debug(f"Consortium health: {health['participants']} participants")
                    
                    time.sleep(10)  # Poll every 10 seconds
                    
                except Exception as e:
                    logger.debug(f"Polling error: {e}")
                    time.sleep(30)  # Back off on error
        
        threading.Thread(target=poll_loop, daemon=True).start()
        logger.info(f"🔄 Started polling for {self.config.node_id}")
    
    def simulate_inference_response(self, session_id: str, features: List[float]):
        """Simulate receiving and responding to an inference request"""
        try:
            # Process the inference
            result = self.process_inference(features)
            
            # Submit the score
            success = self.submit_score(session_id, result)
            
            if success:
                logger.info(f"✅ {self.config.node_id} completed inference for session {session_id}")
            else:
                logger.error(f"❌ {self.config.node_id} failed to submit score for session {session_id}")
                
        except Exception as e:
            logger.error(f"❌ Simulation error: {e}")
    
    def stop(self):
        """Stop the participant node"""
        self.is_running = False
        logger.info(f"🛑 Stopped {self.config.node_id}")

def create_bank_nodes(consortium_url: str) -> List[ParticipantNode]:
    """Create the three bank participant nodes"""
    nodes = []
    
    # Bank A - Wire Transfer Specialist
    config_a = NodeConfig(
        node_id="bank_A",
        specialty="wire_transfer_specialist",
        consortium_url=consortium_url,
        model_path="models/bank_A_model.pkl"
    )
    nodes.append(ParticipantNode(config_a))
    
    # Bank B - Identity Verification Expert
    config_b = NodeConfig(
        node_id="bank_B", 
        specialty="identity_verification_expert",
        consortium_url=consortium_url,
        model_path="models/bank_B_model.pkl"
    )
    nodes.append(ParticipantNode(config_b))
    
    # Bank C - Network Pattern Analyst
    config_c = NodeConfig(
        node_id="bank_C",
        specialty="network_pattern_analyst", 
        consortium_url=consortium_url,
        model_path="models/bank_C_model.pkl"
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
