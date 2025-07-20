#!/usr/bin/env python3
"""
Distributed Consortium Test Script
Tests the full HTTP-based distributed architecture
"""

import requests
import time
import threading
import logging
from participant_node import create_bank_nodes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONSORTIUM_HUB_URL = "http://localhost:8080"

def wait_for_hub():
    """Wait for consortium hub to be available"""
    logger.info("⏳ Waiting for consortium hub to start...")
    
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{CONSORTIUM_HUB_URL}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Consortium hub is ready!")
                return True
        except:
            pass
        
        time.sleep(1)
    
    logger.error("❌ Consortium hub failed to start")
    return False

def register_test_banks():
    """Register test bank nodes with the consortium"""
    logger.info("🏦 Registering test banks...")
    
    nodes = create_bank_nodes(CONSORTIUM_HUB_URL)
    registered_nodes = []
    
    for node in nodes:
        if node.register_with_consortium():
            registered_nodes.append(node)
            logger.info(f"✅ Registered {node.config.node_id}")
        else:
            logger.error(f"❌ Failed to register {node.config.node_id}")
    
    return registered_nodes

def test_inference_flow(nodes):
    """Test the complete inference flow"""
    logger.info("🧪 Testing inference flow...")
    
    # BEC demo transaction features
    test_features = [0.35, 0.45, 0.75, 0.40, 0.85, 0.35, 0.40, 0.70, 0.80, 0.90,
                     0.25, 0.35, 0.15, 0.30, 0.10, 0.70, 0.85, 0.90, 0.40, 0.35,
                     0.75, 0.35, 0.65, 0.55, 0.85, 0.75, 0.70, 0.75, 0.45, 0.40]
    
    # 1. Submit inference request
    try:
        response = requests.post(
            f"{CONSORTIUM_HUB_URL}/inference",
            json={"features": test_features, "use_case": "fraud_detection"},
            timeout=10
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Failed to submit inference: {response.status_code}")
            return False
        
        result = response.json()
        session_id = result['session_id']
        logger.info(f"✅ Submitted inference request: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Inference submission error: {e}")
        return False
    
    # 2. Simulate participant responses
    logger.info("🤖 Simulating participant responses...")
    
    def simulate_responses():
        time.sleep(2)  # Wait a bit for session to be set up
        
        for node in nodes:
            try:
                # Simulate receiving and processing the inference request
                node.simulate_inference_response(session_id, test_features)
                time.sleep(1)  # Stagger responses
            except Exception as e:
                logger.error(f"❌ Response simulation error for {node.config.node_id}: {e}")
    
    # Start response simulation in background
    threading.Thread(target=simulate_responses, daemon=True).start()
    
    # 3. Poll for results
    logger.info("⏳ Waiting for consortium results...")
    
    for i in range(45):  # Wait up to 45 seconds
        try:
            response = requests.get(f"{CONSORTIUM_HUB_URL}/results/{session_id}", timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'final_score' in result:
                    # Results are ready
                    logger.info("✅ Received consortium results!")
                    
                    # Display results
                    final_score = result['final_score']
                    variance = result['variance']
                    recommendation = result['recommendation']
                    individual_scores = result.get('individual_scores', {})
                    
                    logger.info(f"📊 Final Score: {final_score:.3f}")
                    logger.info(f"📊 Variance: {variance:.3f}")
                    logger.info(f"📊 Recommendation: {recommendation}")
                    
                    for bank_id, score in individual_scores.items():
                        logger.info(f"📊 {bank_id}: {float(score):.3f}")
                    
                    return True
                else:
                    # Still waiting
                    responses_received = result.get('responses_received', 0)
                    total_participants = result.get('total_participants', len(nodes))
                    logger.info(f"⏳ Progress: {responses_received}/{total_participants} responses")
            
        except Exception as e:
            logger.debug(f"Polling error: {e}")
        
        time.sleep(1)
    
    logger.error("❌ Test timed out waiting for results")
    return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Distributed Consortium Test")
    
    # Wait for hub to be available
    if not wait_for_hub():
        logger.error("❌ Test failed: Hub not available")
        return False
    
    # Register test banks
    nodes = register_test_banks()
    if len(nodes) < 2:
        logger.error("❌ Test failed: Not enough participants registered")
        return False
    
    # Test inference flow
    success = test_inference_flow(nodes)
    
    if success:
        logger.info("✅ Distributed consortium test PASSED!")
        logger.info("🎯 The HTTP-based distributed architecture is working correctly!")
        logger.info("🌐 You can now start the UI with: streamlit run distributed_consortium_ui.py")
    else:
        logger.error("❌ Distributed consortium test FAILED!")
    
    return success

if __name__ == "__main__":
    main()
