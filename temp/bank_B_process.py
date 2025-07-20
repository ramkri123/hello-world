#!/usr/bin/env python3
"""
Bank B - Identity Verification Expert
Runs as individual Python process
"""

import sys
import os
import logging
from participant_node import ParticipantNode, NodeConfig

# Configure logging for Bank B
logging.basicConfig(
    level=logging.INFO,
    format='[BANK_B] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for Bank B process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bank B - Identity Verification Expert')
    parser.add_argument('--consortium-url', default='http://localhost:8080', help='Consortium hub URL')
    
    args = parser.parse_args()
    
    # Bank B configuration
    config = NodeConfig(
        node_id="bank_B",
        specialty="identity_verification_expert",
        consortium_url=args.consortium_url,
        model_path="models/bank_B_model.pkl",
        geolocation="US-Central"
    )
    
    # Create and start Bank B node
    logger.info("🔍 Starting Bank B - Identity Verification Expert")
    logger.info(f"   Consortium: {args.consortium_url}")
    logger.info(f"   Model: {config.model_path}")
    logger.info("   Connection: Outbound-only HTTP client")
    
    node = ParticipantNode(config)
    
    try:
        # Register with consortium
        if node.register_with_consortium():
            logger.info("✅ Successfully registered with consortium")
            
            # Start polling for inference requests
            node.start_polling()
            
            logger.info("🚀 Bank B is now active and waiting for inference requests")
            logger.info("   Specialization: Identity verification and account validation")
            logger.info("   Focus: New account detection, identity document verification, KYC")
            logger.info("Press Ctrl+C to stop Bank B")
            
            # Keep running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal")
        else:
            logger.error("❌ Failed to register with consortium")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Bank B startup error: {e}")
        sys.exit(1)
    
    finally:
        logger.info("🛑 Stopping Bank B")
        node.stop()

if __name__ == "__main__":
    main()
