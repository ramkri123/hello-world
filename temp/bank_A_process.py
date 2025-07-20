#!/usr/bin/env python3
"""
Bank A - Wire Transfer Specialist
Runs as individual Python process
"""

import sys
import os
import logging
from participant_node import ParticipantNode, NodeConfig

# Configure logging for Bank A
logging.basicConfig(
    level=logging.INFO,
    format='[BANK_A] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for Bank A process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bank A - Wire Transfer Specialist')
    parser.add_argument('--consortium-url', default='http://localhost:8080', help='Consortium hub URL')
    
    args = parser.parse_args()
    
    # Bank A configuration
    config = NodeConfig(
        node_id="bank_A",
        specialty="wire_transfer_specialist",
        consortium_url=args.consortium_url,
        model_path="models/bank_A_model.pkl",
        geolocation="US-East"
    )
    
    # Create and start Bank A node
    logger.info("🏦 Starting Bank A - Wire Transfer Specialist")
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
            
            logger.info("🚀 Bank A is now active and waiting for inference requests")
            logger.info("   Specialization: Wire transfer fraud detection")
            logger.info("   Focus: Transaction amounts, business legitimacy, geographic patterns")
            logger.info("Press Ctrl+C to stop Bank A")
            
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
        logger.error(f"❌ Bank A startup error: {e}")
        sys.exit(1)
    
    finally:
        logger.info("🛑 Stopping Bank A")
        node.stop()

if __name__ == "__main__":
    main()
