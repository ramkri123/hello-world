#!/usr/bin/env python3
"""
Bank C - Network Pattern Analyst
Runs as individual Python process
"""

import sys
import os
import logging
from participant_node import ParticipantNode, NodeConfig

# Configure logging for Bank C
logging.basicConfig(
    level=logging.INFO,
    format='[BANK_C] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for Bank C process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bank C - Network Pattern Analyst')
    parser.add_argument('--consortium-url', default='http://localhost:8080', help='Consortium hub URL')
    
    args = parser.parse_args()
    
    # Bank C configuration
    config = NodeConfig(
        node_id="bank_C",
        specialty="network_pattern_analyst",
        consortium_url=args.consortium_url,
        model_path="models/bank_C_model.pkl",
        geolocation="US-West"
    )
    
    # Create and start Bank C node
    logger.info("🌐 Starting Bank C - Network Pattern Analyst")
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
            
            logger.info("🚀 Bank C is now active and waiting for inference requests")
            logger.info("   Specialization: Network pattern analysis and behavioral detection")
            logger.info("   Focus: Cross-institutional patterns, behavioral anomalies, network analysis")
            logger.info("Press Ctrl+C to stop Bank C")
            
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
        logger.error(f"❌ Bank C startup error: {e}")
        sys.exit(1)
    
    finally:
        logger.info("🛑 Stopping Bank C")
        node.stop()

if __name__ == "__main__":
    main()
