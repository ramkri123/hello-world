#!/usr/bin/env python3
"""
Generic Bank Process - Configurable bank participant
Can be specialized through configuration and custom modules
"""

import sys
import os
import logging
import importlib.util
from participant_node import ParticipantNode, NodeConfig

def load_bank_specialization(bank_id):
    """Load bank-specific specialization module if it exists"""
    try:
        spec_file = f"specializations/{bank_id}_specialization.py"
        if os.path.exists(spec_file):
            spec = importlib.util.spec_from_file_location(f"{bank_id}_spec", spec_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        logging.warning(f"Could not load specialization for {bank_id}: {e}")
    return None

def get_bank_config(bank_id):
    """Get bank configuration based on bank ID"""
    bank_configs = {
        "bank_A": {
            "specialty": "wire_transfer_specialist",
            "model_path": "models/bank_A_model.pkl",
            "geolocation": "US-East",
            "description": "Wire Transfer Specialist",
            "icon": "🏦",
            "focus": "Transaction amounts, business legitimacy, geographic patterns"
        },
        "bank_B": {
            "specialty": "identity_verification_expert", 
            "model_path": "models/bank_B_model.pkl",
            "geolocation": "US-Central",
            "description": "Identity Verification Expert",
            "icon": "🔍", 
            "focus": "New account detection, identity document verification, KYC"
        },
        "bank_C": {
            "specialty": "network_pattern_analyst",
            "model_path": "models/bank_C_model.pkl", 
            "geolocation": "US-West",
            "description": "Network Pattern Analyst",
            "icon": "🌐",
            "focus": "Cross-institutional patterns, behavioral anomalies, network analysis"
        }
    }
    return bank_configs.get(bank_id)

def main():
    """Main function for generic bank process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generic Bank Process')
    parser.add_argument('--bank-id', required=True, choices=['bank_A', 'bank_B', 'bank_C'], 
                       help='Bank identifier')
    parser.add_argument('--consortium-url', default='http://localhost:8080', 
                       help='Consortium hub URL')
    
    args = parser.parse_args()
    
    # Get bank configuration
    bank_config = get_bank_config(args.bank_id)
    if not bank_config:
        print(f"❌ Unknown bank ID: {args.bank_id}")
        sys.exit(1)
    
    # Configure logging for this bank
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{args.bank_id.upper()}] %(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create bank configuration
    config = NodeConfig(
        node_id=args.bank_id,
        specialty=bank_config["specialty"],
        consortium_url=args.consortium_url,
        model_path=bank_config["model_path"],
        geolocation=bank_config["geolocation"]
    )
    
    # Load bank-specific specializations
    specialization = load_bank_specialization(args.bank_id)
    
    # Create and start bank node
    logger.info(f"{bank_config['icon']} Starting {bank_config['description']}")
    logger.info(f"   Consortium: {args.consortium_url}")
    logger.info(f"   Model: {config.model_path}")
    logger.info("   Connection: Outbound-only HTTP client")
    if specialization:
        logger.info(f"   Specialization: Custom module loaded")
    
    node = ParticipantNode(config)
    
    # Apply specializations if available
    if specialization and hasattr(specialization, 'customize_node'):
        specialization.customize_node(node)
    
    try:
        # Register with consortium
        if node.register_with_consortium():
            logger.info("✅ Successfully registered with consortium")
            
            # Start polling for inference requests
            node.start_polling()
            
            logger.info(f"🚀 {bank_config['description']} is now active and waiting for inference requests")
            logger.info(f"   Specialization: {bank_config['specialty']}")
            logger.info(f"   Focus: {bank_config['focus']}")
            logger.info("Press Ctrl+C to stop")
            
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
        logger.error(f"❌ Bank startup error: {e}")
        sys.exit(1)
    
    finally:
        logger.info(f"🛑 Stopping {bank_config['description']}")
        node.stop()

if __name__ == "__main__":
    main()
