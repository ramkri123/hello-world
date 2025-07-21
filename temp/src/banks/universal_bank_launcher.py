#!/usr/bin/env python3
"""
Universal Bank Process Launcher
Unified launcher that can start any bank with different configurations
Maximum code reuse with configuration-driven specialization
"""

import sys
import os
import logging
import argparse
import time
from participant_node import ParticipantNode, NodeConfig

# Bank configurations - all banks share the same code, different configs
BANK_CONFIGS = {
    "bank_A": {
        "display_name": "Bank A - Wire Transfer Specialist",
        "specialty": "wire_transfer_specialist", 
        "model_path": "models/bank_A_model.pkl",
        "log_prefix": "BANK_A",
        "emoji": "🏦",
        "specialization_desc": "Wire transfer fraud detection",
        "focus_areas": "Transaction amounts, business legitimacy, geographic patterns",
        "feature_range": "0-14 (sender/transaction patterns)"
    },
    "bank_B": {
        "display_name": "Bank B - Identity Verification Expert",
        "specialty": "identity_receiver_specialist",
        "model_path": "models/bank_B_model.pkl", 
        "log_prefix": "BANK_B",
        "emoji": "🔍",
        "specialization_desc": "Identity verification and KYC expertise",
        "focus_areas": "Account verification, identity patterns, receiver analysis",
        "feature_range": "15-29 (identity/receiver patterns)"
    },
    "bank_C": {
        "display_name": "Bank C - Network Pattern Analyst", 
        "specialty": "network_account_specialist",
        "model_path": "models/bank_C_model.pkl",
        "log_prefix": "BANK_C", 
        "emoji": "🌐",
        "specialization_desc": "Network pattern analysis and behavioral detection",
        "focus_areas": "Behavioral signatures, network analysis, timing patterns",
        "feature_range": "30+ (network/behavioral patterns)"
    }
}

class UniversalBankLauncher:
    """Universal launcher for any bank configuration"""
    
    def __init__(self, bank_id: str, consortium_url: str = "http://localhost:8080"):
        if bank_id not in BANK_CONFIGS:
            raise ValueError(f"Unknown bank_id: {bank_id}. Available: {list(BANK_CONFIGS.keys())}")
            
        self.bank_id = bank_id
        self.config_data = BANK_CONFIGS[bank_id]
        self.consortium_url = consortium_url
        
        # Setup bank-specific logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[{self.config_data["log_prefix"]}] %(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def create_node_config(self) -> NodeConfig:
        """Create NodeConfig for this bank"""
        return NodeConfig(
            node_id=self.bank_id,
            specialty=self.config_data["specialty"],
            consortium_url=self.consortium_url,
            model_path=self.config_data["model_path"],
            geolocation="US-East"
        )
    
    def start_bank(self):
        """Start this bank process"""
        
        self.logger.info(f"{self.config_data['emoji']} Starting {self.config_data['display_name']}")
        self.logger.info(f"   Consortium: {self.consortium_url}")
        self.logger.info(f"   Model: {self.config_data['model_path']}")
        self.logger.info(f"   Specialization: {self.config_data['specialization_desc']}")
        self.logger.info(f"   Feature Range: {self.config_data['feature_range']}")
        self.logger.info("   Connection: Outbound-only HTTP client")
        
        # Create participant node
        node_config = self.create_node_config()
        node = ParticipantNode(node_config)
        
        try:
            # Register with consortium
            if node.register_with_consortium():
                self.logger.info("✅ Successfully registered with consortium")
                
                # Start polling for inference requests
                node.start_polling()
                
                self.logger.info(f"🚀 {self.bank_id.upper()} is now active and waiting for inference requests")
                self.logger.info(f"   Focus: {self.config_data['focus_areas']}")
                self.logger.info("Press Ctrl+C to stop this bank")
                
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.logger.info("🛑 Received shutdown signal")
            else:
                self.logger.error("❌ Failed to register with consortium")
                sys.exit(1)
        
        except Exception as e:
            self.logger.error(f"❌ {self.bank_id.upper()} startup error: {e}")
            sys.exit(1)
        
        finally:
            self.logger.info(f"🛑 Stopping {self.bank_id.upper()}")
            node.stop()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Universal Bank Process Launcher')
    parser.add_argument('bank_id', nargs='?', choices=list(BANK_CONFIGS.keys()), 
                       help='Bank to start')
    parser.add_argument('--consortium-url', default='http://localhost:8080', 
                       help='Consortium hub URL')
    parser.add_argument('--list-banks', action='store_true',
                       help='List available bank configurations')
    
    args = parser.parse_args()
    
    # List banks if requested
    if args.list_banks:
        print("🏦 Available Bank Configurations:")
        print("=" * 50)
        for bank_id, config in BANK_CONFIGS.items():
            print(f"{config['emoji']} {bank_id}:")
            print(f"   Name: {config['display_name']}")
            print(f"   Specialty: {config['specialization_desc']}")
            print(f"   Features: {config['feature_range']}")
            print(f"   Model: {config['model_path']}")
            print()
        return
    
    # Check if bank_id is provided
    if not args.bank_id:
        parser.error("bank_id is required unless --list-banks is used")
    
    # Start the specified bank
    try:
        launcher = UniversalBankLauncher(args.bank_id, args.consortium_url)
        launcher.start_bank()
    except Exception as e:
        print(f"❌ Failed to start {args.bank_id}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
