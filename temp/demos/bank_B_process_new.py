#!/usr/bin/env python3
"""
Bank B - Identity Verification Expert
Runs as individual Python process
UPDATED: Now uses the universal bank launcher for maximum code reuse
"""

import sys
import os

# Import the universal launcher
from universal_bank_launcher import UniversalBankLauncher

def main():
    """Main function for Bank B process - delegates to universal launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bank B - Identity Verification Expert')
    parser.add_argument('--consortium-url', default='http://localhost:8080', help='Consortium hub URL')
    
    args = parser.parse_args()
    
    # Use the universal launcher with bank_B configuration
    try:
        launcher = UniversalBankLauncher("bank_B", args.consortium_url)
        launcher.start_bank()
    except Exception as e:
        print(f"❌ Failed to start Bank B: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
