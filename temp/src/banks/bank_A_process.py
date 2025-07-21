#!/usr/bin/env python3
"""
Bank A - Wire Transfer Specialist
Runs as individual Python process
UPDATED: Now uses the universal bank launcher for maximum code reuse
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the universal launcher
from banks.universal_bank_launcher import UniversalBankLauncher

def main():
    """Main function for Bank A process - delegates to universal launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bank A - Wire Transfer Specialist')
    parser.add_argument('--consortium-url', default='http://localhost:8080', help='Consortium hub URL')
    
    args = parser.parse_args()
    
    # Use the universal launcher with bank_A configuration
    try:
        launcher = UniversalBankLauncher("bank_A", args.consortium_url)
        launcher.start_bank()
    except Exception as e:
        print(f"❌ Failed to start Bank A: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
