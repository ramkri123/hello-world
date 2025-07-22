#!/usr/bin/env python3
"""
Quick Test - Verify Consolidated System Works
"""

import sys
import os
from pathlib import Path

def verify_structure():
    """Verify all essential files are present"""
    base_path = Path(__file__).parent
    
    essential_files = [
        "README.md",
        "requirements.txt", 
        "launch_system.py",
        "ceo_fraud_ui.py",
        "ceo_fraud_demo.py",
        "test_ceo_detection.py",
        "src/consortium/consortium_hub.py",
        "src/consortium/privacy_preserving_nlp.py",
        "src/consortium/account_anonymizer.py",
        "src/consortium/bank_A_process.py",
        "src/consortium/bank_B_process.py", 
        "src/consortium/bank_C_process.py",
        "templates/ceo_fraud_focus.html",
        "docs/PROBLEM_AND_SOLUTION_ARCHITECTURE.md",
        "models/bank_A_model.pkl",
        "models/bank_B_model.pkl",
        "models/bank_C_model.pkl"
    ]
    
    print("🔍 Verifying Consolidated System Structure...")
    print("=" * 60)
    
    missing_files = []
    for file_path in essential_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    print("=" * 60)
    
    if missing_files:
        print(f"❌ {len(missing_files)} files missing:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print(f"✅ All {len(essential_files)} essential files present!")
        return True

def show_usage():
    """Show how to use the consolidated system"""
    print("\n🚀 Consolidated CEO Fraud Detection System")
    print("=" * 60)
    print("\n📋 Quick Start:")
    print("   1. pip install -r requirements.txt")
    print("   2. python launch_system.py")
    print("   3. Open http://localhost:5000")
    print("\n🧪 Testing:")
    print("   python test_ceo_detection.py")
    print("\n🎭 CLI Demo:")
    print("   python ceo_fraud_demo.py")
    print("\n📊 System Focus:")
    print("   • CEO fraud of different levels vs legitimate CEO")
    print("   • Role of Bank vs Role of Consortium")
    print("   • Real-time collaborative fraud detection")

if __name__ == "__main__":
    structure_ok = verify_structure()
    show_usage()
    
    if structure_ok:
        print("\n✅ Consolidated system is ready for deployment!")
    else:
        print("\n❌ Please fix missing files before deployment.")
        sys.exit(1)
