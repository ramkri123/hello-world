#!/usr/bin/env python3
"""
Debug the consortium analysis to see what's being returned
"""

import numpy as np
from consortium_comparison_score_prototype import ConsortiumComparisonService, BankSimulator

def debug_analysis():
    """Debug a single scenario to see the response structure"""
    print("🔍 DEBUGGING CONSORTIUM ANALYSIS")
    
    # Initialize consortium and load banks
    consortium = ConsortiumComparisonService()
    
    # Load banks
    print("Loading banks...")
    banks = {}
    for bank_id in ['bank_A', 'bank_B', 'bank_C']:
        bank = BankSimulator(bank_id, f'{bank_id}_data.csv')
        if bank.load_model():
            banks[bank_id] = bank
            consortium.register_bank(bank_id, bank)
            print(f"✅ Loaded {bank_id}")
        else:
            print(f"❌ Failed to load {bank_id}")
    
    print(f"Total banks registered: {len(consortium.participating_banks)}")
    
    # Use the BEC demo scenario
    bec_demo = [0.35, 0.45, 0.75, 0.40, 0.85, 0.35, 0.40, 0.70, 0.80, 0.90,
                0.25, 0.35, 0.15, 0.30, 0.10, 0.70, 0.85, 0.90, 0.40, 0.35,
                0.75, 0.35, 0.65, 0.55, 0.85, 0.75, 0.70, 0.75, 0.45, 0.40]
    
    print(f"\nInput features length: {len(bec_demo)}")
    
    # Get result
    result = consortium.generate_comparison_score(bec_demo)
    
    print(f"\nResult keys: {list(result.keys())}")
    
    # Show key results
    print(f"\n📊 RESULTS:")
    print(f"   Final Score: {result['final_comparison_score']:.3f}")
    print(f"   Consensus Score: {result['consensus_score']:.3f}")
    print(f"   Variance: {result['variance_score']:.3f}")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Participating Banks: {result['participating_banks']}")
    
    if result['individual_scores']:
        print(f"\n🏦 INDIVIDUAL BANK SCORES:")
        for bank, score in result['individual_scores'].items():
            print(f"   {bank}: {score:.3f}")
    else:
        print(f"\n❌ No individual scores returned")

if __name__ == "__main__":
    debug_analysis()
