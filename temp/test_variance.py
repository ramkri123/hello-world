"""
Test with different fraud feature combinations to find realistic variance
"""

from consortium_comparison_score_prototype import BankSimulator, ConsortiumComparisonService

# Load the retrained models using the correct initialization
consortium = ConsortiumComparisonService()

# The models use default paths from constructor
bank_A = BankSimulator('bank_A', 'bank_A_data.csv')
bank_A.load_model()

bank_B = BankSimulator('bank_B', 'bank_B_data.csv')  
bank_B.load_model()

bank_C = BankSimulator('bank_C', 'bank_C_data.csv')
bank_C.load_model()

consortium.register_bank('bank_A', bank_A)
consortium.register_bank('bank_B', bank_B)
consortium.register_bank('bank_C', bank_C)

print("Testing different feature combinations for realistic variance:\n")

# Test 1: More subtle BEC (Bank A misses, Bank C catches)
test1 = [0.45, 0.55, 0.70, 0.30, 0.80, 0.40, 0.50, 0.60, 0.75, 0.85,  # Lower amounts, mixed trust
         0.35, 0.45, 0.25, 0.40, 0.20, 0.60, 0.80, 0.85, 0.50, 0.45,  # Low geo risk, good identity  
         0.90, 0.45, 0.75, 0.65, 0.80, 0.70, 0.65, 0.70, 0.55, 0.50]   # Strong email, moderate network

result1 = consortium.generate_comparison_score(test1)
print("Test 1 - Subtle BEC:")
if isinstance(result1['individual_scores'], dict):
    scores = [float(v) for v in result1['individual_scores'].values()]
    print(f"  Individual: {[f'{s:.3f}' for s in scores]}")
else:
    print(f"  Individual: {[f'{float(s):.3f}' for s in result1['individual_scores']]}")
print(f"  Consensus: {result1['consensus_score']:.3f}")
print(f"  Variance: {result1['variance_score']:.3f}")

# Test 2: Even more subtle (borderline case)
test2 = [0.35, 0.45, 0.75, 0.40, 0.85, 0.35, 0.40, 0.70, 0.80, 0.90,  # Very low amounts, high trust
         0.25, 0.35, 0.15, 0.30, 0.10, 0.70, 0.85, 0.90, 0.40, 0.35,  # Very low geo risk  
         0.75, 0.35, 0.65, 0.55, 0.85, 0.75, 0.70, 0.75, 0.45, 0.40]   # Moderate email, low network

result2 = consortium.generate_comparison_score(test2)
print("\nTest 2 - Very Subtle:")
if isinstance(result2['individual_scores'], dict):
    scores = [float(v) for v in result2['individual_scores'].values()]
    print(f"  Individual: {[f'{s:.3f}' for s in scores]}")
else:
    print(f"  Individual: {[f'{float(s):.3f}' for s in result2['individual_scores']]}")
print(f"  Consensus: {result2['consensus_score']:.3f}")
print(f"  Variance: {result2['variance_score']:.3f}")

# Test 3: Mixed signals (some banks see fraud, others don't)
test3 = [0.25, 0.35, 0.85, 0.45, 0.90, 0.25, 0.30, 0.80, 0.85, 0.95,  # Small amount, high trust
         0.15, 0.25, 0.05, 0.20, 0.05, 0.80, 0.90, 0.95, 0.30, 0.25,  # Domestic, good identity  
         0.85, 0.25, 0.55, 0.45, 0.90, 0.80, 0.75, 0.80, 0.35, 0.30]   # High email risk only

result3 = consortium.generate_comparison_score(test3)
print("\nTest 3 - Mixed Signals:")
if isinstance(result3['individual_scores'], dict):
    scores = [float(v) for v in result3['individual_scores'].values()]
    print(f"  Individual: {[f'{s:.3f}' for s in scores]}")
else:
    print(f"  Individual: {[f'{float(s):.3f}' for s in result3['individual_scores']]}")
print(f"  Consensus: {result3['consensus_score']:.3f}")
print(f"  Variance: {result3['variance_score']:.3f}")

print(f"\nBest case for demo (highest variance): Test {[1,2,3][result3['variance_score'] > max(result1['variance_score'], result2['variance_score']) and 3 or (result2['variance_score'] > result1['variance_score'] and 2 or 1)]}")
