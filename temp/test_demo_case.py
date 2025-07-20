from consortium_comparison_score_prototype import BankSimulator, ConsortiumComparisonService

# Create consortium and banks
consortium = ConsortiumComparisonService()

# Load pre-trained banks
bank_A = BankSimulator('bank_A', 'bank_A_data.csv', 'models/bank_A_model.pkl')
bank_A.load_model()

bank_B = BankSimulator('bank_B', 'bank_B_data.csv', 'models/bank_B_model.pkl')  
bank_B.load_model()

bank_C = BankSimulator('bank_C', 'bank_C_data.csv', 'models/bank_C_model.pkl')
bank_C.load_model()

# Register banks
consortium.register_bank('bank_A', bank_A)
consortium.register_bank('bank_B', bank_B)
consortium.register_bank('bank_C', bank_C)

# Demo BEC case features - adjusted to trigger higher fraud scores
bec_demo = [0.85, 0.90, 0.15, 0.95, 0.90, 0.85, 0.95, 0.85, 0.10, 0.15,  # High amount, urgency, low trust
            0.75, 0.80, 0.90, 0.85, 0.95, 0.90, 0.05, 0.10, 0.95, 0.90,  # Geographic risk, low verification  
            0.95, 0.85, 0.95, 0.90, 0.15, 0.10, 0.05, 0.05, 0.95, 0.90]   # Email fraud, urgency, network patterns

print('=== Testing Demo BEC Case ===')
result = consortium.generate_comparison_score(bec_demo)
print(f'Final Score: {result["final_score"]:.3f}')
print(f'Individual Scores: {[f"{s:.3f}" for s in result["individual_scores"]]}')
print(f'Consensus: {result["consensus_score"]:.3f}')
print(f'Variance: {result["variance_score"]:.3f}')
print(f'Network Anomaly: {result["network_anomaly_score"]:.3f}')
print(f'Recommendation: {result["recommendation"]}')

# Test with more extreme fraud features for comparison
print('\n=== Testing High Fraud Case ===')
high_fraud = [0.95, 0.95, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05,
              0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.95, 0.95,
              0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95]

result2 = consortium.generate_comparison_score(high_fraud)
print(f'Final Score: {result2["final_score"]:.3f}')
print(f'Individual Scores: {[f"{s:.3f}" for s in result2["individual_scores"]]}')
print(f'Recommendation: {result2["recommendation"]}')
