from consortium_comparison_score_prototype import generate_cross_institutional_fraud_scenarios

scenarios = generate_cross_institutional_fraud_scenarios()
for i, (name, features, explanation) in enumerate(scenarios[:3]):
    print(f'Scenario {i+1}: {name}')
    print(f'Explanation: {explanation[:150]}...')
    print(f'High-risk features (>0.8): {[i for i, f in enumerate(features) if f > 0.8]}')
    print(f'Features sample: {features[:10]}')
    print('-' * 50)
