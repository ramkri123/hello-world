#!/usr/bin/env python3
"""
Enhanced Consortium Hub with Scenario-Aware Weighting
Handles different bank knowledge scenarios with appropriate confidence weights
"""

def determine_bank_scenario(bank_id, sender_account, receiver_account):
    """Determine what scenario each bank is in for this transaction"""
    
    # In a real system, this would check which bank owns which accounts
    # For demo purposes, we'll simulate realistic scenarios
    
    scenarios = {}
    
    # Bank A: Wire Transfer Specialist (often knows sender)
    if sender_account.startswith('BANK_A') and receiver_account.startswith('BANK_A'):
        scenarios['bank_A'] = 0  # Knows both
    elif sender_account.startswith('BANK_A'):
        scenarios['bank_A'] = 1  # Knows only sender
    elif receiver_account.startswith('BANK_A'):
        scenarios['bank_A'] = 2  # Knows only receiver
    else:
        scenarios['bank_A'] = 3  # Knows neither
    
    # Bank B: Identity Verification Specialist (often knows receiver)
    if sender_account.startswith('BANK_B') and receiver_account.startswith('BANK_B'):
        scenarios['bank_B'] = 0  # Knows both
    elif sender_account.startswith('BANK_B'):
        scenarios['bank_B'] = 1  # Knows only sender
    elif receiver_account.startswith('BANK_B'):
        scenarios['bank_B'] = 2  # Knows only receiver
    else:
        scenarios['bank_B'] = 3  # Knows neither
    
    # Bank C: Network Analysis (external perspective)
    # Typically knows neither account but has network intelligence
    scenarios['bank_C'] = 3  # Usually knows neither (network patterns only)
    
    return scenarios

def calculate_scenario_weights(scenarios):
    """Calculate confidence weights based on bank knowledge scenarios"""
    
    # Scenario confidence weights
    scenario_weights = {
        0: 1.0,  # Knows both accounts - highest confidence
        1: 0.8,  # Knows sender only - high confidence
        2: 0.7,  # Knows receiver only - medium-high confidence  
        3: 0.4   # Knows neither - lower confidence but still valuable
    }
    
    weights = {}
    for bank_id, scenario in scenarios.items():
        weights[bank_id] = scenario_weights[scenario]
    
    return weights

def calculate_weighted_consensus(individual_scores, weights):
    """Calculate weighted consensus based on bank confidence levels"""
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for bank_id, score in individual_scores.items():
        weight = weights.get(bank_id, 0.5)  # Default weight if not found
        total_weighted_score += score * weight
        total_weight += weight
    
    if total_weight > 0:
        weighted_consensus = total_weighted_score / total_weight
    else:
        weighted_consensus = 0.5  # Fallback
    
    return weighted_consensus

# Add to consortium hub's inference processing
def enhanced_inference_processing(self, session_id):
    """Enhanced inference processing with scenario-aware weighting"""
    
    if session_id not in self.active_sessions:
        return
    
    session = self.active_sessions[session_id]
    responses = session.responses
    
    if len(responses) < self.min_participants:
        return
    
    # Get transaction details for scenario determination
    transaction_data = session.raw_data.get('transaction_data', {})
    sender_account = transaction_data.get('sender_account', 'UNKNOWN')
    receiver_account = transaction_data.get('receiver_account', 'UNKNOWN')
    
    # Determine what scenario each bank is in
    bank_scenarios = determine_bank_scenario(
        None, sender_account, receiver_account
    )
    
    # Calculate confidence weights
    scenario_weights = calculate_scenario_weights(bank_scenarios)
    
    # Extract individual scores
    individual_scores = {pid: r['risk_score'] for pid, r in responses.items()}
    
    # Calculate weighted consensus
    weighted_consensus = calculate_weighted_consensus(individual_scores, scenario_weights)
    
    # Apply fraud pattern boost
    final_score = self._apply_fraud_pattern_boost(weighted_consensus, session)
    
    # Enhanced logging
    logger.info(f"   📊 SCENARIO-AWARE CONSENSUS:")
    for bank_id, scenario in bank_scenarios.items():
        weight = scenario_weights.get(bank_id, 0.5)
        score = individual_scores.get(bank_id, 0.5)
        weighted_score = score * weight
        scenario_names = [
            "Knows both accounts",
            "Knows sender only", 
            "Knows receiver only",
            "Knows neither account"
        ]
        logger.info(f"     {bank_id}: Scenario {scenario} ({scenario_names[scenario]})")
        logger.info(f"       Score: {score:.3f}, Weight: {weight:.2f}, Weighted: {weighted_score:.3f}")
    
    logger.info(f"   🎯 Weighted Consensus: {weighted_consensus:.3f}")
    logger.info(f"   🚨 Final Score (with pattern boost): {final_score:.3f}")
    
    # Continue with rest of processing...
    variance = sum((r['risk_score'] - weighted_consensus) ** 2 for r in responses.values()) / len(responses)
    
    # Rest of the consensus logic...
