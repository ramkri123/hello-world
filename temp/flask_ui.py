#!/usr/bin/env python3
"""
Flask-based Consortium Fraud Detection UI
Simple web interface for the one-way hash anonymization system
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from consortium.account_anonymizer import AccountAnonymizer
import numpy as np

app = Flask(__name__)

# Sample bank accounts for demo
SAMPLE_BANK_ACCOUNTS = {
    'bank_a': ['ACCA12345', 'ACCA67890', 'ACCA11111'],
    'bank_b': ['ACCB67890', 'ACCB12345', 'ACCB22222'],
    'bank_c': ['ACCC99999', 'ACCC12345', 'ACCC55555']
}

def analyze_fraud_risk(email_content, amount, transaction_type):
    """Analyze fraud risk based on email content and transaction details"""
    indicators = {
        'urgency_keywords': 0,
        'authority_impersonation': 0,
        'secrecy_demands': 0,
        'unusual_amount': 0,
        'crypto_mentions': 0,
        'total_risk': 0
    }
    
    email_lower = email_content.lower()
    
    # Check for urgency keywords
    urgency_words = ['urgent', 'immediately', 'asap', 'deadline', 'expires', 'limited time', 'hurry']
    indicators['urgency_keywords'] = sum(1 for word in urgency_words if word in email_lower) / len(urgency_words)
    
    # Check for authority impersonation
    authority_words = ['ceo', 'president', 'director', 'manager', 'boss', 'executive']
    indicators['authority_impersonation'] = sum(1 for word in authority_words if word in email_lower) / len(authority_words)
    
    # Check for secrecy demands
    secrecy_words = ['confidential', 'secret', "don't tell", 'private', 'discreet', "don't discuss"]
    indicators['secrecy_demands'] = sum(1 for word in secrecy_words if word in email_lower) / len(secrecy_words)
    
    # Check for unusual amount (>$100k is considered high risk)
    indicators['unusual_amount'] = min(amount / 100000, 1.0)
    
    # Check for crypto mentions
    crypto_words = ['bitcoin', 'crypto', 'cryptocurrency', 'btc', 'ethereum', 'blockchain']
    indicators['crypto_mentions'] = sum(1 for word in crypto_words if word in email_lower) / len(crypto_words)
    
    # Calculate total risk
    indicators['total_risk'] = (
        indicators['urgency_keywords'] * 0.3 +
        indicators['authority_impersonation'] * 0.25 +
        indicators['secrecy_demands'] * 0.2 +
        indicators['unusual_amount'] * 0.15 +
        indicators['crypto_mentions'] * 0.1
    )
    
    return indicators

def generate_bank_risk_score(fraud_indicators, bank_name, scenario):
    """Generate bank-specific risk score based on fraud indicators and scenario"""
    base_risk = fraud_indicators['total_risk']
    
    # Bank-specific adjustments (different banks have different detection capabilities)
    bank_adjustments = {
        'bank_a': {'ceo_fraud': 1.2, 'crypto': 0.8, 'baseline': 1.0},
        'bank_b': {'ceo_fraud': 1.0, 'crypto': 1.3, 'baseline': 1.1},
        'bank_c': {'ceo_fraud': 0.9, 'crypto': 1.1, 'baseline': 1.2}
    }
    
    adjustment = bank_adjustments[bank_name]['baseline']
    
    # Adjust based on fraud type
    if fraud_indicators['authority_impersonation'] > 0.5:
        adjustment *= bank_adjustments[bank_name]['ceo_fraud']
    if fraud_indicators['crypto_mentions'] > 0.3:
        adjustment *= bank_adjustments[bank_name]['crypto']
    
    # Scenario-based adjustment (banks know their customers better)
    if scenario in ['knows_both', 'knows_sender']:
        adjustment *= 1.1  # Better detection for known customers
    elif scenario == 'knows_neither':
        adjustment *= 0.9  # Slightly worse detection for unknown customers
    
    # Add some realistic randomness
    import hashlib
    seed = int(hashlib.md5(f"{bank_name}{base_risk}".encode()).hexdigest()[:8], 16)
    np.random.seed(seed % (2**32))
    noise = np.random.uniform(0.85, 1.15)
    
    final_score = min(base_risk * adjustment * noise, 1.0)
    return max(final_score, 0.05)  # Minimum 5% risk score

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>🏦 Consortium Fraud Detection</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background-color: #f5f7fa;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        }
        .card { 
            background: white; padding: 1.5rem; border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;
        }
        .bank-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
        .bank-card { border-left: 4px solid #667eea; }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: 600; }
        .form-group input, .form-group select { 
            width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; 
        }
        .btn { 
            background: #667eea; color: white; padding: 0.75rem 1.5rem; 
            border: none; border-radius: 4px; cursor: pointer; font-size: 1rem;
        }
        .btn:hover { background: #5a6fd8; }
        .metric { text-align: center; padding: 1rem; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; margin-top: 0.5rem; }
        .alert { padding: 1rem; border-radius: 4px; margin: 1rem 0; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .alert-danger { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .code-block { background: #f8f9fa; padding: 1rem; border-radius: 4px; font-family: monospace; }
        #results { display: none; }
        .scenario-badge {
            display: inline-block; padding: 0.25rem 0.5rem; border-radius: 12px;
            font-size: 0.8rem; font-weight: bold; color: white; margin-left: 0.5rem;
        }
        .knows-both { background: #28a745; }
        .knows-sender { background: #007bff; }
        .knows-receiver { background: #ffc107; color: #000; }
        .knows-neither { background: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 Consortium Fraud Detection Dashboard</h1>
            <p>Privacy-Preserving Fraud Detection with One-Way Hash Anonymization</p>
        </div>

        <div class="card">
            <h2>🔧 Transaction Analysis</h2>
            <form id="analysisForm">
                <div class="form-group">
                    <label for="scenario">🎭 Fraud Scenario:</label>
                    <select id="scenario" onchange="loadScenario()">
                        <option value="custom">Custom Transaction</option>
                        <option value="ceo_bec">🏢 CEO Business Email Compromise</option>
                        <option value="crypto_scam">₿ Cryptocurrency Investment Scam</option>
                        <option value="ransomware">🔒 Ransomware Payment</option>
                        <option value="money_laundering">💰 Money Laundering</option>
                        <option value="vendor_fraud">🏭 Vendor Payment Fraud</option>
                        <option value="romance_scam">💕 Romance Scam</option>
                        <option value="invoice_fraud">📄 Fake Invoice Fraud</option>
                        <option value="lottery_scam">🎰 Lottery/Prize Scam</option>
                        <option value="tech_support">💻 Tech Support Scam</option>
                        <option value="social_engineering">🎭 Social Engineering Attack</option>
                        <option value="shell_company">🏗️ Shell Company Laundering</option>
                        <option value="investment_ponzi">📈 Ponzi Investment Scheme</option>
                        <option value="legitimate">✅ Legitimate Business Payment</option>
                    </select>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem;">
                    <div class="form-group">
                        <label for="sender">Sender Account:</label>
                        <input type="text" id="sender" value="ACCA12345" required>
                    </div>
                    <div class="form-group">
                        <label for="receiver">Receiver Account:</label>
                        <input type="text" id="receiver" value="ACCB67890" required>
                    </div>
                    <div class="form-group">
                        <label for="amount">Amount ($):</label>
                        <input type="number" id="amount" value="5000" min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="type">Transaction Type:</label>
                        <select id="type">
                            <option value="wire_transfer">Wire Transfer</option>
                            <option value="ach">ACH</option>
                            <option value="card_payment">Card Payment</option>
                            <option value="check">Check</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="email">📧 Email Content (associated with transaction):</label>
                    <textarea id="email" rows="4" placeholder="Email content associated with this transaction...">Regular monthly vendor payment for office supplies and equipment maintenance.</textarea>
                </div>
                <button type="submit" class="btn">🔍 Analyze Transaction</button>
            </form>
        </div>

        <div id="results">
            <div class="card">
                <h2>🔐 Anonymization Results</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                    <div>
                        <h3>Original Accounts</h3>
                        <div class="code-block" id="originalAccounts"></div>
                    </div>
                    <div>
                        <h3>Anonymized Accounts</h3>
                        <div class="code-block" id="anonymizedAccounts"></div>
                    </div>
                </div>
                <div class="alert alert-success">
                    🔒 <strong>Privacy Protection:</strong> Original account numbers are anonymized using SHA256 one-way hash. Cannot be reverse-engineered!
                </div>
            </div>

            <div class="card">
                <h2>🏦 Bank Analysis Results</h2>
                <div class="bank-grid" id="bankResults"></div>
            </div>

            <div class="card">
                <h2>🎯 Consortium Consensus</h2>
                <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 2rem;">
                    <div>
                        <h3>Weighted Calculation</h3>
                        <div class="code-block" id="consensusCalculation"></div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="consensusScore">0.000</div>
                        <div class="metric-label">Consensus Risk Score</div>
                        <div id="recommendation" class="alert"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>🛡️ Privacy Verification</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                    <div>
                        <h3>✅ Privacy Properties Verified</h3>
                        <ul>
                            <li>🔒 <strong>One-way hash:</strong> Cannot reverse-engineer original accounts</li>
                            <li>🎯 <strong>Deterministic:</strong> Same account always produces same hash</li>
                            <li>🔀 <strong>No patterns:</strong> Bank prefixes completely hidden</li>
                            <li>🏦 <strong>Bank independence:</strong> Banks determine scenarios themselves</li>
                        </ul>
                    </div>
                    <div>
                        <h3>🔍 What External Observer Sees</h3>
                        <div class="code-block" id="observerView"></div>
                        <p><em>⚠️ Observer CANNOT determine which banks own the accounts!</em></p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📈 System Status</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div class="alert alert-success">🟢 Consortium Hub: Online</div>
                <div class="alert alert-success">🟢 Anonymization: Active</div>
                <div class="alert alert-success">🟢 Bank A: Connected</div>
                <div class="alert alert-success">🟢 Bank B: Connected</div>
                <div class="alert alert-success">🟢 Bank C: Connected</div>
            </div>
        </div>
    </div>

    <script>
        // Predefined fraud scenarios
        const scenarios = {
            'ceo_bec': {
                sender: 'ACCA12345',
                receiver: 'ACCB67890',
                amount: 485000,
                type: 'wire_transfer',
                email: `Hi John,

This is CEO Sarah Wilson. We have an urgent strategic acquisition opportunity that requires immediate action. Please wire $485,000 to our new strategic partner Global Tech Solutions for the acquisition deposit.

This is highly confidential and time sensitive - we need to complete this before market close Friday. Please process this immediately and don't discuss with anyone else on the team.

Thanks for your help with this critical transaction.

Best regards,
Sarah Wilson
CEO`
            },
            'crypto_scam': {
                sender: 'ACCA67890',
                receiver: 'ACCB22222',
                amount: 250000,
                type: 'wire_transfer',
                email: `URGENT: Crypto Investment Opportunity - LIMITED TIME

Dear Investor,

This is a once-in-a-lifetime opportunity to invest in the new CryptoMax currency before it goes public. Our algorithm has identified massive profit potential.

Send $250,000 IMMEDIATELY to secure your position. This offer expires in 4 hours. Don't miss out on 10x returns!

Wire to our secure account for immediate processing. Complete confidentiality guaranteed.

Best regards,
Investment Director`
            },
            'ransomware': {
                sender: 'ACCB12345',
                receiver: 'ACCC99999',
                amount: 75000,
                type: 'wire_transfer',
                email: `Your files have been encrypted. Pay $75,000 in Bitcoin equivalent to decrypt your systems. Wire transfer to this account for Bitcoin purchase. You have 48 hours before files are permanently deleted. Do not contact authorities.`
            },
            'money_laundering': {
                sender: 'ACCC12345',
                receiver: 'ACCA11111',
                amount: 950000,
                type: 'wire_transfer',
                email: `Payment for consulting services rendered Q4 2024. International business development and market analysis project completion. Reference: CONSULTING-2024-Q4-FINAL.`
            },
            'vendor_fraud': {
                sender: 'ACCB67890',
                receiver: 'ACCC55555',
                amount: 125000,
                type: 'ach',
                email: `URGENT: Updated banking details for TechSupply Corp. Our bank has changed. Please update payment details immediately for invoice #TSC-2024-1127. New account required for all future payments. Process immediately to avoid service disruption.`
            },
            'romance_scam': {
                sender: 'ACCA67890',
                receiver: 'ACCB22222',
                amount: 35000,
                type: 'wire_transfer',
                email: `My darling, I'm stuck in Dubai and need $35,000 for emergency medical treatment. The doctors won't treat me without payment upfront. Please wire the money immediately. I'll pay you back as soon as I return home. I love you so much.`
            },
            'legitimate': {
                sender: 'ACCA12345',
                receiver: 'ACCB67890',
                amount: 50000,
                type: 'ach',
                email: `Monthly vendor payment for office supplies and equipment maintenance. Invoice #12455 dated November 15, 2024. Net 30 payment terms. Standard business transaction per our service agreement.`
            },
            'invoice_fraud': {
                sender: 'ACCB12345',
                receiver: 'ACCC77777',
                amount: 180000,
                type: 'wire_transfer',
                email: `RE: Invoice #INV-2024-8832 - URGENT PAYMENT REQUIRED

Dear Accounts Payable,

Our records indicate that Invoice #INV-2024-8832 for $180,000 is now 45 days overdue. We have updated our banking details due to a merger.

NEW PAYMENT DETAILS:
Account: ACCC77777
Reference: INV-2024-8832

Please process payment immediately to avoid service interruption and late fees. Our legal team has been notified of this overdue payment.

Best regards,
Global Services Ltd
Accounts Receivable`
            },
            'lottery_scam': {
                sender: 'ACCA11111',
                receiver: 'ACCB44444',
                amount: 25000,
                type: 'wire_transfer',
                email: `🎉 CONGRATULATIONS! YOU'VE WON THE INTERNATIONAL LOTTERY! 🎉

Dear Lucky Winner,

You have been selected as the winner of $2,500,000 in the International Business Lottery! Your winning number is: ILB-2024-7749.

To claim your prize, you must pay the processing fee of $25,000 immediately. This covers:
- Government taxes and fees
- International transfer charges  
- Prize verification costs

Wire the fee to account ACCB44444 within 48 hours or forfeit your winnings!

Congratulations again!
International Lottery Commission`
            },
            'tech_support': {
                sender: 'ACCB22222',
                receiver: 'ACCC66666',
                amount: 85000,
                type: 'wire_transfer',
                email: `URGENT: Microsoft Security Alert - Immediate Action Required

Dear Customer,

Our security systems have detected multiple unauthorized access attempts to your business Microsoft account. Your system has been compromised and hackers are currently accessing your financial data.

You must purchase Microsoft Security Pro licenses immediately to protect your business:
- Premium Security Suite: $85,000
- Immediate activation required

Wire payment to our secure account ACCC66666. Reference: MSFT-SEC-2024

DO NOT use credit cards as hackers may have compromised them. Only wire transfer is secure.

Call our emergency helpline: 1-800-FAKE-HELP

Microsoft Security Team`
            },
            'social_engineering': {
                sender: 'ACCC12345',
                receiver: 'ACCA33333',
                amount: 320000,
                type: 'wire_transfer',
                email: `CONFIDENTIAL: Board Resolution - Project Phoenix

Dear CFO,

Following the emergency board meeting this morning, we need to execute the confidential Project Phoenix acquisition immediately.

The board has authorized the $320,000 deposit for the Shell Creek Properties acquisition. Due to the sensitive nature of this deal and regulatory requirements, this must be processed today before the announcement.

Wire to holding account ACCA33333 immediately. Use reference: PHOENIX-DEPOT-2024

This information is strictly confidential until the public announcement next week.

Best regards,
Board Secretary
On behalf of the Board of Directors`
            },
            'shell_company': {
                sender: 'ACCB55555',
                receiver: 'ACCC12345',
                amount: 775000,
                type: 'wire_transfer',
                email: `International Consulting Agreement - Final Payment

Reference: Contract ICA-2024-009
Project: Market Development Services - Eastern Europe

Final payment for completion of market analysis and business development services across Poland, Czech Republic, and Slovakia markets.

Deliverables completed:
✓ Market penetration analysis
✓ Regulatory compliance framework  
✓ Distribution network establishment
✓ Key partnership agreements

Payment amount: $775,000
Wire to: ACCC12345
Reference: ICA-2024-009-FINAL

Net 30 terms as per contract.

Best regards,
Eastern European Consulting Ltd.`
            },
            'investment_ponzi': {
                sender: 'ACCA22222',
                receiver: 'ACCB11111',
                amount: 450000,
                type: 'wire_transfer',
                email: `EXCLUSIVE: Quantum AI Trading Fund - Early Investor Opportunity

Dear High-Net-Worth Investor,

Congratulations! You've been selected for our exclusive Quantum AI Trading Fund based on your investment profile.

Our proprietary AI algorithm has generated:
- 47% returns in Q3 2024
- 52% returns in Q4 2024  
- Projected 65% for Q1 2025

Minimum investment: $450,000
Expected annual return: 180%
Risk level: Guaranteed principal protection

This opportunity is limited to 25 investors worldwide. We have 3 spots remaining.

Wire your investment to ACCB11111 by midnight tonight to secure your position.

Best regards,
Dr. Michael Chen, PhD
Quantum Financial Technologies`
            }
        };

        function loadScenario() {
            const scenario = document.getElementById('scenario').value;
            if (scenario === 'custom') return;
            
            const data = scenarios[scenario];
            if (data) {
                document.getElementById('sender').value = data.sender;
                document.getElementById('receiver').value = data.receiver;
                document.getElementById('amount').value = data.amount;
                document.getElementById('type').value = data.type;
                document.getElementById('email').value = data.email;
            }
        }

        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const data = {
                sender: document.getElementById('sender').value,
                receiver: document.getElementById('receiver').value,
                amount: document.getElementById('amount').value,
                type: document.getElementById('type').value,
                email: document.getElementById('email').value
            };
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                displayResults(result);
                document.getElementById('results').style.display = 'block';
            } catch (error) {
                alert('Error analyzing transaction: ' + error.message);
            }
        });

        function displayResults(result) {
            // Original and anonymized accounts
            document.getElementById('originalAccounts').textContent = 
                `Sender: ${result.original.sender}\\nReceiver: ${result.original.receiver}`;
            document.getElementById('anonymizedAccounts').textContent = 
                `Sender: ${result.anonymized.sender_anonymous}\\nReceiver: ${result.anonymized.receiver_anonymous}`;
            
            // Bank results
            const bankResultsDiv = document.getElementById('bankResults');
            bankResultsDiv.innerHTML = '';
            
            Object.entries(result.banks).forEach(([bankName, bankData]) => {
                const scenarioClass = bankData.scenario.replace('_', '-');
                const scenarioBadge = `<span class="scenario-badge ${scenarioClass}">${bankData.scenario}</span>`;
                
                bankResultsDiv.innerHTML += `
                    <div class="card bank-card">
                        <h3>🏛️ ${bankName.toUpperCase()} ${scenarioBadge}</h3>
                        <div class="metric">
                            <div class="metric-value">${bankData.confidence.toFixed(2)}</div>
                            <div class="metric-label">Confidence Weight</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${bankData.base_risk.toFixed(3)}</div>
                            <div class="metric-label">Base Risk Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${bankData.weighted_risk.toFixed(3)}</div>
                            <div class="metric-label">Weighted Risk</div>
                        </div>
                    </div>
                `;
            });
            
            // Consensus
            document.getElementById('consensusCalculation').textContent = result.consensus.calculation;
            document.getElementById('consensusScore').textContent = result.consensus.score.toFixed(3);
            
            const recommendationDiv = document.getElementById('recommendation');
            if (result.consensus.score < 0.3) {
                recommendationDiv.className = 'alert alert-success';
                recommendationDiv.textContent = '✅ APPROVE - Low fraud risk';
            } else if (result.consensus.score < 0.6) {
                recommendationDiv.className = 'alert alert-warning';
                recommendationDiv.textContent = '⚠️ REVIEW - Medium fraud risk';
            } else {
                recommendationDiv.className = 'alert alert-danger';
                recommendationDiv.textContent = '🚨 BLOCK - High fraud risk';
            }
            
            // Observer view
            document.getElementById('observerView').textContent = JSON.stringify(result.observer_view, null, 2);
        }
    </script>
</body>
</html>
"""

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    
    sender_account = data.get('sender', 'ACCA12345')
    receiver_account = data.get('receiver', 'ACCB67890')
    amount = float(data.get('amount', 5000))
    transaction_type = data.get('type', 'wire_transfer')
    email_content = data.get('email', '')
    
    # Step 1: Anonymization
    anonymized = AccountAnonymizer.anonymize_transaction_accounts(sender_account, receiver_account)
    
    # Step 2: Fraud risk analysis based on email content and transaction details
    fraud_indicators = analyze_fraud_risk(email_content, amount, transaction_type)
    
    # Step 3: Bank scenario determination with realistic risk scoring
    bank_results = {}
    total_weighted_risk = 0
    total_confidence = 0
    
    for bank_name, accounts in SAMPLE_BANK_ACCOUNTS.items():
        scenario = AccountAnonymizer.bank_can_determine_ownership(accounts, anonymized)
        confidence = AccountAnonymizer.get_scenario_confidence_weight(scenario)
        
        # Generate risk score based on fraud indicators and bank-specific factors
        base_risk = generate_bank_risk_score(fraud_indicators, bank_name, scenario)
        weighted_risk = base_risk * confidence
        
        bank_results[bank_name] = {
            'scenario': scenario,
            'confidence': confidence,
            'base_risk': base_risk,
            'weighted_risk': weighted_risk
        }
        
        total_weighted_risk += weighted_risk
        total_confidence += confidence
    
    # Step 4: Consensus
    consensus_score = total_weighted_risk / total_confidence if total_confidence > 0 else 0
    
    # Build calculation string
    calculation_lines = []
    for bank_name, result in bank_results.items():
        calculation_lines.append(f"{bank_name}: {result['base_risk']:.3f} × {result['confidence']:.2f} = {result['weighted_risk']:.3f}")
    calculation_lines.append(f"\\nConsensus: ({total_weighted_risk:.3f}) / ({total_confidence:.2f}) = {consensus_score:.3f}")
    
    # Observer view (what external observer can see)
    observer_view = {
        'sender_anonymous': anonymized['sender_anonymous'],
        'receiver_anonymous': anonymized['receiver_anonymous'],
        'amount': amount,
        'type': transaction_type,
        'email_length': len(email_content),
        'email_preview': email_content[:50] + '...' if len(email_content) > 50 else email_content
    }
    
    return jsonify({
        'original': {
            'sender': sender_account,
            'receiver': receiver_account
        },
        'anonymized': anonymized,
        'banks': bank_results,
        'consensus': {
            'score': consensus_score,
            'calculation': '\\n'.join(calculation_lines)
        },
        'fraud_indicators': fraud_indicators,
        'observer_view': observer_view
    })

if __name__ == '__main__':
    print("🏦 Starting Consortium Fraud Detection UI...")
    print("📱 Access the dashboard at: http://localhost:5000")
    print("🔒 Privacy-preserving fraud detection with one-way hash anonymization")
    app.run(host='0.0.0.0', port=5000, debug=True)
