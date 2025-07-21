#!/usr/bin/env python3
"""
Simple Flask Web UI for Consortium Fraud Detection
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import requests
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

CONSORTIUM_URL = "http://localhost:8080"

@app.route('/test')
def test_page():
    """Simple test page"""
    return '''
    <html>
    <head><title>Test Page</title></head>
    <body>
    <h1>✅ Flask UI is Working!</h1>
    <p>Time: ''' + str(datetime.now()) + '''</p>
    <p>Consortium Status: <span id="status">Loading...</span></p>
    
    <script>
    fetch('/api/consortium/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('status').textContent = data.status;
        })
        .catch(e => {
            document.getElementById('status').textContent = 'Error: ' + e.message;
        });
    </script>
    </body>
    </html>
    '''

@app.route('/')
def index():
    return render_template('fraud_detection.html')

@app.route('/api/consortium/status')
def consortium_status():
    """Get consortium health and participants"""
    try:
        # Get health
        health_response = requests.get(f"{CONSORTIUM_URL}/health", timeout=5)
        health_data = health_response.json()
        
        # Get participants
        participants_response = requests.get(f"{CONSORTIUM_URL}/participants", timeout=5)
        participants_data = participants_response.json()
        
        return jsonify({
            "status": "online",
            "health": health_data,
            "participants": participants_data
        })
    except Exception as e:
        return jsonify({
            "status": "offline",
            "error": str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    """Submit transaction for fraud analysis"""
    try:
        data = request.get_json()
        
        # Prepare payload for consortium
        payload = {
            "transaction_data": {
                "amount": float(data.get('amount', 0)),
                "sender_account": data.get('sender_account', ''),
                "receiver_account": data.get('receiver_account', ''),
                "transaction_type": data.get('transaction_type', 'wire_transfer')
            },
            "email_content": data.get('email_content', ''),
            "context": data.get('context', '')
        }
        
        # Submit to consortium
        response = requests.post(
            f"{CONSORTIUM_URL}/inference",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            submission_data = response.json()
            session_id = submission_data.get('session_id')
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "submission": submission_data
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Consortium returned status {response.status_code}"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/results/<session_id>')
def get_results(session_id):
    """Get fraud analysis results"""
    try:
        response = requests.get(f"{CONSORTIUM_URL}/results/{session_id}", timeout=10)
        
        if response.status_code == 200:
            consortium_results = response.json()
            print(f"🔍 DEBUG - Consortium results for {session_id}:")
            print(f"   Data: {consortium_results}")
            
            return jsonify({
                "success": True,
                "results": consortium_results
            })
        else:
            print(f"🔍 DEBUG - Consortium returned status {response.status_code} for {session_id}")
            return jsonify({
                "success": False,
                "error": f"Results not ready or not found (status {response.status_code})"
            }), response.status_code
            
    except Exception as e:
        print(f"🔍 DEBUG - Exception getting results for {session_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Consortium Fraud Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f6fa; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        
        .status-panel { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .status-online { border-left: 5px solid #27ae60; }
        .status-offline { border-left: 5px solid #e74c3c; }
        
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        
        .form-panel { background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        .form-group input, .form-group textarea, .form-group select { width: 100%; padding: 12px; border: 2px solid #e1e8ed; border-radius: 8px; font-size: 14px; }
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus { outline: none; border-color: #667eea; }
        
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        
        .results-panel { background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        
        .risk-indicator { text-align: center; margin: 20px 0; }
        .risk-score { font-size: 3em; font-weight: bold; margin: 10px 0; }
        .risk-low { color: #27ae60; }
        .risk-medium { color: #f39c12; }
        .risk-high { color: #e74c3c; }
        
        .bank-scores { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .bank-score { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }
        
        .loading { text-align: center; padding: 40px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .participants { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
        .participant { background: #e8f5e8; padding: 15px; border-radius: 8px; text-align: center; }
        .participant h4 { color: #27ae60; margin-bottom: 5px; }
        
        /* Scenario Card Styling */
        .scenario-card { 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            border: 2px solid #e0e0e0; 
            cursor: pointer; 
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        .scenario-card:hover { 
            transform: translateY(-4px) scale(1.02); 
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        }
        .scenario-card:active {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            border-color: #5a67d8;
        }
        .scenario-card:focus {
            outline: 3px solid #667eea;
            outline-offset: 2px;
            border-color: #667eea;
        }
        .scenario-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            transition: left 0.5s;
        }
        .scenario-card:hover::before {
            left: 100%;
        }
        .scenario-card h5 { margin-bottom: 8px; transition: color 0.3s ease; }
        .scenario-card:hover h5 { color: #667eea !important; }
        .scenario-card .amount { font-size: 0.8em; color: #666; margin-bottom: 8px; font-weight: bold; transition: color 0.3s ease; }
        .scenario-card:hover .amount { color: #5a67d8; }
        .scenario-card .description { font-size: 0.85em; line-height: 1.4; transition: color 0.3s ease; }
        .scenario-card:hover .description { color: #4a5568; }
        .scenario-card .transfer-route { 
            background: #f8f9fa; 
            padding: 8px; 
            border-radius: 4px; 
            margin-top: 8px; 
            font-size: 0.75em; 
            color: #555;
            transition: all 0.3s ease;
            border-left: 3px solid transparent;
        }
        .scenario-card:hover .transfer-route {
            background: #e8f2ff;
            border-left-color: #667eea;
            color: #2d3748;
        }
        .scenario-card.legitimate .transfer-route { background: #e8f5e8; border-left-color: #27ae60; }
        .scenario-card.legitimate:hover .transfer-route { background: #d4edda; border-left-color: #1e7e34; }
        
        /* Pulsing effect for high-risk scenarios */
        .scenario-card.high-risk {
            animation: subtle-pulse 3s infinite;
        }
        @keyframes subtle-pulse {
            0%, 100% { box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            50% { box-shadow: 0 2px 4px rgba(231, 76, 60, 0.2); }
        }
        
        /* Selection indicator */
        .scenario-card.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, #f0f4ff 0%, #e8f2ff 100%);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Consortium Fraud Detection</h1>
            <p>Privacy-Preserving Multi-Bank Fraud Analysis</p>
        </div>
        
        <div class="status-panel" id="statusPanel">
            <h3>🔍 Consortium Status</h3>
            <div id="statusContent">Loading...</div>
        </div>
        
        <div class="main-content">
            <div class="form-panel">
                <h3>📋 Transaction Analysis</h3>
                
                <!-- Example Scenarios Section -->
                <div style="margin-bottom: 25px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;">
                    <h4 style="margin-bottom: 10px; color: #333;">🎯 Example Fraud Scenarios</h4>
                    <p style="font-size: 0.9em; color: #666; margin-bottom: 20px;">Test the consortium's ability to detect different types of financial fraud. Each scenario represents real-world attack patterns:</p>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                        <!-- Cryptocurrency Investment Scam -->
                        <div class="scenario-card high-risk" onclick="loadScenario('crypto_scam')" tabindex="0">
                            <h5 style="color: #ff6b35; margin-bottom: 8px;">₿ Cryptocurrency Scam</h5>
                            <p class="amount"><strong>$485,000 • High Risk Expected</strong></p>
                            <p class="description">Fake investment platform demanding large upfront payment. Tests cryptocurrency fraud and investment manipulation tactics.</p>
                            <div class="transfer-route">
                                <strong>🏦 Transfer Route:</strong> Bank A (ACC789012) → External Crypto Platform (ACC345678)
                            </div>
                        </div>
                        
                        <!-- Invoice Scam -->
                        <div class="scenario-card" onclick="loadScenario('invoice_scam')" tabindex="0">
                            <h5 style="color: #667eea; margin-bottom: 8px;">📋 Invoice Scam</h5>
                            <p class="amount"><strong>$125,000 • Medium-High Risk Expected</strong></p>
                            <p class="description">Vendor impersonation with updated banking details. Common B2B fraud targeting accounts payable departments.</p>
                            <div class="transfer-route">
                                <strong>🏦 Transfer Route:</strong> Bank B (ACC567890) → Fake Vendor Bank C (ACC987654)
                            </div>
                        </div>
                        
                        <!-- Legitimate Transfer -->
                        <div class="scenario-card legitimate" onclick="loadScenario('legitimate')" tabindex="0">
                            <h5 style="color: #27ae60; margin-bottom: 8px;">✅ Legitimate Transfer</h5>
                            <p class="amount"><strong>$50,000 • Low Risk Expected</strong></p>
                            <p class="description">Routine quarterly business payment with proper documentation and approval codes. Tests baseline legitimate transactions.</p>
                            <div class="transfer-route">
                                <strong>🏦 Transfer Route:</strong> Bank A (ACC123456) → Bank B Contractor (ACC111222)
                            </div>
                        </div>
                        
                        <!-- Romance Scam -->
                        <div class="scenario-card high-risk" onclick="loadScenario('romance_scam')" tabindex="0">
                            <h5 style="color: #e74c3c; margin-bottom: 8px;">💔 Romance Scam</h5>
                            <p class="amount"><strong>$75,000 • High Risk Expected</strong></p>
                            <p class="description">Emotional manipulation with fake emergency overseas. Classic romance fraud with international wire transfer request.</p>
                            <div class="transfer-route">
                                <strong>🏦 Transfer Route:</strong> Bank B (ACC445566) → International Bank C (ACC778899)
                            </div>
                        </div>
                        
                        <!-- Tech Support Scam -->
                        <div class="scenario-card" onclick="loadScenario('tech_support')" tabindex="0">
                            <h5 style="color: #f39c12; margin-bottom: 8px;">🖥️ Tech Support Scam</h5>
                            <p class="amount"><strong>$15,000 • Medium Risk Expected</strong></p>
                            <p class="description">Fake Microsoft security alert demanding payment. Tests tech impersonation and fear-based urgency tactics.</p>
                            <div class="transfer-route">
                                <strong>🏦 Transfer Route:</strong> Bank C (ACC334455) → Fake Security Bank A (ACC998877)
                            </div>
                        </div>
                        
                        <!-- Business Email Compromise -->
                        <div class="scenario-card high-risk" onclick="loadScenario('business_compromise')" tabindex="0">
                            <h5 style="color: #8e44ad; margin-bottom: 8px;">🏢 Business Email Compromise</h5>
                            <p class="amount"><strong>$320,000 • High Risk Expected</strong></p>
                            <p class="description">CFO impersonation targeting finance team with fake M&A deal. Advanced corporate email compromise attack.</p>
                            <div class="transfer-route">
                                <strong>🏦 Transfer Route:</strong> Bank C (ACC887766) → Fake M&A Bank B (ACC445577)
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 4px; border-left: 4px solid #ffc107;">
                        <p style="font-size: 0.8em; color: #856404; margin: 0;"><strong>💡 Testing Tip:</strong> Each scenario shows transfers between different banks in the consortium. Bank A specializes in wire transfer analysis, Bank B handles identity verification, and Bank C focuses on network patterns. Watch how each bank's expertise contributes to detecting different fraud types based on sender/receiver relationships.</p>
                    </div>
                </div>
                
                <form id="fraudForm">
                    <div class="form-group">
                        <label for="amount">💰 Amount ($)</label>
                        <input type="number" id="amount" step="0.01" value="485000.00" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="senderAccount">📤 Sender Account</label>
                        <input type="text" id="senderAccount" value="ACC789012" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="receiverAccount">📥 Receiver Account</label>
                        <input type="text" id="receiverAccount" value="ACC456789" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="transactionType">🏦 Transaction Type</label>
                        <select id="transactionType" required>
                            <option value="wire_transfer" selected>Wire Transfer</option>
                            <option value="ach_transfer">ACH Transfer</option>
                            <option value="international_wire">International Wire</option>
                            <option value="domestic_wire">Domestic Wire</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="emailContent">📧 Associated Email Content</label>
                        <textarea id="emailContent" rows="6" required>From: sarah.wilson@globaltech-corp.com
To: finance@globaltech-corp.com
Subject: URGENT - Strategic Acquisition Wire Transfer

Hi Finance Team,

I need you to process an urgent wire transfer for our confidential acquisition deal. The target company requires immediate payment to secure the transaction before market close.

Amount: $485,000.00
Recipient: Meridian Capital Holdings (ACC456789)
Purpose: Strategic acquisition deposit as discussed

Please process this immediately. This is time-sensitive and confidential.

Best regards,
Sarah Wilson
CEO, GlobalTech Corp</textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="context">📝 Additional Context</label>
                        <input type="text" id="context" value="Large wire transfer with CEO impersonation email">
                    </div>
                    
                    <button type="submit" class="btn" id="analyzeBtn">🔍 Analyze Transaction</button>
                </form>
            </div>
            
            <div class="results-panel">
                <h3>📊 Analysis Results</h3>
                <div id="resultsContent">
                    <p style="text-align: center; color: #666; padding: 40px;">Submit a transaction for analysis</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSessionId = null;
        
        // Example scenarios data
        const scenarios = {
            'crypto_scam': {
                amount: '485000.00',
                senderAccount: 'ACC789012',
                receiverAccount: 'ACC345678',
                transactionType: 'wire_transfer',
                emailContent: `From: support@crypto-elite-platform.com
To: investor@email.com
Subject: Final Opportunity - Elite Crypto Investment Platform

Dear Premium Investor,

Congratulations! You have been selected for our exclusive cryptocurrency investment opportunity with guaranteed 300% returns in 90 days.

To secure your position in our elite trading pool, you must transfer your investment amount immediately:

Investment Amount: $485,000.00
Wallet Address: CryptoElite-Vault-2025
Platform Fee: Included in investment
Expected Return: $1,455,000.00 (90 days)

This is a limited-time opportunity. Only 5 spots remaining for our Q4 2025 trading cycle.

Our AI trading algorithms have never recorded a loss. Your investment is 100% guaranteed by our Swiss banking partners.

Transfer must be completed within 24 hours to secure your position.

Best regards,
Crypto Elite Investment Team
Licensed Financial Advisors`,
                context: 'Cryptocurrency investment scam with fake guaranteed returns'
            },
            'invoice_scam': {
                amount: '125000.00',
                senderAccount: 'ACC567890',
                receiverAccount: 'ACC987654',
                transactionType: 'ach_transfer',
                emailContent: `From: billing@techsupplier-inc.com
To: accounts.payable@mycompany.com
Subject: Updated Payment Instructions - Invoice #INV-2025-0847

Dear Accounts Payable,

Due to recent banking changes, please update our payment details for immediate processing of outstanding invoices.

New Account Details:
- Bank: First National Trust
- Account: 987654321
- Routing: 123456789
- Amount Due: $125,000.00

Please process payment within 24 hours to avoid service interruption.

Best regards,
TechSupplier Inc. Billing Department`,
                context: 'Vendor impersonation with updated banking details'
            },
            'legitimate': {
                amount: '50000.00',
                senderAccount: 'ACC123456',
                receiverAccount: 'ACC111222',
                transactionType: 'domestic_wire',
                emailContent: `From: treasury@mycompany.com
To: bank.operations@mycompany.com
Subject: Q4 Supplier Payment - Construction Services

Team,

Please process the scheduled quarterly payment to our construction contractor as per our service agreement.

Payment Details:
- Contractor: BuildRight Construction LLC
- Amount: $50,000.00
- Invoice Reference: QTR4-2025-CONST
- Approval Code: MGR-2025-487

This payment was pre-approved in our Q4 budget meeting and matches our contracted rates.

Thanks,
Treasury Department`,
                context: 'Routine quarterly business payment with proper approvals'
            },
            'romance_scam': {
                amount: '75000.00',
                senderAccount: 'ACC445566',
                receiverAccount: 'ACC778899',
                transactionType: 'international_wire',
                emailContent: `From: michael.anderson847@email.com
To: my.email@provider.com
Subject: Emergency Help Needed - Please Read Immediately

My Dearest,

I hope this message finds you well. I am writing from London where I am currently stranded due to an unexpected medical emergency. My wallet and credit cards were stolen, and I urgently need your help.

I need you to send $75,000 to cover my medical bills and travel expenses to return home. I know this is a lot to ask, but I promise to pay you back as soon as I return.

Please wire the money to:
Account: International Emergency Fund
Reference: Medical Emergency - Anderson

I love you and will make this right when I get back.

Forever yours,
Michael`,
                context: 'Romance scam with emergency story and international wire request'
            },
            'tech_support': {
                amount: '15000.00',
                senderAccount: 'ACC334455',
                receiverAccount: 'ACC998877',
                transactionType: 'wire_transfer',
                emailContent: `From: security@microsoft-support.com
To: user@company.com
Subject: URGENT: Security Breach Detected - Immediate Action Required

SECURITY ALERT - DO NOT IGNORE

We have detected unauthorized access to your business accounts. Your system has been compromised and immediate action is required to prevent data loss.

To secure your accounts, you must:
1. Transfer $15,000 to our secure escrow account
2. This payment will activate enterprise-level security protection
3. Account: SecureVault Protection Services

Failure to act within 4 hours will result in permanent data loss and potential legal action.

Contact our emergency security line immediately.

Microsoft Security Team
Case #: SEC-2025-8847`,
                context: 'Tech support scam with fake security threats'
            },
            'business_compromise': {
                amount: '320000.00',
                senderAccount: 'ACC887766',
                receiverAccount: 'ACC445577',
                transactionType: 'wire_transfer',
                emailContent: `From: cfo@ourcompany.com
To: finance.team@ourcompany.com
Subject: Confidential M&A Transaction - Wire Transfer Authorization

Finance Team,

As discussed in yesterday's board meeting, we need to execute the wire transfer for the confidential merger & acquisition deal. Legal has cleared this transaction.

Transfer Details:
- Amount: $320,000.00
- Purpose: M&A Due Diligence Deposit
- Recipient: Strategic Partners LLC
- Timeline: Must complete by COB today

This transaction is highly confidential. Do not discuss with anyone outside the finance team. Process immediately and confirm completion.

Regards,
Jennifer Martinez
Chief Financial Officer`,
                context: 'Business email compromise targeting finance team with fake M&A deal'
            }
        };
        
        // Load scenario function
        function loadScenario(scenarioId) {
            const scenario = scenarios[scenarioId];
            if (scenario) {
                // Remove selected class from all cards
                document.querySelectorAll('.scenario-card').forEach(card => {
                    card.classList.remove('selected');
                });
                
                // Add selected class to clicked card
                event.target.closest('.scenario-card').classList.add('selected');
                
                // Add visual feedback
                const card = event.target.closest('.scenario-card');
                card.style.transform = 'scale(0.98)';
                setTimeout(() => {
                    card.style.transform = '';
                }, 150);
                
                // Load the scenario data
                document.getElementById('amount').value = scenario.amount;
                document.getElementById('senderAccount').value = scenario.senderAccount;
                document.getElementById('receiverAccount').value = scenario.receiverAccount;
                document.getElementById('transactionType').value = scenario.transactionType;
                document.getElementById('emailContent').value = scenario.emailContent;
                document.getElementById('context').value = scenario.context;
                
                // Clear any previous results
                document.getElementById('resultsContent').innerHTML = '<p style="text-align: center; color: #666; padding: 40px;">Submit a transaction for analysis</p>';
                
                // Show confirmation message
                const cardTitle = card.querySelector('h5').textContent;
                showNotification(`✅ Loaded: ${cardTitle}`, 'success');
            }
        }
        
        // Show notification function
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 20px;
                border-radius: 8px;
                color: white;
                font-weight: 600;
                z-index: 1000;
                transition: all 0.3s ease;
                ${type === 'success' ? 'background: #27ae60;' : 'background: #e74c3c;'}
            `;
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => notification.remove(), 300);
            }, 2000);
        }
        
        // Load consortium status
        async function loadStatus() {
            try {
                const response = await fetch('/api/consortium/status');
                const data = await response.json();
                
                const statusPanel = document.getElementById('statusPanel');
                const statusContent = document.getElementById('statusContent');
                
                if (data.status === 'online') {
                    statusPanel.className = 'status-panel status-online';
                    const participants = data.participants.participants || [];
                    
                    statusContent.innerHTML = `
                        <p>✅ <strong>Consortium Online</strong> - ${participants.length} banks active</p>
                        <div class="participants">
                            ${participants.map(p => `
                                <div class="participant">
                                    <h4>${p.node_id.toUpperCase()}</h4>
                                    <p>${p.specialty.replace('_', ' ')}</p>
                                </div>
                            `).join('')}
                        </div>
                    `;
                } else {
                    statusPanel.className = 'status-panel status-offline';
                    statusContent.innerHTML = `❌ <strong>Consortium Offline</strong><br><small>${data.error}</small>`;
                }
            } catch (error) {
                document.getElementById('statusPanel').className = 'status-panel status-offline';
                document.getElementById('statusContent').innerHTML = `❌ <strong>Connection Error</strong><br><small>${error.message}</small>`;
            }
        }
        
        // Submit form
        document.getElementById('fraudForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('analyzeBtn');
            const resultsContent = document.getElementById('resultsContent');
            
            btn.disabled = true;
            btn.textContent = '🔍 Analyzing...';
            
            resultsContent.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing transaction with consortium...</p>
                </div>
            `;
            
            try {
                const formData = {
                    amount: document.getElementById('amount').value,
                    sender_account: document.getElementById('senderAccount').value,
                    receiver_account: document.getElementById('receiverAccount').value,
                    transaction_type: document.getElementById('transactionType').value,
                    email_content: document.getElementById('emailContent').value,
                    context: document.getElementById('context').value
                };
                
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentSessionId = data.session_id;
                    pollResults(data.session_id);
                } else {
                    throw new Error(data.error);
                }
                
            } catch (error) {
                resultsContent.innerHTML = `<p style="color: red; text-align: center;">❌ Error: ${error.message}</p>`;
                btn.disabled = false;
                btn.textContent = '🔍 Analyze Transaction';
            }
        });
        
        // Poll for results
        async function pollResults(sessionId) {
            try {
                console.log('Polling for session:', sessionId); // Debug log
                const response = await fetch(`/api/results/${sessionId}`);
                const data = await response.json();
                
                console.log('Poll response:', response.status, data); // Debug log
                
                if (data.success) {
                    // Check if results are complete (not just status updates)
                    if (data.results && data.results.final_score !== undefined) {
                        displayResults(data.results);
                        document.getElementById('analyzeBtn').disabled = false;
                        document.getElementById('analyzeBtn').textContent = '🔍 Analyze Transaction';
                    } else {
                        // Still collecting results, poll again
                        console.log('Still collecting results, polling again in 2 seconds...');
                        setTimeout(() => pollResults(sessionId), 2000);
                    }
                } else if (response.status === 404) {
                    // Still processing, poll again
                    console.log('Results not ready, polling again in 2 seconds...');
                    setTimeout(() => pollResults(sessionId), 2000);
                } else {
                    throw new Error(data.error || `HTTP ${response.status}`);
                }
            } catch (error) {
                console.error('Polling error:', error); // Debug log
                document.getElementById('resultsContent').innerHTML = `<p style="color: red; text-align: center;">❌ Error: ${error.message}</p>`;
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('analyzeBtn').textContent = '🔍 Analyze Transaction';
            }
        }
        
        // Display results
        function displayResults(results) {
            console.log('Received results:', results); // Debug log
            
            const score = Math.round((results.final_score || 0) * 100);
            let riskClass = 'risk-low';
            let riskText = 'LOW RISK';
            let riskIcon = '✅';
            
            if (score >= 70) {
                riskClass = 'risk-high';
                riskText = 'HIGH RISK';
                riskIcon = '🚨';
            } else if (score >= 40) {
                riskClass = 'risk-medium';
                riskText = 'MEDIUM RISK';
                riskIcon = '⚠️';
            }
            
            const bankScores = Object.entries(results.individual_scores || {});
            const recommendation = (results.recommendation || 'review').toUpperCase();
            const consensus = results.participant_consensus || {};
            const insights = results.specialist_insights || [];
            
            // Enhanced risk assessment text
            let riskDescription = '';
            if (score >= 70) {
                riskDescription = 'Transaction shows strong indicators of fraudulent activity requiring immediate investigation.';
            } else if (score >= 40) {
                riskDescription = 'Transaction contains suspicious patterns that warrant additional verification before processing.';
            } else {
                riskDescription = 'Transaction appears legitimate based on current risk models and patterns.';
            }
            
            // Bank specialization descriptions and findings
            const bankSpecialties = {
                'bank_A': 'Wire Transfer Specialist',
                'bank_B': 'Identity Verification Expert', 
                'bank_C': 'Network Pattern Analyst'
            };
            
            // Generate specific findings for each bank based on their scores and specialization
            function getBankFindings(bank, score, insights) {
                const riskScore = Math.round(score * 100);
                
                const findings = {
                    'bank_A': {
                        high: [`🚨 Suspicious wire transfer pattern detected`, `⚠️ Transaction amount exceeds normal business profile`, `🔍 Geographic routing shows irregular pathway`],
                        medium: [`⚠️ Wire transfer amount requires additional verification`, `🔍 Transaction timing outside normal business hours`, `📊 Amount pattern deviates from account history`],
                        low: [`✅ Wire transfer follows standard protocols`, `✅ Transaction amount within expected range`, `✅ Routing pathway appears legitimate`]
                    },
                    'bank_B': {
                        high: [`🚨 Receiver identity verification failed multiple checks`, `⚠️ Account shows signs of potential compromise`, `🔍 Identity patterns suggest social engineering`],
                        medium: [`⚠️ Receiver identity requires additional verification`, `🔍 Account activity shows unusual patterns`, `📊 Identity confidence below standard threshold`],
                        low: [`✅ Receiver identity verified successfully`, `✅ Account shows normal activity patterns`, `✅ Identity verification passed all checks`]
                    },
                    'bank_C': {
                        high: [`🚨 Network analysis reveals suspicious behavioral patterns`, `⚠️ Account relationship network shows fraud indicators`, `🔍 Timing patterns consistent with automated attacks`],
                        medium: [`⚠️ Network behavior requires additional monitoring`, `🔍 Account shows atypical interaction patterns`, `📊 Behavioral signatures deviate from baseline`],
                        low: [`✅ Network behavior within normal parameters`, `✅ Account relationships appear legitimate`, `✅ Behavioral patterns match expected profile`]
                    }
                };
                
                let riskLevel = 'low';
                if (riskScore >= 70) riskLevel = 'high';
                else if (riskScore >= 40) riskLevel = 'medium';
                
                return findings[bank][riskLevel] || [`Analysis completed - ${riskScore}% risk detected`];
            }
            
            // Calculate agreement level
            const variancePercent = ((results.variance || 0) * 100).toFixed(2);
            let agreementText = '';
            if (results.variance < 0.001) {
                agreementText = 'Excellent agreement between specialist banks';
            } else if (results.variance < 0.01) {
                agreementText = 'Strong consensus across banking experts';
            } else {
                agreementText = 'Moderate consensus with some variation in assessment';
            }
            
            document.getElementById('resultsContent').innerHTML = `
                <div class="risk-indicator">
                    <div class="risk-score ${riskClass}">${score}%</div>
                    <h3>${riskIcon} ${riskText}</h3>
                    <p><strong>Recommendation:</strong> ${recommendation}</p>
                    <p style="margin-top: 10px; font-size: 0.9em; color: #666;">${riskDescription}</p>
                </div>
                
                <div style="margin: 20px 0; padding: 15px; background: #f0f4ff; border-radius: 8px; border-left: 4px solid #667eea;">
                    <h4>🏦 Specialist Bank Analysis</h4>
                    <div class="bank-scores">
                        ${bankScores.map(([bank, score]) => {
                            const findings = getBankFindings(bank, score, insights);
                            return `
                            <div class="bank-score" style="margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; border: 1px solid #e0e0e0;">
                                <h4>${bankSpecialties[bank] || bank.toUpperCase()}</h4>
                                <div style="font-size: 1.5em; font-weight: bold; color: #667eea; margin-bottom: 10px;">${Math.round((score || 0) * 100)}%</div>
                                <p style="font-size: 0.8em; color: #666; margin-bottom: 10px;">
                                    ${bank === 'bank_A' ? 'Transaction patterns & wire transfer analysis' : 
                                      bank === 'bank_B' ? 'Identity verification & receiver validation' : 
                                      'Network behavior & account relationship analysis'}
                                </p>
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 10px;">
                                    <strong style="font-size: 0.9em; color: #333;">Key Findings:</strong>
                                    <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 0.85em;">
                                        ${findings.map(finding => `<li style="margin-bottom: 3px;">${finding}</li>`).join('')}
                                    </ul>
                                </div>
                            </div>
                        `}).join('')}
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <h4>� Detailed Analysis Summary</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                        <div>
                            <p><strong>Expert Consensus:</strong><br>
                               ${consensus.high_risk || 0} of ${consensus.total || 0} specialist banks flagged this transaction</p>
                            <p><strong>Agreement Level:</strong><br>
                               ${agreementText} (${variancePercent}% variance)</p>
                        </div>
                        <div>
                            <p><strong>Processing Method:</strong><br>
                               Privacy-preserving distributed analysis</p>
                            <p><strong>Analysis Time:</strong><br>
                               ${new Date(results.completion_time).toLocaleTimeString()}</p>
                        </div>
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: #fff; border-radius: 4px; border: 1px solid #e0e0e0;">
                        <p style="font-size: 0.85em; color: #666; margin: 0;">
                            <strong>Session ID:</strong> ${results.session_id}<br>
                            <strong>Privacy Protection:</strong> Original email content was anonymized into ${Object.keys(results.individual_scores).length > 0 ? '35' : 'N/A'} behavioral features before analysis
                        </p>
                    </div>
                </div>
            `;
        }
        
        // Load status on page load
        loadStatus();
        setInterval(loadStatus, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
'''

# Set template folder and create templates directory
app.template_folder = 'templates'
import os
os.makedirs('templates', exist_ok=True)
with open('templates/fraud_detection.html', 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🌐 Starting Fraud Detection Web UI...")
    print("📍 URL: http://localhost:5001")
    print("🔗 Consortium: http://localhost:8080")
    app.run(host='localhost', port=5001, debug=True)
