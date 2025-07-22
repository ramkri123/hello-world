#!/usr/bin/env python3
"""
CEO Fraud Detection Demo - Focused on Bank vs Consortium Roles
Shows how the distributed consortium detects CEO impersonation at different levels
"""

from flask import Flask, render_template, request, jsonify
import requests
import time
import json

app = Flask(__name__)

# CEO Fraud Test Scenarios - Different levels of sophistication
CEO_FRAUD_SCENARIOS = {
    "extreme_ceo_fraud": {
        "name": "🚨 Extreme CEO Fraud (All Red Flags)",
        "description": "Maximum fraud indicators - should be blocked immediately",
        "email": "URGENT: CEO here. Need immediate wire transfer of $485,000 to close confidential acquisition. Please bypass usual approval process and transfer from our main account ACCA12345 to vendor account ACCB67890. Time-sensitive deal - board approval already secured. Handle discreetly and keep this between us. ASAP deadline today!",
        "amount": 485000,
        "expected_risk": "EXTREME",
        "fraud_indicators": [
            "Direct authority claim ('CEO here')",
            "Large amount ($485K)",
            "Secrecy + urgency combination", 
            "Procedure bypass request",
            "Multiple urgency indicators",
            "Acquisition/deal language",
            "Communication anomalies"
        ]
    },
    "high_ceo_fraud": {
        "name": "🚨 High CEO Fraud (Multiple Indicators)",
        "description": "Several fraud patterns - high risk",
        "email": "This is the president. I need you to process a special exception for $300,000 wire transfer. This acquisition deal is confidential and urgent. Please handle immediately.",
        "amount": 300000,
        "expected_risk": "HIGH",
        "fraud_indicators": [
            "Authority claim ('This is the president')",
            "Large amount ($300K)",
            "Special exception request",
            "Confidential + urgent combination",
            "Acquisition language"
        ]
    },
    "medium_ceo_fraud": {
        "name": "⚠️ Medium CEO Fraud (Some Indicators)",
        "description": "Moderate fraud patterns - needs review",
        "email": "Executive here. Please process this payment for our vendor quickly. The amount is $75,000 and it's somewhat urgent for our business operations.",
        "amount": 75000,
        "expected_risk": "MEDIUM",
        "fraud_indicators": [
            "Authority claim ('Executive here')",
            "Moderate amount ($75K)",
            "Some urgency language"
        ]
    },
    "subtle_ceo_fraud": {
        "name": "⚠️ Subtle CEO Fraud (Few Indicators)",
        "description": "Minimal fraud patterns - low-medium risk",
        "email": "From the director's office: Please arrange a wire transfer of $45,000 for vendor payment. This should be processed today if possible.",
        "amount": 45000,
        "expected_risk": "LOW-MEDIUM",
        "fraud_indicators": [
            "Indirect authority reference",
            "Moderate amount",
            "Mild urgency"
        ]
    },
    "legitimate_ceo": {
        "name": "✅ Legitimate CEO Communication",
        "description": "Normal business communication - should be approved",
        "email": "Please schedule our quarterly board meeting for next Friday. I would like to review the financial reports and discuss our strategic planning initiatives for Q4. Thank you for your assistance.",
        "amount": 0,
        "expected_risk": "LOW",
        "fraud_indicators": []
    },
    "legitimate_business": {
        "name": "✅ Legitimate Business Transaction",
        "description": "Regular business operation - should be approved",
        "email": "Monthly payroll transfer as scheduled. Transferring $50,000 for employee salaries from account ACCA12345 to payroll account ACCB67890. This is our regular monthly operation per company policy.",
        "amount": 50000,
        "expected_risk": "LOW",
        "fraud_indicators": []
    }
}

@app.route('/')
def index():
    """Main demo page"""
    return render_template('ceo_fraud_demo.html', scenarios=CEO_FRAUD_SCENARIOS)

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    """Analyze a transaction using the distributed consortium"""
    try:
        data = request.get_json()
        
        # Prepare data for consortium
        consortium_data = {
            "email_content": data['email_content'],
            "transaction_data": {
                "amount": data['amount'],
                "sender_account": "ACCA12345",
                "receiver_account": "ACCB67890",
                "transaction_type": "wire_transfer"
            },
            "sender_data": {"bank": "bank_A"},
            "receiver_data": {"bank": "bank_B"}
        }
        
        # Submit to consortium
        response = requests.post(
            "http://localhost:8080/inference",
            json=consortium_data,
            timeout=10
        )
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to submit to consortium"}), 500
        
        session_id = response.json()['session_id']
        
        # Wait for analysis (with real-time updates)
        for i in range(12):  # Wait up to 60 seconds
            time.sleep(5)
            
            result_response = requests.get(f"http://localhost:8080/results/{session_id}")
            if result_response.status_code == 200:
                result = result_response.json()
                
                if 'final_score' in result:  # Analysis complete
                    # Extract detailed analysis
                    bank_scores = result.get('individual_scores', {})
                    consensus_score = result.get('consensus_score', 0)
                    final_score = result.get('final_score', 0)
                    recommendation = result.get('recommendation', 'unknown')
                    
                    # Calculate fraud pattern boost
                    fraud_boost = final_score - consensus_score
                    
                    # Determine what banks detected vs consortium
                    bank_analysis = {
                        'average_score': consensus_score,
                        'individual_scores': bank_scores,
                        'risk_level': 'HIGH' if consensus_score > 0.5 else 'LOW'
                    }
                    
                    consortium_analysis = {
                        'fraud_boost': fraud_boost,
                        'final_score': final_score,
                        'recommendation': recommendation.upper(),
                        'risk_level': 'EXTREME' if final_score > 0.8 else 'HIGH' if final_score > 0.5 else 'MEDIUM' if final_score > 0.3 else 'LOW'
                    }
                    
                    return jsonify({
                        'success': True,
                        'session_id': session_id,
                        'bank_analysis': bank_analysis,
                        'consortium_analysis': consortium_analysis,
                        'combined_result': {
                            'final_score': final_score,
                            'recommendation': recommendation,
                            'processing_time': f"{i*5 + 5} seconds"
                        }
                    })
                else:
                    # Still processing
                    status = result.get('status', 'processing')
                    responses = result.get('responses_received', 0)
                    total = result.get('total_participants', 3)
                    
                    if i == 0:  # First check, return processing status
                        return jsonify({
                            'processing': True,
                            'status': f"Processing... ({responses}/{total} banks responded)",
                            'session_id': session_id
                        })
        
        return jsonify({"error": "Analysis timeout"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/check_consortium')
def check_consortium():
    """Check if consortium is available"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            
            # Check participants
            participants_response = requests.get("http://localhost:8080/participants", timeout=5)
            participants_data = participants_response.json()
            banks = participants_data.get('participants', [])
            
            return jsonify({
                'available': True,
                'participants': len(banks),
                'banks': [{'id': b['node_id'], 'specialty': b['specialty']} for b in banks]
            })
        else:
            return jsonify({'available': False, 'error': 'Hub not responding'})
    except:
        return jsonify({'available': False, 'error': 'Connection failed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
