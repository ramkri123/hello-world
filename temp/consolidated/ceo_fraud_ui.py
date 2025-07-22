#!/usr/bin/env python3
"""
CEO Fraud Detection Flask UI
Focused demonstration of Bank vs Consortium fraud detection capabilities
"""

from flask import Flask, render_template, request, jsonify
import requests
import json
import logging

app = Flask(__name__)
app.config['DEBUG'] = True

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONSORTIUM_HUB_URL = "http://localhost:8080"

@app.route('/')
def index():
    """Main dashboard page - CEO Fraud Focus"""
    return render_template('ceo_fraud_focus.html')

@app.route('/full_demo')
def full_demo():
    """Full 13-scenario demo page"""
    return render_template('index.html')

@app.route('/analyze_transaction', methods=['POST'])
def analyze_transaction():
    """Analyze a transaction using the consortium system"""
    try:
        # Get the transaction data from the request
        transaction_data = request.get_json()
        
        logger.info(f"Analyzing transaction: {json.dumps(transaction_data, indent=2)}")
        
        # Send to consortium hub for analysis
        response = requests.post(
            f"{CONSORTIUM_HUB_URL}/analyze",
            json=transaction_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Analysis result: {json.dumps(result, indent=2)}")
            return jsonify(result)
        else:
            logger.error(f"Consortium hub error: {response.status_code} - {response.text}")
            return jsonify({
                'error': f'Consortium hub returned status {response.status_code}',
                'details': response.text
            }), 500
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error to consortium hub: {str(e)}")
        return jsonify({
            'error': 'Could not connect to consortium hub',
            'details': str(e),
            'suggestion': 'Make sure the consortium hub is running on port 8080'
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_transaction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check if consortium hub is available
        response = requests.get(f"{CONSORTIUM_HUB_URL}/health", timeout=5)
        hub_status = response.status_code == 200
    except:
        hub_status = False
    
    return jsonify({
        'ui_status': 'healthy',
        'consortium_hub_status': 'connected' if hub_status else 'disconnected',
        'consortium_hub_url': CONSORTIUM_HUB_URL
    })

@app.route('/api/scenarios')
def get_scenarios():
    """Get available test scenarios"""
    scenarios = {
        'high_fraud': {
            'name': 'High Sophistication CEO Fraud',
            'description': 'Sophisticated impersonator mimics CEO style with urgent wire transfer',
            'risk_level': 'high',
            'expected_outcome': 'blocked'
        },
        'medium_fraud': {
            'name': 'Medium Sophistication CEO Fraud',
            'description': 'Basic CEO impersonation with authority claims',
            'risk_level': 'medium',
            'expected_outcome': 'review'
        },
        'low_fraud': {
            'name': 'Low Sophistication CEO Fraud',
            'description': 'Obvious impersonation attempt with poor execution',
            'risk_level': 'low',
            'expected_outcome': 'flagged'
        },
        'legitimate_urgent': {
            'name': 'Legitimate CEO (Urgent)',
            'description': 'Real CEO making urgent but legitimate business request',
            'risk_level': 'legitimate',
            'expected_outcome': 'approved'
        },
        'legitimate_routine': {
            'name': 'Legitimate CEO (Routine)',
            'description': 'Real CEO making routine business communication',
            'risk_level': 'legitimate',
            'expected_outcome': 'approved'
        }
    }
    return jsonify(scenarios)

if __name__ == '__main__':
    print("🎭 CEO Fraud Detection UI Starting...")
    print("📊 Focused Demo: CEO Fraud of Different Levels vs Legitimate CEO")
    print("🏦 Demonstrating: Role of Bank vs Role of Consortium")
    print("🌐 Access the demo at: http://localhost:5000")
    print("🔗 Full demo available at: http://localhost:5000/full_demo")
    print()
    print("💡 This demo shows how:")
    print("   🏦 Banks detect technical patterns through ML")
    print("   🤝 Consortium recognizes behavioral manipulation")
    print("   🛡️ Together they provide superior fraud protection")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
