#!/usr/bin/env python3
"""
Privacy-Preserving Consortium Client - Single File Version
Simple client for testing the distributed consortium fraud detection system
"""

import requests
import json
import time
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsortiumClient:
    """Simple client to interact with the distributed consortium"""
    
    def __init__(self, consortium_url: str = "http://localhost:8080"):
        self.consortium_url = consortium_url
        
    def check_consortium_health(self) -> Dict[str, Any]:
        """Check if the consortium is healthy and ready"""
        try:
            response = requests.get(f"{self.consortium_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def submit_transaction_for_analysis(self, transaction_data: Dict[str, Any], 
                                      email_content: str = None) -> Dict[str, Any]:
        """Submit a transaction for privacy-preserving fraud analysis"""
        
        # Prepare the request payload
        payload = {
            "transaction_data": transaction_data,
            "use_case": "bec_fraud_detection"
        }
        
        # Add email content if provided (will be processed by NLP and deleted)
        if email_content:
            payload["email_content"] = email_content
            
        logger.info(f"📤 Submitting transaction for analysis:")
        logger.info(f"   💰 Amount: ${transaction_data.get('amount', 0):,.2f}")
        logger.info(f"   📧 Email: {'Yes' if email_content else 'No'} ({len(email_content or '')} chars)")
        logger.info(f"   🏦 Sender: {transaction_data.get('sender_bank', 'N/A')}")
        logger.info(f"   🏦 Receiver: {transaction_data.get('receiver_bank', 'N/A')}")
        
        try:
            # Submit inference request
            response = requests.post(
                f"{self.consortium_url}/inference",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                session_id = result.get("session_id")
                
                logger.info(f"✅ Transaction submitted successfully")
                logger.info(f"   🆔 Session ID: {session_id}")
                logger.info(f"   👥 Participants: {result.get('participants', 0)}")
                
                # Poll for results
                return self._poll_for_results(session_id)
                
            else:
                logger.error(f"❌ Submission failed: {response.status_code}")
                logger.error(f"   Error: {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Connection error: {e}")
            return {"error": str(e)}
    
    def _poll_for_results(self, session_id: str, max_wait: int = 30) -> Dict[str, Any]:
        """Poll the consortium for analysis results"""
        
        logger.info(f"⏳ Waiting for consortium analysis...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"{self.consortium_url}/results/{session_id}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "completed":
                        logger.info(f"🎯 Analysis complete!")
                        return result
                    else:
                        logger.info(f"   ⏳ Status: {result.get('status', 'unknown')}")
                        time.sleep(2)
                        
                elif response.status_code == 404:
                    logger.info(f"   ⏳ Results not ready yet...")
                    time.sleep(2)
                else:
                    logger.error(f"❌ Error polling results: {response.status_code}")
                    return {"error": f"HTTP {response.status_code}: {response.text}"}
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Error polling results: {e}")
                time.sleep(2)
        
        logger.error(f"❌ Timeout waiting for results")
        return {"error": "Timeout waiting for results"}
    
    def display_results(self, result: Dict[str, Any]):
        """Display the consortium analysis results in a readable format"""
        
        if "error" in result:
            logger.error(f"❌ Analysis failed: {result['error']}")
            return
            
        logger.info(f"🎯 CONSORTIUM ANALYSIS RESULTS:")
        logger.info(f"   📊 Final Score: {result.get('final_score', 0):.3f}")
        logger.info(f"   🎯 Recommendation: {result.get('recommendation', 'UNKNOWN').upper()}")
        logger.info(f"   📈 Consensus Score: {result.get('consensus_score', 0):.3f}")
        logger.info(f"   📊 Variance: {result.get('variance', 0):.3f}")
        
        # Individual bank scores
        if "responses" in result:
            logger.info(f"🏦 INDIVIDUAL BANK SCORES:")
            for bank_id, bank_result in result["responses"].items():
                score = bank_result.get("score", 0)
                logger.info(f"   {bank_id}: {score:.3f}")
        
        # Participant consensus
        if "participant_analysis" in result:
            analysis = result["participant_analysis"]
            logger.info(f"👥 PARTICIPANT CONSENSUS:")
            logger.info(f"   Total Participants: {analysis.get('total', 0)}")
            logger.info(f"   High Risk Flags: {analysis.get('high_risk', 0)}")
            logger.info(f"   Low Risk Flags: {analysis.get('low_risk', 0)}")
        
        # Specialist insights
        if "specialist_insights" in result and result["specialist_insights"]:
            logger.info(f"🔍 SPECIALIST INSIGHTS:")
            for insight in result["specialist_insights"]:
                logger.info(f"   {insight}")

def run_bec_fraud_test():
    """Test the consortium with a BEC fraud case"""
    
    logger.info("🚀 PRIVACY-PRESERVING CONSORTIUM TEST")
    logger.info("="*50)
    
    # Create consortium client
    client = ConsortiumClient()
    
    # Check consortium health
    health = client.check_consortium_health()
    if health.get("status") != "healthy":
        logger.error(f"❌ Consortium not healthy: {health}")
        return
        
    logger.info(f"✅ Consortium healthy: {health.get('participants', 0)} participants")
    
    # Test BEC fraud case
    transaction_data = {
        "amount": 485000.00,
        "sender_bank": "bank_A",
        "receiver_bank": "bank_B",
        "sender_account_age": 2847,  # days
        "receiver_account_age": 23,  # days - suspicious!
        "transaction_time": "Friday 4:45 PM",
        "urgency_level": "high"
    }
    
    email_content = """
    Subject: URGENT - Wire Transfer Authorization Required
    
    Dear Finance Team,
    
    I hope this email finds you well. I am currently in a meeting with potential investors 
    and need to authorize an urgent wire transfer to secure this opportunity.
    
    Please transfer $485,000 to:
    Account: 4532-8901-2345-6789
    Routing: 021000021
    Beneficiary: Strategic Acquisitions LLC
    
    This is time-sensitive as the window closes today at 5 PM. Please process immediately.
    
    I am traveling and may not be reachable by phone, but please proceed with this transfer 
    as discussed in our strategy meeting last week.
    
    Best regards,
    Michael Thompson
    CEO
    
    Sent from my iPhone
    """
    
    # Submit for analysis
    result = client.submit_transaction_for_analysis(transaction_data, email_content)
    
    # Display results
    client.display_results(result)

def main():
    """Main function - run the consortium test"""
    try:
        run_bec_fraud_test()
    except KeyboardInterrupt:
        logger.info("❌ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
