#!/usr/bin/env python3
"""
Consortium Hub - Central aggregator/comparator service
Runs as separate HTTP server process
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import uuid
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import sys
import os

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARN)

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from privacy_preserving_nlp import PrivacyPreservingNLP
from account_anonymizer import AccountAnonymizer

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ParticipantInfo:
    node_id: str
    specialty: str
    endpoint: str
    geolocation: str
    registered_at: datetime
    last_heartbeat: datetime
    status: str = "active"

@dataclass
class InferenceSession:
    session_id: str
    use_case: str
    features: List[float]
    raw_data: Dict[str, Any]  # Store original data for fraud pattern analysis
    created_at: datetime
    deadline: datetime
    responses: Dict[str, Any]
    anonymized_accounts: Dict[str, str] = None  # Store anonymized account identifiers
    status: str = "pending"  # pending, collecting, completed, timeout

class ConsortiumHub:
    def __init__(self, port=8080):
        self.app = Flask(__name__)
        CORS(self.app)
        self.port = port
        
        # CRITICAL: Secret salt known ONLY to consortium hub
        # Banks cannot reproduce these hashes without knowing this secret
        import secrets
        self._secret_salt = secrets.token_hex(32)  # 64-character secret
        
        # State management
        self.participants: Dict[str, ParticipantInfo] = {}  # Legacy system (kept for compatibility)
        self.banks: Dict[str, ParticipantInfo] = {}  # New secure bank registration system
        self.active_sessions: Dict[str, InferenceSession] = {}
        self.session_results: Dict[str, Dict] = {}
        self.pending_inferences: Dict[str, Dict] = {}  # Queue for bank polling
        
        # Configuration
        self.inference_timeout = 30  # seconds
        self.min_participants = 2
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            # Log the source of the health check request
            client_ip = request.remote_addr
            user_agent = request.headers.get('User-Agent', 'Unknown')
            
            # Try to identify the source based on user agent or other headers
            source = "Unknown"
            if 'python-requests' in user_agent.lower():
                if 'streamlit' in request.headers.get('Referer', '').lower():
                    source = "Streamlit UI"
                else:
                    source = "Bank Process"
            elif 'powershell' in user_agent.lower() or 'invoke-webrequest' in user_agent.lower():
                source = "PowerShell/Manual"
            elif 'mozilla' in user_agent.lower() or 'chrome' in user_agent.lower():
                source = "Web Browser"
            
            logger.info(f"🔍 Health check from {source} ({client_ip}) - UA: {user_agent[:50]}...")
            
            return jsonify({
                "status": "healthy",
                "participants": len(self.participants),
                "active_sessions": len(self.active_sessions),
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/register', methods=['POST'])
        def register_participant():
            try:
                data = request.get_json()
                node_id = data['node_id']
                specialty = data['specialty']
                endpoint = data['endpoint']
                geolocation = data.get('geolocation', 'unknown')
                
                # Validate registration
                if node_id in self.participants:
                    return jsonify({"error": "Participant already registered"}), 400
                
                # Create participant info
                participant = ParticipantInfo(
                    node_id=node_id,
                    specialty=specialty,
                    endpoint=endpoint,
                    geolocation=geolocation,
                    registered_at=datetime.now(),
                    last_heartbeat=datetime.now()
                )
                
                self.participants[node_id] = participant
                
                logger.info(f"✅ Registered participant: {node_id} ({specialty})")
                
                return jsonify({
                    "status": "registered",
                    "node_id": node_id,
                    "session_token": f"token_{node_id}_{int(time.time())}"
                })
                
            except Exception as e:
                logger.error(f"Registration error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/participants', methods=['GET'])
        def list_participants():
            return jsonify({
                "participants": [
                    {
                        "node_id": p.node_id,
                        "specialty": p.specialty,
                        "status": p.status,
                        "registered_at": p.registered_at.isoformat()
                    }
                    for p in self.participants.values()
                ]
            })
        
        @self.app.route('/inference', methods=['POST'])
        def submit_inference():
            try:
                data = request.get_json()
                
                # Check if we have raw transaction data that needs NLP processing
                if 'email_content' in data or 'transaction_data' in data:
                    # Privacy-preserving NLP feature extraction
                    nlp = PrivacyPreservingNLP()
                    
                    transaction_data = data.get('transaction_data', {})
                    email_content = data.get('email_content', '')
                    sender_data = data.get('sender_data', None)
                    receiver_data = data.get('receiver_data', None)
                    
                    logger.info(f"🔄 PROCESSING RAW TRANSACTION DATA:")
                    logger.info(f"   📧 Email Content: {len(email_content)} characters")
                    logger.info(f"   💰 Transaction: ${transaction_data.get('amount', 0):,.2f}")
                    logger.info(f"   🏦 Converting to anonymous features...")
                    
                    # Create anonymized account identifiers for banks
                    sender_account = transaction_data.get('sender_account', '')
                    receiver_account = transaction_data.get('receiver_account', '')
                    
                    # Generate anonymized identifiers that preserve bank ownership recognition
                    anonymized_accounts = self._create_anonymized_account_identifiers(
                        sender_account, receiver_account
                    )

                    # ramki
                    print("\ndata start\n")
                    print(data)
                    print(anonymized_accounts)
                    print("\ndata end\n")
                    data['transaction_data']['email_content'] = ''

                    data['transaction_data']['sender_data'] = ''
                    data['transaction_data']['sender_account'] = ''
                    data['transaction_data']['sender_anonymous'] = anonymized_accounts['sender_anonymous']

                    data['transaction_data']['receiver_data'] = ''
                    data['transaction_data']['receiver_account'] = ''
                    data['transaction_data']['receiver_anonymous'] = anonymized_accounts['receiver_anonymous']
                    
                    # Extract anonymous features using NLP
                    features = nlp.convert_to_anonymous_features(
                        transaction_data, email_content, sender_data, receiver_data
                    )
                    
                    logger.info(f"   ✅ Generated {len(features)} anonymous features")
                    logger.info(f"   🔒 EMAIL CONTENT DELETED (privacy preserved)")
                    logger.info(f"   🔑 Account identifiers anonymized")
                    
                else:
                    # Traditional features array
                    features = data['features']
                
                use_case = data.get('use_case', 'fraud_detection')
                
                logger.info(f"🔄 NEW INFERENCE REQUEST:")
                logger.info(f"   📊 Use Case: {use_case}")
                logger.info(f"   📈 Features: {len(features)} values - First 10: {[f'{f:.3f}' for f in features[:10]]}")
                logger.info(f"   🌐 Source IP: {request.remote_addr}")
                
                # Check if enough banks available for secure inference
                active_banks = [b for b in self.banks.values() if b.status in ["active", "online"]]
                logger.info(f"🔍 BANK AVAILABILITY CHECK:")
                logger.info(f"   📊 Total banks: {len(self.banks)}")
                logger.info(f"   ✅ Active banks: {len(active_banks)}")
                logger.info(f"   🏦 Bank IDs: {list(self.banks.keys())}")
                
                if len(active_banks) < self.min_participants:
                    logger.error(f"❌ Not enough banks: Need {self.min_participants}, have {len(active_banks)}")
                    return jsonify({
                        "error": f"Not enough banks. Need {self.min_participants}, have {len(active_banks)}"
                    }), 400
                
                # Create session
                session_id = str(uuid.uuid4())
                session = InferenceSession(
                    session_id=session_id,
                    use_case=use_case,
                    features=features,
                    raw_data=data,  # Store original data for fraud pattern analysis
                    anonymized_accounts=anonymized_accounts if 'anonymized_accounts' in locals() else None,
                    created_at=datetime.now(),
                    deadline=datetime.now() + timedelta(seconds=self.inference_timeout),
                    responses={}
                )
                
                self.active_sessions[session_id] = session
                
                logger.info(f"🎯 CREATED SESSION: {session_id}")
                logger.info(f"   ⏰ Deadline: {session.deadline.strftime('%H:%M:%S')}")
                logger.info(f"   🏦 Will distribute to: {[b.node_id for b in active_banks]}")
                
                # Start inference distribution in background
                threading.Thread(
                    target=self._distribute_inference,
                    args=(session_id,),
                    daemon=True
                ).start()
                
                logger.info(f"🚀 Started inference session: {session_id}")
                
                return jsonify({
                    "session_id": session_id,
                    "status": "submitted",
                    "participants": len(active_banks),
                    "estimated_completion": (datetime.now() + timedelta(seconds=self.inference_timeout)).isoformat()
                })
                
            except Exception as e:
                logger.error(f"Inference submission error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/score', methods=['POST'])
        def receive_score():
            try:
                data = request.get_json()
                session_id = data['session_id']
                participant_id = data['participant_id']
                risk_score = data['risk_score']
                confidence = data.get('confidence', 1.0)
                
                # Validate session
                if session_id not in self.active_sessions:
                    return jsonify({"error": "Invalid session ID"}), 400
                
                session = self.active_sessions[session_id]
                if session.status != "collecting":
                    return jsonify({"error": "Session not accepting scores"}), 400
                
                # Store response
                session.responses[participant_id] = {
                    "risk_score": risk_score,
                    "confidence": confidence,
                    "received_at": datetime.now().isoformat()
                }
                
                logger.info(f"📊 Received score from {participant_id}: {risk_score:.3f}")
                
                # Check if all responses collected
                active_participants = [p.node_id for p in self.participants.values() if p.status == "active"]
                if len(session.responses) >= len(active_participants):
                    self._complete_inference(session_id)
                
                return jsonify({"status": "received"})
                
            except Exception as e:
                logger.error(f"Score reception error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/results/<session_id>', methods=['GET'])
        def get_results(session_id):
            try:
                if session_id in self.session_results:
                    return jsonify(self.session_results[session_id])
                elif session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    return jsonify({
                        "session_id": session_id,
                        "status": session.status,
                        "responses_received": len(session.responses),
                        "total_participants": len([p for p in self.participants.values() if p.status == "active"]),
                        "time_remaining": max(0, (session.deadline - datetime.now()).total_seconds())
                    })
                else:
                    return jsonify({"error": "Session not found"}), 404
                    
            except Exception as e:
                logger.error(f"Results retrieval error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/poll_inference', methods=['GET'])
        def poll_inference():
            """Endpoint for banks to poll for pending inference requests"""
            try:
                participant_id = request.args.get('participant_id')
                if not participant_id:
                    return jsonify({"error": "participant_id required"}), 400
                
                # Check if participant is registered
                if participant_id not in self.participants:
                    return jsonify({"error": "Participant not registered"}), 401
                
                # Find pending inference for this participant
                for session_id, inference_data in list(self.pending_inferences.items()):
                    if participant_id in inference_data.get('pending_participants', []):
                        # Remove this participant from pending list
                        inference_data['pending_participants'].remove(participant_id)
                        
                        # If no more pending participants, remove from queue
                        if not inference_data['pending_participants']:
                            del self.pending_inferences[session_id]
                        
                        logger.info(f"📤 Sending inference request to {participant_id} for session {session_id}")
                        
                        return jsonify({
                            "session_id": session_id,
                            "features": inference_data['features'],
                            "anonymized_accounts": inference_data.get('anonymized_accounts', {}),
                            "use_case": inference_data['use_case'],
                            "deadline": inference_data['deadline']
                        })
                
                # No pending inference for this participant
                return jsonify({"message": "No pending inference"}), 204
                
            except Exception as e:
                logger.error(f"Poll inference error: {e}")
                return jsonify({"error": str(e)}), 500
        
        # New endpoints for outbound-only bank architecture
        @self.app.route('/register_bank', methods=['POST'])
        def register_bank():
            """Register a bank with outbound-only communication"""
            try:
                data = request.get_json()
                bank_id = data['bank_id']
                specialty = data['specialty']
                status = data.get('status', 'online')
                
                # Store bank information in the new banks registry
                self.banks[bank_id] = ParticipantInfo(
                    node_id=bank_id,
                    specialty=specialty,
                    endpoint=f"outbound-only",  # No inbound endpoint
                    geolocation="secure",
                    registered_at=datetime.now(),
                    last_heartbeat=datetime.now(),
                    status=status
                )
                
                logger.info(f"🏦 Bank {bank_id} registered ({specialty}) - Outbound only")
                
                return jsonify({
                    "status": "registered",
                    "bank_id": bank_id,
                    "message": "Bank registered successfully"
                })
                
            except Exception as e:
                logger.error(f"Bank registration error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/get_analysis_request/<bank_id>', methods=['GET'])
        def get_analysis_request(bank_id):
            """Poll endpoint for banks to get analysis requests"""
            try:
                # Check if bank is registered
                if bank_id not in self.banks:
                    return jsonify({"error": "Bank not registered"}), 401
                
                # Update heartbeat
                self.banks[bank_id].last_heartbeat = datetime.now()
                
                # Check for pending analysis requests for this bank
                for request_id, request_data in list(self.pending_inferences.items()):
                    if bank_id in request_data.get('pending_participants', []):
                        # Remove this bank from pending list
                        request_data['pending_participants'].remove(bank_id)
                        
                        return jsonify({
                            "has_request": True,
                            "request_id": request_id,
                            "transaction_data": request_data['raw_data'],
                            "deadline": request_data['deadline']
                        })
                
                # No requests pending
                return jsonify({"has_request": False})
                
            except Exception as e:
                logger.error(f"Get analysis request error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/submit_analysis', methods=['POST'])
        def submit_analysis():
            """Receive analysis results from banks"""
            try:
                data = request.get_json()
                request_id = data['request_id']
                analysis = data['analysis']
                
                # Find the active session
                if request_id in self.active_sessions:
                    session = self.active_sessions[request_id]
                    bank_id = analysis['bank_id']
                    
                    # Store the response
                    session.responses[bank_id] = analysis
                    
                    logger.info(f"📊 Received analysis from {bank_id} for {request_id}: {analysis['fraud_score']:.3f}")
                    
                    # Check if we have all responses
                    if len(session.responses) >= self.min_participants:
                        # Process the complete analysis
                        threading.Thread(
                            target=self._complete_inference,
                            args=(request_id,),
                            daemon=True
                        ).start()
                    
                    return jsonify({"status": "received"})
                else:
                    return jsonify({"error": "Request not found"}), 404
                
            except Exception as e:
                logger.error(f"Submit analysis error: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _create_anonymized_account_identifiers(self, sender_account: str, receiver_account: str) -> Dict[str, str]:
        """Create anonymized account identifiers using agreed-upon consortium hash function"""
        
        
        # Use consortium-standard one-way hash function that all banks can also use
        return AccountAnonymizer.anonymize_transaction_accounts(sender_account, receiver_account)
    
    # ramki
    def _detect_ceo_impersonation(self, email_content: str, amount: float) -> float:
        """
        Advanced CEO impersonation detection using multiple behavioral indicators
        Returns confidence score (0.0 to 0.35) based on suspicious patterns
        """
        confidence_score = 0.0
        
        # 1. Authority Claims (Direct impersonation)
        authority_phrases = [
            'ceo here', 'this is the ceo', 'from the ceo', 'ceo speaking',
            'i am the ceo', 'i am ceo', 'this is ceo', 'ceo so do it',
            'president here', 'i am the president', 'this is your president',
            'executive here', 'senior executive', 'c-suite', 'board member',
            'founder here', 'owner here', 'director here'
        ]
        if any(phrase in email_content for phrase in authority_phrases):
            confidence_score += 0.05  # Strong indicator
            
        # 2. Authority + Secrecy Combination (Classic CEO fraud)
        authority_words = ['ceo', 'president', 'executive', 'director', 'founder']
        secrecy_words = ['confidential', 'discreet', 'private', 'secret', 'classified', 'need to know']
        
        has_authority = any(word in email_content for word in authority_words)
        has_secrecy = any(word in email_content for word in secrecy_words)
        
        if has_authority and has_secrecy:
            confidence_score += 0.05
            
        # 3. Bypass Normal Procedures (Red flag for impersonation)
        bypass_phrases = [
            'bypass', 'skip the usual', 'exception', 'special case', 'different process',
            'directly to', 'not through normal', 'outside normal', 'urgent exception'
        ]
        if any(phrase in email_content for phrase in bypass_phrases):
            confidence_score += 0.05
            
        # 4. Time Pressure + Authority (Manipulation tactic)
        urgency_phrases = ['urgent', 'immediate', 'asap', 'right now', 'quickly', 'deadline']
        urgency_count = sum(1 for phrase in urgency_phrases if phrase in email_content)
        
        if has_authority and urgency_count >= 2:
            confidence_score += 0.05
            
        # 5. Acquisition/Deal Language (Common CEO fraud scenario)
        deal_phrases = [
            'acquisition', 'merger', 'deal', 'purchase', 'investment opportunity',
            'close the deal', 'business opportunity', 'vendor payment', 'contract'
        ]
        if any(phrase in email_content for phrase in deal_phrases) and amount > 100000:
            confidence_score += 0.05
            
        # 6. Communication Anomalies (Behavioral red flags)
        anomaly_phrases = [
            'handle discreetly', 'between us', 'don\'t tell', 'keep quiet',
            'off the books', 'cash transaction', 'wire transfer only'
        ]
        if any(phrase in email_content for phrase in anomaly_phrases):
            confidence_score += 0.05
            
        # 7. Grammar/Style Inconsistencies (Often indicates impersonation)
        # Look for overly formal language mixed with urgency (suspicious pattern)
        formal_words = ['pursuant', 'hereby', 'aforementioned', 'heretofore']
        informal_urgent = ['asap', 'quick', 'hurry', 'fast']
        
        has_formal = any(word in email_content for word in formal_words)
        has_informal_urgent = any(word in email_content for word in informal_urgent)
        
        if has_formal and has_informal_urgent:
            confidence_score += 0.05
            
        # 8. External Pressure References (Social engineering)
        pressure_phrases = [
            'board approval', 'investor pressure', 'deadline from', 'client demands',
            'regulatory requirement', 'audit requirement', 'compliance issue'
        ]
        if any(phrase in email_content for phrase in pressure_phrases):
            confidence_score += 0.05
            
        # Cap the maximum confidence score
        return min(confidence_score, 0.15)
    
    def _calculate_scenario_weights(self, individual_scores: Dict[str, float], sender_account: str, receiver_account: str) -> Dict[str, float]:
        """Calculate confidence weights - banks determine their own scenarios using shared hash function"""
        
        
        # Banks determine their own knowledge scenarios by hashing their accounts
        # and comparing to the anonymized identifiers they received
        # We use default weights here since banks handle scenario determination themselves
        
        weights = {}
        for participant_id in individual_scores.keys():
            # Default weight - in practice, banks calculate their own scenario weights
            # and include confidence in their risk score submissions
            weights[participant_id] = 1.0
            
            logger.info(f"   🎯 {participant_id}: Using bank-determined scenario weight")
        
        return weights
    
    def _apply_fraud_pattern_boost(self, consensus_score: float, session) -> float:
        """Apply fraud pattern boost based on obvious fraud indicators"""
        try:
            print(f"🚨 DEBUG: FRAUD PATTERN BOOST CALLED")  # Force console output
            logger.info(f"🚨 DEBUG: FRAUD PATTERN BOOST CALLED")
            
            # Get original email content for analysis - try multiple sources
            email_content = session.raw_data.get('email_content', '').lower()
            transaction_data = session.raw_data.get('transaction_data', {})
            
            # If email_content is empty (deleted for privacy), try communication field
            if not email_content:
                email_content = transaction_data.get('communication', '').lower()
            
            amount = transaction_data.get('amount', 0)
            
            print(f"🔍 FRAUD PATTERN ANALYSIS - Content: '{email_content}'")  # Force console output
            logger.info(f"   🔍 FRAUD PATTERN ANALYSIS:")
            logger.info(f"      Email content length: {len(email_content)} chars")
            logger.info(f"      Amount: ${amount:,.2f}")
            logger.info(f"      Content preview: '{email_content[:50]}...'")
            
            boost_factor = 0.0
            fraud_indicators = []
            
            # Enhanced CEO/Executive Impersonation Detection
            ceo_impersonation_score = self._detect_ceo_impersonation(email_content, amount)
            if ceo_impersonation_score > 0:
                boost_factor += ceo_impersonation_score
                fraud_indicators.append(f"CEO impersonation (confidence: {ceo_impersonation_score:.2f})")
            
            # ramki
            # High-value transaction boost
            if amount > 100000:
                boost_factor += 0.05
                fraud_indicators.append(f"Large amount: ${amount:,.2f}")
            
            # Crypto/investment scam patterns
            if any(phrase in email_content for phrase in ['crypto', 'investment', 'return', 'guarantee', 'opportunity']):
                boost_factor += 0.05
                fraud_indicators.append("Investment scam pattern")
            
            # Romance/dating scam patterns  
            if any(phrase in email_content for phrase in ['love', 'relationship', 'meet', 'dating', 'emergency']):
                boost_factor += 0.05
                fraud_indicators.append("Romance scam pattern")
            
            # Business email compromise patterns
            if any(phrase in email_content for phrase in ['wire transfer', 'vendor', 'payment', 'invoice', 'urgent']):
                if amount > 50000:
                    boost_factor += 0.05
                    fraud_indicators.append("BEC pattern with large amount")
            
            # Multiple urgency/pressure indicators
            urgency_count = sum(1 for phrase in ['urgent', 'immediate', 'asap', 'quickly', 'deadline'] if phrase in email_content)
            if urgency_count >= 2:
                boost_factor += 0.05
                fraud_indicators.append("Multiple urgency indicators")
            
            if boost_factor >= .2:
                boost_factor = .2

            # Apply boost but cap at reasonable levels
            boosted_score = min(consensus_score + boost_factor, 0.95)
            
            if boost_factor > 0:
                logger.info(f"   🚨 FRAUD PATTERN BOOST APPLIED:")
                logger.info(f"      Original consensus: {consensus_score:.3f}")
                logger.info(f"      Boost factor: +{boost_factor:.3f}")
                logger.info(f"      Final score: {boosted_score:.3f}")
                for indicator in fraud_indicators:
                    logger.info(f"      📍 {indicator}")
            
            return boosted_score
            
        except Exception as e:
            logger.error(f"Error applying fraud pattern boost: {e}")
            return consensus_score
    
    def _distribute_inference(self, session_id: str):
        """Distribute inference request to all banks via polling queue"""
        try:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found")
                return
                
            session = self.active_sessions[session_id]
            session.status = "collecting"
            
            logger.info(f"📤 DISTRIBUTING INFERENCE SESSION: {session_id}")
            logger.info(f"   🏦 Banks: {len(self.banks)}")
            logger.info(f"   📊 Features: {len(session.features)} anonymous features")
            
            # Create pending inference request for bank polling
            active_banks = [b.node_id for b in self.banks.values() if b.status in ["active", "online"]]
            
            # ramki
            session.raw_data['email_content'] = ''
            session.raw_data['sender_data'] = ''
            session.raw_data['receiver_data'] = ''

            self.pending_inferences[session_id] = {
                'session_id': session_id,
                'features': session.features,
                'raw_data': session.raw_data,  # Include raw data for banks
                'anonymized_accounts': session.anonymized_accounts or {},
                'use_case': session.use_case,
                'deadline': session.deadline.isoformat(),
                'pending_participants': active_banks.copy(),
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"   📋 Queued inference for banks: {active_banks}")
            logger.info(f"   ⏰ Banks have {self.inference_timeout} seconds to respond")
            
            # Start timeout monitoring
            threading.Thread(
                target=self._monitor_inference_timeout,
                args=(session_id,),
                daemon=True
            ).start()
                
        except Exception as e:
            logger.error(f"Distribution error: {e}")
    
    def _monitor_inference_timeout(self, session_id: str):
        """Monitor inference session for timeout"""
        try:
            import time
            
            # Wait for the timeout period
            time.sleep(self.inference_timeout)
            
            # Check if session is still active
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                expected_responses = len([b for b in self.banks.values() if b.status in ["active", "online"]])
                actual_responses = len(session.responses)
                
                logger.warning(f"⏰ TIMEOUT for session {session_id}")
                logger.info(f"   📊 Responses: {actual_responses}/{expected_responses}")
                
                if actual_responses > 0:
                    # Complete with partial responses
                    logger.info(f"   ✅ Completing with partial responses")
                    self._complete_inference(session_id, timeout=True)
                else:
                    # No responses received
                    logger.error(f"   ❌ No responses received - marking as failed")
                    if session_id in self.active_sessions:
                        del self.active_sessions[session_id]
                        
                    # Store failed result
                    self.session_results[session_id] = {
                        "session_id": session_id,
                        "status": "failed",
                        "error": "No bank responses received within timeout",
                        "completion_time": datetime.now().isoformat()
                    }
            
            # Clean up pending inference
            if session_id in self.pending_inferences:
                del self.pending_inferences[session_id]
                
        except Exception as e:
            logger.error(f"Timeout monitoring error: {e}")
    
    def _complete_inference(self, session_id: str, timeout: bool = False):
        """Complete inference session and calculate results"""
        try:
            session = self.active_sessions[session_id]
            session.status = "completed"
            
            responses = session.responses
            if not responses:
                logger.warning(f"No responses for session {session_id}")
                return
            
            logger.info(f"🎯 COMPLETING SESSION: {session_id}")
            logger.info(f"   📊 Responses received: {len(responses)}")
            
            # Calculate aggregated results with scenario-aware weighting
            individual_scores = {pid: r['fraud_score'] for pid, r in responses.items()}
            
            # Determine bank scenarios based on account ownership
            transaction_data = session.raw_data.get('transaction_data', {})
            sender_account = transaction_data.get('sender_account', 'UNKNOWN')
            receiver_account = transaction_data.get('receiver_account', 'UNKNOWN')
            
            # Calculate scenario-based weights
            scenario_weights = self._calculate_scenario_weights(individual_scores, sender_account, receiver_account)
            
            # Calculate weighted consensus
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for participant_id, score in individual_scores.items():
                weight = scenario_weights.get(participant_id, 0.5)
                total_weighted_score += score * weight
                total_weight += weight
            
            # ramki
            print("*** total_weight ***");
            print(total_weight)

            if total_weight > 0:
                consensus_score = total_weighted_score / total_weight
            else:
                consensus_score = sum(individual_scores.values()) / len(individual_scores)
            
            # Log individual scores with scenarios
            for pid, score in individual_scores.items():
                participant = self.participants.get(pid)
                specialty = participant.specialty if participant else "Unknown"
                weight = scenario_weights.get(pid, 0.5)
                weighted_score = score * weight
                # ramki
                logger.info(f"   🏦 {pid} ({specialty}): {score:.3f} (weight: {weight:.2f}, weighted: {weighted_score:.3f})")
            
            variance = sum((s - consensus_score) ** 2 for s in individual_scores.values()) / len(individual_scores)
            
            print(f"🚨 DEBUG: About to call fraud pattern boost")  # Force console output
            logger.info(f"🚨 DEBUG: About to call fraud pattern boost")
            
            # Apply fraud pattern boost
            final_score = self._apply_fraud_pattern_boost(consensus_score, session)
            
            logger.info(f"   📈 Consensus Score: {consensus_score:.3f}")
            logger.info(f"   📊 Variance: {variance:.3f}")
            logger.info(f"   🎯 Final Score: {final_score:.3f}")
            
            # Determine recommendation
            if final_score > 0.7:
                recommendation = "block"
            elif final_score > 0.3:
                recommendation = "review"
            else:
                recommendation = "approve"
            
            # Enhanced recommendation for high variance
            if variance > 0.1 and recommendation == "approve":
                recommendation = "approve_with_investigation"
                logger.info(f"   🔍 High variance detected - upgrading to investigation")
            
            logger.info(f"   🎯 RECOMMENDATION: {recommendation.upper()}")
            
            # Create specialist insights
            specialist_insights = []
            for participant_id, response in responses.items():
                participant = self.participants.get(participant_id)
                if participant and response['fraud_score'] > 0.5:
                    insight = {
                        "specialty": participant.specialty,
                        "risk_level": "high" if response['fraud_score'] > 0.7 else "medium",
                        "confidence": response['confidence']
                    }
                    specialist_insights.append(insight)
                    logger.info(f"   🔍 Specialist Alert: {participant.specialty} flagged {insight['risk_level']} risk")
            
            # Store results
            scores = [r['fraud_score'] for r in responses.values()]
            results = {
                "session_id": session_id,
                "use_case": session.use_case,
                "final_score": final_score,
                "consensus_score": consensus_score,
                "variance": variance,
                "recommendation": recommendation,
                # ramki
                "individual_scores": {pid: r['fraud_score'] for pid, r in responses.items()},
                "participant_consensus": {
                    "total": len(responses),
                    "high_risk": len([s for s in scores if s > 0.5]),
                    "low_risk": len([s for s in scores if s <= 0.5])
                },
                "specialist_insights": specialist_insights,
                "completion_time": datetime.now().isoformat(),
                "timeout": timeout
            }
            
            self.session_results[session_id] = results
            
            # Clean up
            del self.active_sessions[session_id]
            
            logger.info(f"✅ SESSION COMPLETE: {session_id}")
            logger.info(f"   🎯 Final Decision: {recommendation.upper()}")
            logger.info(f"   📊 Score: {final_score:.3f} | Variance: {variance:.3f}")
            logger.info(f"   🏦 Participant Consensus: {len([s for s in scores if s > 0.5])}/{len(scores)} flagged high risk")
            
        except Exception as e:
            logger.error(f"Completion error: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def cleanup_old_sessions():
            while True:
                try:
                    now = datetime.now()
                    # Clean up old results (keep for 1 hour)
                    cutoff = now - timedelta(hours=1)
                    old_sessions = [
                        sid for sid, result in self.session_results.items()
                        if datetime.fromisoformat(result['completion_time']) < cutoff
                    ]
                    for sid in old_sessions:
                        del self.session_results[sid]
                    
                    if old_sessions:
                        logger.info(f"🧹 Cleaned up {len(old_sessions)} old sessions")
                    
                    time.sleep(300)  # Run every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                    time.sleep(60)
        
        threading.Thread(target=cleanup_old_sessions, daemon=True).start()
    
    def run(self, debug=False):
        """Start the consortium hub server"""
        logger.info(f"🚀 Starting Consortium Hub on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)

def main():
    """Main function to start the consortium hub"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Consortium Hub Server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    hub = ConsortiumHub(port=args.port)
    hub.run(debug=args.debug)

if __name__ == "__main__":
    main()
