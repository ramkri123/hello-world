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

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    created_at: datetime
    deadline: datetime
    responses: Dict[str, Any]
    status: str = "pending"  # pending, collecting, completed, timeout

class ConsortiumHub:
    def __init__(self, port=8080):
        self.app = Flask(__name__)
        CORS(self.app)
        self.port = port
        
        # State management
        self.participants: Dict[str, ParticipantInfo] = {}
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
                    from privacy_preserving_nlp import PrivacyPreservingNLP
                    nlp = PrivacyPreservingNLP()
                    
                    transaction_data = data.get('transaction_data', {})
                    email_content = data.get('email_content', '')
                    sender_data = data.get('sender_data', None)
                    receiver_data = data.get('receiver_data', None)
                    
                    logger.info(f"🔄 PROCESSING RAW TRANSACTION DATA:")
                    logger.info(f"   📧 Email Content: {len(email_content)} characters")
                    logger.info(f"   💰 Transaction: ${transaction_data.get('amount', 0):,.2f}")
                    logger.info(f"   🏦 Converting to anonymous features...")
                    
                    # Extract anonymous features using NLP
                    features = nlp.convert_to_anonymous_features(
                        transaction_data, email_content, sender_data, receiver_data
                    )
                    
                    logger.info(f"   ✅ Generated {len(features)} anonymous features")
                    logger.info(f"   🔒 EMAIL CONTENT DELETED (privacy preserved)")
                    
                else:
                    # Traditional features array
                    features = data['features']
                
                use_case = data.get('use_case', 'fraud_detection')
                
                logger.info(f"🔄 NEW INFERENCE REQUEST:")
                logger.info(f"   📊 Use Case: {use_case}")
                logger.info(f"   📈 Features: {len(features)} values - First 10: {[f'{f:.3f}' for f in features[:10]]}")
                logger.info(f"   🌐 Source IP: {request.remote_addr}")
                
                # Check if enough participants
                active_participants = [p for p in self.participants.values() if p.status == "active"]
                if len(active_participants) < self.min_participants:
                    logger.error(f"❌ Not enough participants: Need {self.min_participants}, have {len(active_participants)}")
                    return jsonify({
                        "error": f"Not enough participants. Need {self.min_participants}, have {len(active_participants)}"
                    }), 400
                
                # Create session
                session_id = str(uuid.uuid4())
                session = InferenceSession(
                    session_id=session_id,
                    use_case=use_case,
                    features=features,
                    created_at=datetime.now(),
                    deadline=datetime.now() + timedelta(seconds=self.inference_timeout),
                    responses={}
                )
                
                self.active_sessions[session_id] = session
                
                logger.info(f"🎯 CREATED SESSION: {session_id}")
                logger.info(f"   ⏰ Deadline: {session.deadline.strftime('%H:%M:%S')}")
                logger.info(f"   🏦 Will distribute to: {[p.node_id for p in active_participants]}")
                
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
                    "participants": len(active_participants),
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
                            "use_case": inference_data['use_case'],
                            "deadline": inference_data['deadline']
                        })
                
                # No pending inference for this participant
                return jsonify({"message": "No pending inference"}), 204
                
            except Exception as e:
                logger.error(f"Poll inference error: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _distribute_inference(self, session_id: str):
        """Distribute inference request to all participants via polling queue"""
        try:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found")
                return
                
            session = self.active_sessions[session_id]
            session.status = "collecting"
            
            logger.info(f"📤 DISTRIBUTING INFERENCE SESSION: {session_id}")
            logger.info(f"   🏦 Participants: {len(self.participants)}")
            logger.info(f"   📊 Features: {len(session.features)} anonymous features")
            
            # Create pending inference request for bank polling
            active_participants = [p.node_id for p in self.participants.values() if p.status == "active"]
            
            self.pending_inferences[session_id] = {
                'session_id': session_id,
                'features': session.features,
                'use_case': session.use_case,
                'deadline': session.deadline.isoformat(),
                'pending_participants': active_participants.copy(),
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"   📋 Queued inference for participants: {active_participants}")
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
                expected_responses = len([p for p in self.participants.values() if p.status == "active"])
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
            
            # Calculate aggregated results
            scores = [r['risk_score'] for r in responses.values()]
            confidences = [r['confidence'] for r in responses.values()]
            
            # Log individual scores
            for pid, response in responses.items():
                participant = self.participants.get(pid)
                specialty = participant.specialty if participant else "Unknown"
                logger.info(f"   🏦 {pid} ({specialty}): {response['risk_score']:.3f}")
            
            # Consensus scoring
            consensus_score = sum(scores) / len(scores)
            variance = sum((s - consensus_score) ** 2 for s in scores) / len(scores)
            final_score = consensus_score  # Can implement weighted scoring
            
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
                if participant and response['risk_score'] > 0.5:
                    insight = {
                        "specialty": participant.specialty,
                        "risk_level": "high" if response['risk_score'] > 0.7 else "medium",
                        "confidence": response['confidence']
                    }
                    specialist_insights.append(insight)
                    logger.info(f"   🔍 Specialist Alert: {participant.specialty} flagged {insight['risk_level']} risk")
            
            # Store results
            results = {
                "session_id": session_id,
                "use_case": session.use_case,
                "final_score": final_score,
                "consensus_score": consensus_score,
                "variance": variance,
                "recommendation": recommendation,
                "individual_scores": {pid: r['risk_score'] for pid, r in responses.items()},
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
