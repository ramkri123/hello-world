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
                features = data['features']
                use_case = data.get('use_case', 'fraud_detection')
                
                # Check if enough participants
                active_participants = [p for p in self.participants.values() if p.status == "active"]
                if len(active_participants) < self.min_participants:
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
    
    def _distribute_inference(self, session_id: str):
        """Distribute inference request to all participants"""
        try:
            session = self.active_sessions[session_id]
            session.status = "collecting"
            
            # In a real implementation, this would make HTTP requests to participant endpoints
            # For now, we'll simulate the distribution
            logger.info(f"📤 Distributing inference to {len(self.participants)} participants")
            
            # Wait for timeout
            time.sleep(self.inference_timeout)
            
            # Check if session is still active and incomplete
            if session_id in self.active_sessions and session.status == "collecting":
                logger.warning(f"⏰ Session {session_id} timed out")
                self._complete_inference(session_id, timeout=True)
                
        except Exception as e:
            logger.error(f"Distribution error: {e}")
    
    def _complete_inference(self, session_id: str, timeout: bool = False):
        """Complete inference session and calculate results"""
        try:
            session = self.active_sessions[session_id]
            session.status = "completed"
            
            responses = session.responses
            if not responses:
                logger.warning(f"No responses for session {session_id}")
                return
            
            # Calculate aggregated results
            scores = [r['risk_score'] for r in responses.values()]
            confidences = [r['confidence'] for r in responses.values()]
            
            # Consensus scoring
            consensus_score = sum(scores) / len(scores)
            variance = sum((s - consensus_score) ** 2 for s in scores) / len(scores)
            final_score = consensus_score  # Can implement weighted scoring
            
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
            
            # Create specialist insights
            specialist_insights = []
            for participant_id, response in responses.items():
                participant = self.participants.get(participant_id)
                if participant and response['risk_score'] > 0.5:
                    specialist_insights.append({
                        "specialty": participant.specialty,
                        "risk_level": "high" if response['risk_score'] > 0.7 else "medium",
                        "confidence": response['confidence']
                    })
            
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
            
            logger.info(f"✅ Completed session {session_id}: {recommendation} (score: {final_score:.3f}, variance: {variance:.3f})")
            
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
