#!/usr/bin/env python3
"""
Distributed Consortium UI - Connects to Consortium Hub via HTTP
Runs as separate Streamlit process
"""

import streamlit as st
import requests
import time
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONSORTIUM_HUB_URL = "http://localhost:8080"

def check_consortium_connection():
    """Check if consortium hub is available"""
    try:
        response = requests.get(f"{CONSORTIUM_HUB_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def get_sample_transactions():
    """Get sample transactions for testing"""
    samples = []
    
    # BEC Demo - Optimized for realistic disagreement
    bec_demo = [0.35, 0.45, 0.75, 0.40, 0.85, 0.35, 0.40, 0.70, 0.80, 0.90,
                0.25, 0.35, 0.15, 0.30, 0.10, 0.70, 0.85, 0.90, 0.40, 0.35,
                0.75, 0.35, 0.65, 0.55, 0.85, 0.75, 0.70, 0.75, 0.45, 0.40]
    
    samples.append((
        "🎯 DEMO: CEO Email Fraud - ABC Manufacturing ($485K Wire)", 
        bec_demo,
        """**🚨 SOPHISTICATED BUSINESS EMAIL COMPROMISE (BEC) FRAUD**

**Scenario:** CEO email spoofing requesting $485K wire transfer to new supplier
- Legitimate business customer (ABC Manufacturing)
- Sophisticated social engineering attack  
- New recipient account (only 3 days old)
- High-value wire transfer within business norms

**Expected Consortium Response:**
• **Bank A (Wire Specialist)**: Should approve - sees legitimate business amounts
• **Bank B (Identity Expert)**: Should block - catches new recipient account
• **Bank C (Network Analyst)**: Should approve - too subtle for network detection

**Demonstrates:** How specialized expertise creates investigation intelligence even when majority approves"""
    ))
    
    # Low risk transaction
    low_risk = [0.1, 0.05, 0.2, 0.3, 0.5, 0.2, 0.0, 0.0, 0.9, 0.8,
                0.05, 0.1, 0.0, 0.05, 0.0, 0.1, 0.9, 0.8, 0.0, 0.0,
                0.1, 0.0, 0.95, 0.05, 0.95, 1.0, 0.95, 0.95, 0.0, 0.0]
    samples.append(("✅ Low Risk: Regular merchant payment ($50)", low_risk))
    
    # High risk transaction
    high_risk = [0.95, 0.9, 0.9, 0.05, 0.1, 0.9, 1.0, 0.0, 0.3, 0.2,
                 0.9, 0.9, 1.0, 0.8, 1.0, 0.9, 0.2, 0.1, 1.0, 1.0,
                 0.9, 1.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.1, 0.8, 0.7]
    samples.append(("🚨 High Risk: Unusual international transfer ($50,000)", high_risk))
    
    return samples

def submit_inference_request(features, use_case="fraud_detection"):
    """Submit inference request to consortium hub"""
    try:
        payload = {
            "features": features,
            "use_case": use_case
        }
        
        response = requests.post(
            f"{CONSORTIUM_HUB_URL}/inference",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, str(e)

def get_inference_results(session_id):
    """Get results for a specific inference session"""
    try:
        response = requests.get(f"{CONSORTIUM_HUB_URL}/results/{session_id}", timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, str(e)

def get_consortium_participants():
    """Get list of consortium participants"""
    try:
        response = requests.get(f"{CONSORTIUM_HUB_URL}/participants", timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, str(e)

def create_score_gauge(score, title):
    """Create a gauge chart for risk scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_bank_comparison_chart(individual_scores):
    """Create a bar chart comparing individual bank scores"""
    if not individual_scores:
        return None
    
    banks = list(individual_scores.keys())
    scores = [float(score) for score in individual_scores.values()]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig = go.Figure(data=[
        go.Bar(
            x=banks,
            y=scores,
            marker_color=colors[:len(banks)],
            text=[f'{score:.3f}' for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Individual Bank Risk Scores",
        xaxis_title="Banks",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def main():
    """Main Streamlit UI function"""
    st.set_page_config(
        page_title="Distributed Consortium Intelligence",
        page_icon="🏛️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .safe-transaction {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-alert {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🏛️ Distributed Consortium Intelligence Platform")
    st.markdown("### Privacy-Preserving AI/ML Collaboration Across Sovereign Organizations")
    
    # Check consortium connection
    is_connected, health_data = check_consortium_connection()
    
    if not is_connected:
        st.error("❌ Cannot connect to Consortium Hub. Please ensure the hub is running on http://localhost:8080")
        st.markdown("""
        **To start the consortium:**
        1. Run: `python consortium_hub.py`
        2. Run: `python participant_node.py --all`
        3. Refresh this page
        """)
        return
    
    # Connection status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Hub Status", "🟢 Connected")
    
    with col2:
        st.metric("Participants", health_data.get('participants', 0))
    
    with col3:
        st.metric("Active Sessions", health_data.get('active_sessions', 0))
    
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    # Participants status
    st.markdown("## 🏦 Consortium Participants")
    success, participants_data = get_consortium_participants()
    
    if success and participants_data.get('participants'):
        participants_df = pd.DataFrame(participants_data['participants'])
        st.dataframe(participants_df, use_container_width=True)
    else:
        st.warning("⚠️ No participants registered. Please start participant nodes.")
        st.markdown("Run: `python participant_node.py --all`")
    
    # Transaction Analysis
    st.markdown("## 📊 Transaction Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Sample Transactions", "Manual Input"],
        index=0  # Default to Sample Transactions
    )
    
    transaction_features = None
    scenario_description = None
    
    if input_method == "Sample Transactions":
        samples = get_sample_transactions()
        
        # Sample selection
        sample_names = [sample[0] for sample in samples]
        selected_sample = st.selectbox("Select a sample transaction:", sample_names)
        
        # Find selected sample
        for name, features, description in samples:
            if name == selected_sample:
                transaction_features = features
                if len(samples[0]) > 2:  # Has description
                    scenario_description = description
                break
        
        # Display scenario description
        if scenario_description:
            st.markdown("### 📋 Scenario Details")
            st.markdown(scenario_description)
    
    else:
        st.markdown("### 💳 Manual Transaction Input")
        st.info("Enter normalized feature values (0.0 to 1.0)")
        
        # Simple feature input for demo
        col1, col2 = st.columns(2)
        
        with col1:
            amount_score = st.slider("Amount Risk Score", 0.0, 1.0, 0.5, 0.01)
            timing_score = st.slider("Timing Risk Score", 0.0, 1.0, 0.3, 0.01)
            geo_score = st.slider("Geographic Risk Score", 0.0, 1.0, 0.2, 0.01)
        
        with col2:
            identity_score = st.slider("Identity Risk Score", 0.0, 1.0, 0.4, 0.01)
            behavior_score = st.slider("Behavioral Risk Score", 0.0, 1.0, 0.3, 0.01)
            device_score = st.slider("Device Risk Score", 0.0, 1.0, 0.2, 0.01)
        
        # Create full feature vector (pad to 30 features)
        base_features = [amount_score, timing_score, geo_score, identity_score, behavior_score, device_score]
        transaction_features = base_features + [0.5] * 24  # Pad to 30 features
    
    # Analysis section
    st.markdown("## 📊 Consortium Analysis")
    
    if transaction_features and st.button("🚀 Analyze Transaction", type="primary"):
        with st.spinner("Submitting to consortium for distributed analysis..."):
            # Submit inference request
            success, result = submit_inference_request(transaction_features)
            
            if not success:
                st.error(f"❌ Failed to submit inference request: {result}")
                return
            
            session_id = result['session_id']
            st.success(f"✅ Inference submitted! Session ID: {session_id}")
            
            # Display submission details
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Participants Contacted", result.get('participants', 0))
            with col2:
                estimated_completion = result.get('estimated_completion', 'Unknown')
                st.metric("Estimated Completion", estimated_completion.split('T')[1][:8] if 'T' in estimated_completion else estimated_completion)
        
        # Poll for results
        with st.spinner("Waiting for consortium responses..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_wait = 35  # seconds
            poll_interval = 1
            
            for i in range(max_wait):
                # Get current status
                success, status_result = get_inference_results(session_id)
                
                if success:
                    if 'final_score' in status_result:
                        # Results are ready
                        progress_bar.progress(100)
                        status_text.success("✅ Analysis complete!")
                        break
                    else:
                        # Still waiting
                        responses_received = status_result.get('responses_received', 0)
                        total_participants = status_result.get('total_participants', 1)
                        progress = min(95, (responses_received / total_participants) * 100)
                        progress_bar.progress(int(progress))
                        
                        time_remaining = status_result.get('time_remaining', 0)
                        status_text.info(f"⏳ Received {responses_received}/{total_participants} responses. Time remaining: {time_remaining:.1f}s")
                
                time.sleep(poll_interval)
            
            progress_bar.empty()
            status_text.empty()
        
        # Get final results
        success, final_result = get_inference_results(session_id)
        
        if success and 'final_score' in final_result:
            # Display results
            st.markdown("### 🎯 Consortium Intelligence Analysis")
            
            final_score = final_result['final_score']
            consensus_score = final_result['consensus_score']
            variance = final_result['variance']
            recommendation = final_result['recommendation']
            
            # Enhanced recommendation display
            if recommendation == "approve_with_investigation":
                st.markdown(f'''
                <div class="warning-alert">
                <h3>✅ APPROVED WITH INTELLIGENCE</h3>
                <p><strong>Final Risk Score:</strong> {final_score:.3f}</p>
                <p><strong>🔍 Consortium Insight:</strong> While majority of banks approve this transaction, 
                <strong>significant disagreement</strong> (variance: {variance:.3f}) indicates specialized 
                fraud patterns detected by expert banks.</p>
                <p><strong>💡 Action:</strong> Transaction proceeds but flagged for investigation due to expert disagreement.</p>
                </div>
                ''', unsafe_allow_html=True)
            elif recommendation == "approve":
                st.markdown(f'<div class="safe-transaction"><h3>✅ APPROVED</h3><p>Final Risk Score: {final_score:.3f}</p><p>Strong consensus - low risk transaction</p></div>', unsafe_allow_html=True)
            elif recommendation == "review":
                st.markdown(f'<div class="warning-alert"><h3>⚠️ REVIEW REQUIRED</h3><p>Final Risk Score: {final_score:.3f}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="fraud-alert"><h3>🚫 BLOCKED</h3><p>Final Risk Score: {final_score:.3f}</p></div>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Score", f"{final_score:.3f}")
            with col2:
                st.metric("Consensus Score", f"{consensus_score:.3f}")
            with col3:
                st.metric("Variance", f"{variance:.3f}")
            with col4:
                st.metric("Participants", final_result.get('participant_consensus', {}).get('total', 0))
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_score_gauge(final_score, "Final Risk Score")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'individual_scores' in final_result:
                    fig = create_bank_comparison_chart(final_result['individual_scores'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            # Individual bank analysis
            if 'individual_scores' in final_result:
                st.markdown("### 🏦 Individual Bank Analysis")
                
                bank_names = {
                    'bank_A': '🏦 Bank A (Wire Transfer Specialist)',
                    'bank_B': '🔍 Bank B (Identity Verification Expert)',
                    'bank_C': '🌐 Bank C (Network Pattern Analyst)'
                }
                
                for bank_id, score in final_result['individual_scores'].items():
                    bank_name = bank_names.get(bank_id, bank_id)
                    score_val = float(score)
                    decision = "BLOCK 🚨" if score_val > 0.5 else "APPROVE ✅"
                    
                    color_class = "fraud-alert" if score_val > 0.5 else "safe-transaction"
                    
                    st.markdown(f'''
                    <div class="{color_class}">
                    <h4>{bank_name}</h4>
                    <p><strong>Score:</strong> {score_val:.3f} → {decision}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Specialist insights
            if 'specialist_insights' in final_result and final_result['specialist_insights']:
                st.markdown("### 💡 Specialist Insights")
                
                for insight in final_result['specialist_insights']:
                    specialty = insight['specialty'].replace('_', ' ').title()
                    risk_level = insight['risk_level'].upper()
                    confidence = insight.get('confidence', 1.0)
                    
                    st.info(f"**{specialty}** detected **{risk_level}** risk (confidence: {confidence:.2f})")
        
        else:
            st.error("❌ Failed to get final results or analysis timed out")

if __name__ == "__main__":
    main()
