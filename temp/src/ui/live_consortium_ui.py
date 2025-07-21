#!/usr/bin/env python3
"""
Distributed Consortium UI - Live System Dashboard
Connects directly to the running consortium hub and bank processes
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Live Consortium Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def check_consortium_status():
    """Check if consortium hub is running"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=3)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def get_registered_banks():
    """Get list of registered banks"""
    try:
        response = requests.get("http://localhost:8080/participants", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get('participants', [])
        return []
    except Exception as e:
        return []

def submit_fraud_analysis(transaction_data):
    """Submit transaction for fraud analysis"""
    try:
        response = requests.post(
            "http://localhost:8080/inference", 
            json=transaction_data,
            timeout=10
        )
        if response.status_code == 200:
            return True, response.json()
        return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def main():
    st.markdown('<h1 class="main-header">🏦 Live Consortium Fraud Detection</h1>', unsafe_allow_html=True)
    
    # Check system status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌐 Consortium Hub")
        hub_online, hub_status = check_consortium_status()
        if hub_online:
            st.markdown('<p class="status-good">✅ Online</p>', unsafe_allow_html=True)
            if hub_status and 'uptime' in hub_status:
                st.write(f"Uptime: {hub_status['uptime']}")
        else:
            st.markdown('<p class="status-error">❌ Offline</p>', unsafe_allow_html=True)
            st.error("Start consortium hub: `python launcher.py start-hub`")
    
    with col2:
        st.markdown("### 🏦 Registered Banks")
        banks = get_registered_banks()
        if banks:
            st.markdown(f'<p class="status-good">✅ {len(banks)} Active</p>', unsafe_allow_html=True)
            for bank in banks:
                st.write(f"• {bank.get('node_id', 'Unknown')}: {bank.get('specialty', 'Unknown')}")
        else:
            st.markdown('<p class="status-warning">⚠️ No banks registered</p>', unsafe_allow_html=True)
            st.warning("Start banks: `python launcher.py start-all-windowed`")
    
    with col3:
        st.markdown("### 🔄 System Status")
        if hub_online and banks:
            st.markdown('<p class="status-good">✅ Fully Operational</p>', unsafe_allow_html=True)
        elif hub_online:
            st.markdown('<p class="status-warning">⚠️ Hub only</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ System down</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Only show analysis if system is working
    if not (hub_online and banks):
        st.warning("🔧 **System Setup Required**")
        st.info("""
        To use this dashboard:
        1. Start the consortium system: `python launcher.py start-all-windowed`
        2. Wait for all banks to register (green status above)
        3. Refresh this page
        """)
        return
    
    # Transaction Analysis Section
    st.markdown("## 🔍 Live Fraud Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Transaction Details")
        
        # Sample transaction for testing
        if st.button("📧 Load CEO Email Fraud Sample"):
            st.session_state.update({
                'amount': 485000.00,
                'sender_account': 'ACC789012',
                'receiver_account': 'ACC456789',
                'transaction_type': 'wire_transfer',
                'currency': 'USD',
                'country': 'US',
                'sender_name': 'ABC Manufacturing Corp',
                'receiver_name': 'Global Tech Solutions LLC',
                'description': 'Urgent strategic acquisition deposit - confidential transaction as discussed with CEO Sarah Wilson. Process immediately before market close Friday. Highly time sensitive acquisition opportunity.'
            })
        
        amount = st.number_input("💰 Amount", value=st.session_state.get('amount', 485000.0), min_value=0.01)
        sender_account = st.text_input("📤 Sender Account", value=st.session_state.get('sender_account', 'ACC789012'))
        receiver_account = st.text_input("📥 Receiver Account", value=st.session_state.get('receiver_account', 'ACC456789'))
        transaction_type = st.selectbox("🔄 Type", ['wire_transfer', 'ach_transfer', 'check', 'card_payment'], 
                                       index=['wire_transfer', 'ach_transfer', 'check', 'card_payment'].index(st.session_state.get('transaction_type', 'wire_transfer')))
        currency = st.selectbox("💱 Currency", ['USD', 'EUR', 'GBP', 'JPY'], 
                               index=['USD', 'EUR', 'GBP', 'JPY'].index(st.session_state.get('currency', 'USD')))
        country = st.text_input("🌍 Country", value=st.session_state.get('country', 'US'))
        
        st.markdown("### Parties")
        sender_name = st.text_input("👤 Sender Name", value=st.session_state.get('sender_name', 'ABC Manufacturing Corp'))
        receiver_name = st.text_input("👤 Receiver Name", value=st.session_state.get('receiver_name', 'Global Tech Solutions LLC'))
        description = st.text_area("📝 Description", value=st.session_state.get('description', 'Urgent strategic acquisition deposit - confidential transaction as discussed with CEO Sarah Wilson. Process immediately before market close Friday. Highly time sensitive acquisition opportunity.'))
    
    with col2:
        st.markdown("### 🛡️ Analysis Results")
        
        if st.button("🔍 Analyze Transaction", type="primary"):
            if not all([sender_account, receiver_account, sender_name, receiver_name]):
                st.error("Please fill in all required fields")
            else:
                transaction_data = {
                    'amount': amount,
                    'sender_account': sender_account,
                    'receiver_account': receiver_account,
                    'transaction_type': transaction_type,
                    'currency': currency,
                    'country': country,
                    'sender_name': sender_name,
                    'receiver_name': receiver_name,
                    'description': description,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Format for consortium hub NLP processing
                analysis_request = {
                    'transaction_data': transaction_data,
                    'email_content': description,  # Use description as email content
                    'use_case': 'fraud_detection'
                }
                
                with st.spinner("🔄 Analyzing transaction across consortium..."):
                    success, result = submit_fraud_analysis(analysis_request)
                
                if success:
                    st.success("✅ Analysis Complete")
                    
                    # Overall risk score
                    overall_score = result.get('overall_risk_score', 0)
                    risk_level = result.get('risk_level', 'UNKNOWN')
                    
                    # Color based on risk
                    if risk_level == 'LOW':
                        color = 'green'
                    elif risk_level == 'MEDIUM':
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    st.markdown(f"### Overall Risk: <span style='color:{color}'>{risk_level}</span> ({overall_score:.1%})", unsafe_allow_html=True)
                    
                    # Bank-specific results
                    bank_results = result.get('bank_results', {})
                    if bank_results:
                        st.markdown("#### Bank Analysis Results")
                        
                        for bank_id, bank_result in bank_results.items():
                            with st.expander(f"🏦 {bank_id.upper()} - {bank_result.get('specialty', 'Unknown')}"):
                                st.write(f"**Risk Score:** {bank_result.get('risk_score', 0):.1%}")
                                st.write(f"**Confidence:** {bank_result.get('confidence', 0):.1%}")
                                
                                # Features analyzed
                                features = bank_result.get('features_analyzed', [])
                                if features:
                                    st.write("**Key Factors:**")
                                    for feature in features[:5]:  # Top 5
                                        st.write(f"• {feature}")
                    
                    # Privacy info
                    st.markdown("#### 🔐 Privacy Protection")
                    st.info("""✅ **Banks NEVER see raw transaction data**
✅ Consortium converts to 35 anonymous behavioral features
✅ **All banks trained on same 35 anonymous features**
✅ Banks receive identical anonymized feature vectors
✅ Account details 1:1 anonymized with exact/wildcard matching
✅ Raw transaction data deleted after NLP processing""")
                    
                    # Technical details in expander
                    with st.expander("🔍 Technical Privacy Details"):
                        st.write("""
                        **How Privacy Protection Works:**
                        
                        1. **Consortium Hub receives** full transaction + email
                        2. **NLP processor extracts** 35 anonymous behavioral features
                        3. **Raw data is immediately deleted** (email, names, accounts)
                        4. **Same 35 anonymous features sent** to all banks
                        5. **All banks trained on identical feature set** but specialize through:
                           - Bank A: Wire transfer fraud expertise in training data
                           - Bank B: Identity verification fraud expertise in training data
                           - Bank C: Network/timing fraud expertise in training data
                        6. **Account Anonymization (1:1 mapping)** with 4 possibilities:
                           - Sender exact match + Receiver exact match (highest weight)
                           - Sender exact match + Receiver wildcard
                           - Sender wildcard + Receiver exact match  
                           - Sender wildcard + Receiver wildcard (lowest weight)
                        7. **Banks return risk scores** based on anonymous features only
                        
                        **Example Anonymous Features (identical for all banks):**
                        - `authority_score: 0.15` (CEO references detected)
                        - `urgency_score: 0.40` (time pressure language)
                        - `account_age_risk: 1.0` (new account flag)
                        - `timing_risk: 1.0` (Friday evening pattern)
                        - `sender_exact_match: 1.0` (sender account exact match)
                        - `receiver_wildcard: 1.0` (receiver account wildcard match)
                        """)
                    
                    
                else:
                    st.error(f"❌ Analysis failed: {result}")
        
        # System metrics
        if st.checkbox("📊 Show System Metrics"):
            st.markdown("#### Live System Status")
            
            # Refresh every 5 seconds
            placeholder = st.empty()
            
            for i in range(3):  # Show for 15 seconds
                current_banks = get_registered_banks()
                
                with placeholder.container():
                    for bank in current_banks:
                        st.write(f"**{bank.get('node_id', 'Unknown')}**: {bank.get('status', 'unknown')} - {bank.get('registered_at', 'unknown')}")
                
                time.sleep(5)

if __name__ == "__main__":
    main()
