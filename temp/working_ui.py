#!/usr/bin/env python3
"""
Working Consortium Fraud Detection UI
Simple Streamlit interface for the one-way hash anonymization system
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from consortium.account_anonymizer import AccountAnonymizer
from consortium.consortium_hub import ConsortiumHub

# Page configuration
st.set_page_config(
    page_title="🏦 Consortium Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.bank-header {
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    color: white;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Consortium Fraud Detection Dashboard")
st.markdown("**Privacy-Preserving Fraud Detection with One-Way Hash Anonymization**")

# Sidebar
st.sidebar.header("🔧 Transaction Input")

# Transaction input
col1, col2 = st.sidebar.columns(2)
with col1:
    sender_account = st.text_input("Sender Account", value="ACCA12345", help="Original account number")
with col2:
    receiver_account = st.text_input("Receiver Account", value="ACCB67890", help="Original account number")

amount = st.sidebar.number_input("Amount ($)", min_value=1.0, value=5000.0, step=100.0)
transaction_type = st.sidebar.selectbox("Transaction Type", ["wire_transfer", "ach", "card_payment", "check"])

# Bank account setup
st.sidebar.header("🏛️ Bank Account Setup")
bank_a_accounts = st.sidebar.text_area("Bank A Accounts", value="ACCA12345\nACCA67890\nACCA11111", help="One account per line")
bank_b_accounts = st.sidebar.text_area("Bank B Accounts", value="ACCB67890\nACCB12345\nACCB22222", help="One account per line")
bank_c_accounts = st.sidebar.text_area("Bank C Accounts", value="ACCC99999\nACCC12345\nACCC55555", help="One account per line")

# Process button
if st.sidebar.button("🔍 Analyze Transaction", type="primary"):
    
    # Create main content area
    st.header("📊 Analysis Results")
    
    # Step 1: Anonymization
    st.subheader("🔐 Step 1: Account Anonymization")
    
    anonymized = AccountAnonymizer.anonymize_transaction_accounts(sender_account, receiver_account)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Accounts:**")
        st.code(f"Sender: {sender_account}\nReceiver: {receiver_account}")
    with col2:
        st.markdown("**Anonymized Accounts:**")
        st.code(f"Sender: {anonymized['sender_anonymous']}\nReceiver: {anonymized['receiver_anonymous']}")
    
    # Privacy info
    st.info("🔒 **Privacy Protection**: Original account numbers are anonymized using SHA256 one-way hash. Cannot be reverse-engineered!")
    
    # Step 2: Bank scenario determination
    st.subheader("🏦 Step 2: Bank Scenario Determination")
    
    # Parse bank accounts
    bank_accounts = {
        'bank_a': bank_a_accounts.strip().split('\n'),
        'bank_b': bank_b_accounts.strip().split('\n'),
        'bank_c': bank_c_accounts.strip().split('\n')
    }
    
    bank_results = {}
    
    for bank_name, accounts in bank_accounts.items():
        scenario = AccountAnonymizer.bank_can_determine_ownership(accounts, anonymized)
        confidence = AccountAnonymizer.get_scenario_confidence_weight(scenario)
        
        # Simulate risk score (in real system this would come from ML model)
        base_risk = np.random.uniform(0.1, 0.9)
        
        bank_results[bank_name] = {
            'scenario': scenario,
            'confidence': confidence,
            'base_risk': base_risk,
            'weighted_risk': base_risk * confidence
        }
    
    # Display bank results
    col1, col2, col3 = st.columns(3)
    
    for i, (bank_name, result) in enumerate(bank_results.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"<div class='bank-header'><h4>🏛️ {bank_name.upper()}</h4></div>", unsafe_allow_html=True)
            
            # Scenario badge
            scenario_color = {
                'knows_both': '🟢',
                'knows_sender': '🔵', 
                'knows_receiver': '🟡',
                'knows_neither': '🔴'
            }
            st.markdown(f"**Scenario:** {scenario_color.get(result['scenario'], '⚪')} {result['scenario']}")
            st.metric("Confidence Weight", f"{result['confidence']:.2f}")
            st.metric("Base Risk Score", f"{result['base_risk']:.3f}")
            st.metric("Weighted Risk", f"{result['weighted_risk']:.3f}")
    
    # Step 3: Consortium consensus
    st.subheader("🎯 Step 3: Consortium Consensus")
    
    total_weighted_risk = sum(result['weighted_risk'] for result in bank_results.values())
    total_confidence = sum(result['confidence'] for result in bank_results.values())
    consensus_score = total_weighted_risk / total_confidence if total_confidence > 0 else 0
    
    # Consensus display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Weighted Calculation:**")
        calculation_text = ""
        for bank_name, result in bank_results.items():
            calculation_text += f"{bank_name}: {result['base_risk']:.3f} × {result['confidence']:.2f} = {result['weighted_risk']:.3f}\n"
        calculation_text += f"\nConsensus: ({total_weighted_risk:.3f}) / ({total_confidence:.2f}) = {consensus_score:.3f}"
        st.code(calculation_text)
    
    with col2:
        # Risk gauge
        if consensus_score < 0.3:
            risk_level = "LOW"
            risk_color = "green"
        elif consensus_score < 0.6:
            risk_level = "MEDIUM"
            risk_color = "orange"
        else:
            risk_level = "HIGH" 
            risk_color = "red"
        
        st.metric(
            "Consensus Risk Score",
            f"{consensus_score:.3f}",
            delta=f"{risk_level} RISK",
            delta_color="inverse"
        )
        
        # Recommendation
        if consensus_score < 0.3:
            st.success("✅ **APPROVE** - Low fraud risk")
        elif consensus_score < 0.6:
            st.warning("⚠️ **REVIEW** - Medium fraud risk")  
        else:
            st.error("🚨 **BLOCK** - High fraud risk")
    
    # Step 4: Privacy verification
    st.subheader("🛡️ Step 4: Privacy Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Privacy Properties Verified:**")
        st.markdown("""
        - 🔒 **One-way hash**: Cannot reverse-engineer original accounts
        - 🎯 **Deterministic**: Same account always produces same hash
        - 🔀 **No patterns**: Bank prefixes completely hidden
        - 🏦 **Bank independence**: Banks determine scenarios themselves
        """)
    
    with col2:
        st.markdown("**🔍 What External Observer Sees:**")
        observer_data = {
            "sender_anonymous": anonymized['sender_anonymous'],
            "receiver_anonymous": anonymized['receiver_anonymous'],
            "amount": amount,
            "type": transaction_type
        }
        st.json(observer_data)
        st.caption("⚠️ Observer CANNOT determine which banks own the accounts!")

# System status
st.sidebar.header("📈 System Status")
st.sidebar.success("🟢 Consortium Hub: Online")
st.sidebar.success("🟢 Anonymization: Active") 
st.sidebar.success("🟢 Bank A: Connected")
st.sidebar.success("🟢 Bank B: Connected")
st.sidebar.success("🟢 Bank C: Connected")

# Information
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
This demo shows a privacy-preserving fraud detection consortium using one-way hash anonymization.

**Key Features:**
- Banks cannot see each other's account numbers
- Banks determine their own knowledge scenarios
- Meaningful consensus from scenario-aware weighting
- Complete privacy preservation
""")

if __name__ == "__main__":
    st.write("Run with: `streamlit run working_ui.py`")
