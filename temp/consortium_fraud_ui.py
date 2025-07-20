"""
Consortium Fraud Detection Dashboard
Interactive Streamlit UI for Privacy-Preserving Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Any
import time

# Import the consortium classes
from consortium_comparison_score_prototype import BankSimulator, ConsortiumComparisonService

# Page configuration
st.set_page_config(
    page_title="Consortium Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .bank-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .safe-transaction {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .high-variance {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .consortium-intelligence {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .feature-input {
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_consortium_models():
    """Load all available bank models"""
    banks = {}
    bank_ids = ['bank_A', 'bank_B', 'bank_C']
    
    for bank_id in bank_ids:
        try:
            bank = BankSimulator(bank_id, f'{bank_id}_data.csv')
            if bank.load_model():
                banks[bank_id] = bank
            else:
                st.warning(f"⚠️ Model for {bank_id} not found. Please run training first.")
        except Exception as e:
            st.error(f"❌ Error loading {bank_id}: {e}")
    
    return banks

# Real banking transaction field mappings
TRANSACTION_FIELD_MAPPING = {
    # Transaction Amount & Basic Info (Features 0-9)
    0: {"name": "Transaction Amount (Normalized)", "category": "Amount", "description": "Transaction amount normalized by account history"},
    1: {"name": "Account Balance Ratio", "category": "Amount", "description": "Transaction amount / current account balance"},
    2: {"name": "Daily Spending Ratio", "category": "Amount", "description": "Transaction amount / average daily spending"},
    3: {"name": "Time Since Last Transaction", "category": "Timing", "description": "Hours since last transaction (normalized)"},
    4: {"name": "Transaction Hour", "category": "Timing", "description": "Hour of day (0-23, normalized)"},
    5: {"name": "Day of Week", "category": "Timing", "description": "Day of week (0-6, normalized)"},
    6: {"name": "Is Weekend", "category": "Timing", "description": "Whether transaction occurs on weekend"},
    7: {"name": "Is Holiday", "category": "Timing", "description": "Whether transaction occurs on holiday"},
    8: {"name": "Account Age", "category": "Account", "description": "Age of sender account in years (normalized)"},
    9: {"name": "Account Activity Level", "category": "Account", "description": "Historical transaction frequency"},
    
    # Geographic & Location Features (Features 10-19)
    10: {"name": "Sender Location Risk", "category": "Geography", "description": "Risk score of sender's location"},
    11: {"name": "Receiver Location Risk", "category": "Geography", "description": "Risk score of receiver's location"},
    12: {"name": "Cross-Border Transaction", "category": "Geography", "description": "Whether transaction crosses country borders"},
    13: {"name": "Distance from Home", "category": "Geography", "description": "Distance from account holder's typical location"},
    14: {"name": "High-Risk Country", "category": "Geography", "description": "Whether involves high-risk jurisdiction"},
    15: {"name": "Sender-Receiver Distance", "category": "Geography", "description": "Geographic distance between parties"},
    16: {"name": "ATM vs Online", "category": "Channel", "description": "Transaction channel type"},
    17: {"name": "Mobile vs Desktop", "category": "Channel", "description": "Device type used for transaction"},
    18: {"name": "New Device", "category": "Channel", "description": "Whether device is new for this account"},
    19: {"name": "VPN Usage", "category": "Channel", "description": "Whether VPN was detected"},
    
    # Behavioral & Pattern Features (Features 20-29)
    20: {"name": "Velocity Score", "category": "Behavior", "description": "Transaction frequency in last 24 hours"},
    21: {"name": "Round Amount Indicator", "category": "Behavior", "description": "Whether amount is suspiciously round"},
    22: {"name": "Receiver Account Age", "category": "Account", "description": "Age of receiver account"},
    23: {"name": "Receiver Risk Score", "category": "Account", "description": "Historical risk score of receiver"},
    24: {"name": "Previous Relationship", "category": "Relationship", "description": "History between sender and receiver"},
    25: {"name": "Business Hours", "category": "Timing", "description": "Whether during typical business hours"},
    26: {"name": "Amount Consistency", "category": "Behavior", "description": "Consistency with sender's typical amounts"},
    27: {"name": "Type Consistency", "category": "Behavior", "description": "Consistency with sender's typical transaction types"},
    28: {"name": "Failed Attempts", "category": "Behavior", "description": "Recent failed transaction attempts"},
    29: {"name": "Multiple Recipients", "category": "Behavior", "description": "Multiple recipients in short timeframe"}
}

def get_feature_info(feature_index):
    """Get human-readable information about a feature"""
    return TRANSACTION_FIELD_MAPPING.get(feature_index, {
        "name": f"Feature {feature_index}",
        "category": "Unknown",
        "description": "Feature description not available"
    })

@st.cache_data
def get_sample_transactions():
    """Generate realistic sample transactions with banking context"""
    samples = []
    
    # === FEATURED DEMO SCENARIO ===
    # Business Email Compromise - ABC Manufacturing Case (Perfect for UI Demo)
    # Optimized for realistic bank disagreement (variance = 0.004)
    bec_demo = [0.35, 0.45, 0.75, 0.40, 0.85, 0.35, 0.40, 0.70, 0.80, 0.90,  # Low amounts, high trust
                0.25, 0.35, 0.15, 0.30, 0.10, 0.70, 0.85, 0.90, 0.40, 0.35,  # Low geo risk, good identity  
                0.75, 0.35, 0.65, 0.55, 0.85, 0.75, 0.70, 0.75, 0.45, 0.40]   # Moderate email, low network
    samples.append((
        "🎯 DEMO: CEO Email Fraud - ABC Manufacturing ($485K Wire)", 
        bec_demo,
        """**🚨 SOPHISTICATED BUSINESS EMAIL COMPROMISE (BEC) FRAUD**

**The Fraud Email:**
```
From: CEO John Smith <jsmith@abc-manufacturing.com> [SPOOFED]
To: CFO Sarah Johnson
Subject: URGENT - Confidential Supplier Payment

Sarah, I'm in meetings with legal regarding the acquisition we discussed.
We need to wire $485,000 immediately to our new strategic supplier to
secure the deal. This is highly confidential - process ASAP.

Wire to: Global Tech Solutions LLC
Account: 4567-8901-2345-6789 (First National Bank)
Please handle personally. Thanks, John
```

**Why Each Bank Responds Differently:**
• **Bank A (Wire Specialist)**: Sees legitimate business customer with sufficient funds → **APPROVES** ✅
• **Bank B (Identity Expert)**: Notices recipient account opened just 3 days ago → **BLOCKS** 🚨  
• **Bank C (Network Analyst)**: Case too subtle for network patterns → **APPROVES** ✅

**🎯 Consortium Intelligence**: Bank B's expertise catches what others miss - but overall vote is APPROVE because majority see legitimate business. This creates valuable intelligence for investigation!"""
    ))
    
    # Low risk transaction: Regular payment to known merchant
    low_risk_features = [0.2, 0.1, 0.3, 0.2, 0.6, 0.3, 0.0, 0.0, 0.8, 0.7,  # Amount & Basic
                        0.1, 0.1, 0.0, 0.1, 0.0, 0.2, 0.8, 0.7, 0.0, 0.0,  # Geographic
                        0.2, 0.0, 0.9, 0.1, 0.9, 1.0, 0.9, 0.9, 0.0, 0.0]  # Behavioral
    samples.append(("Low Risk: Regular merchant payment ($50)", low_risk_features))
    
    # Medium risk transaction: Large amount to new recipient
    medium_risk_features = [0.8, 0.4, 0.7, 0.1, 0.9, 0.8, 0.0, 0.0, 0.6, 0.5,  # Amount & Basic
                           0.3, 0.4, 0.0, 0.3, 0.0, 0.6, 0.5, 0.3, 0.3, 0.1,  # Geographic
                           0.4, 0.3, 0.2, 0.6, 0.1, 0.7, 0.4, 0.3, 0.2, 0.3]  # Behavioral
    samples.append(("Medium Risk: Large payment to new recipient ($5,000)", medium_risk_features))
    
    # High risk transaction: Unusual international transfer
    high_risk_features = [0.95, 0.9, 0.9, 0.05, 0.1, 0.9, 1.0, 0.0, 0.3, 0.2,  # Amount & Basic
                          0.9, 0.9, 1.0, 0.8, 1.0, 0.9, 0.2, 0.1, 1.0, 1.0,  # Geographic
                          0.9, 1.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.1, 0.8, 0.7]  # Behavioral
    samples.append(("High Risk: Unusual international transfer ($50,000)", high_risk_features))
    
    return samples

def create_transaction_form():
    """Create a realistic transaction input form"""
    st.markdown("### 💳 Enter Transaction Details")
    
    # Transaction Basic Info
    with st.expander("💰 Transaction Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)
            sender_account = st.text_input("Sender Account", value="1234-5678-9012")
            sender_balance = st.number_input("Sender Account Balance ($)", min_value=0.0, value=5000.0, step=0.01)
            
        with col2:
            receiver_account = st.text_input("Receiver Account", value="9876-5432-1098")
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CAD"])
            transaction_type = st.selectbox("Transaction Type", ["Transfer", "Payment", "Withdrawal", "Deposit"])
    
    # Timing Information
    with st.expander("🕐 Timing Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_time = st.time_input("Transaction Time", value=None)
            day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
        with col2:
            is_holiday = st.checkbox("Is Holiday")
            is_business_hours = st.checkbox("During Business Hours", value=True)
    
    # Geographic Information
    with st.expander("🌍 Geographic Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            sender_country = st.selectbox("Sender Country", ["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia"])
            sender_city = st.text_input("Sender City", value="New York")
            
        with col2:
            receiver_country = st.selectbox("Receiver Country", ["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia"])
            receiver_city = st.text_input("Receiver City", value="Los Angeles")
    
    # Device & Channel Information
    with st.expander("📱 Device & Channel"):
        col1, col2 = st.columns(2)
        
        with col1:
            channel = st.selectbox("Transaction Channel", ["Online Banking", "Mobile App", "ATM", "Branch", "Phone"])
            device_type = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet", "ATM"])
            
        with col2:
            new_device = st.checkbox("New Device")
            vpn_detected = st.checkbox("VPN Detected")
    
    # Account Information
    with st.expander("👤 Account Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            sender_account_age = st.number_input("Sender Account Age (years)", min_value=0.0, value=2.5, step=0.1)
            sender_risk_score = st.slider("Sender Historical Risk Score", 0.0, 1.0, 0.1, 0.01)
            
        with col2:
            receiver_account_age = st.number_input("Receiver Account Age (years)", min_value=0.0, value=1.5, step=0.1)
            receiver_risk_score = st.slider("Receiver Historical Risk Score", 0.0, 1.0, 0.2, 0.01)
    
    # Convert to normalized features
    if st.button("🔄 Convert to Model Features"):
        features = convert_transaction_to_features(
            amount, sender_balance, transaction_time, day_of_week, is_holiday, is_business_hours,
            sender_country, receiver_country, channel, device_type, new_device, vpn_detected,
            sender_account_age, receiver_account_age, sender_risk_score, receiver_risk_score
        )
        return features
    
    return None

def convert_transaction_to_features(amount, sender_balance, transaction_time, day_of_week, is_holiday, 
                                  is_business_hours, sender_country, receiver_country, channel, 
                                  device_type, new_device, vpn_detected, sender_account_age, 
                                  receiver_account_age, sender_risk_score, receiver_risk_score):
    """Convert real transaction details to normalized features"""
    
    features = []
    
    # Amount features (0-3)
    features.append(min(amount / 10000, 1.0))  # Normalized amount
    features.append(min(amount / sender_balance, 1.0))  # Balance ratio
    features.append(min(amount / 500, 1.0))  # Daily spending ratio (assuming $500 avg)
    features.append(np.random.random())  # Time since last transaction (random for demo)
    
    # Timing features (4-7)
    hour = transaction_time.hour if transaction_time else 12
    features.append(hour / 23.0)  # Hour normalized
    
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                   "Friday": 4, "Saturday": 5, "Sunday": 6}
    features.append(day_mapping.get(day_of_week, 0) / 6.0)  # Day of week
    features.append(1.0 if day_of_week in ["Saturday", "Sunday"] else 0.0)  # Weekend
    features.append(1.0 if is_holiday else 0.0)  # Holiday
    
    # Account features (8-9)
    features.append(min(sender_account_age / 10, 1.0))  # Account age normalized
    features.append(np.random.random())  # Activity level (random for demo)
    
    # Geographic features (10-15)
    high_risk_countries = ["Somalia", "North Korea", "Iran", "Syria"]
    sender_risk = 0.8 if sender_country in high_risk_countries else np.random.uniform(0.0, 0.3)
    receiver_risk = 0.8 if receiver_country in high_risk_countries else np.random.uniform(0.0, 0.3)
    
    features.append(sender_risk)  # Sender location risk
    features.append(receiver_risk)  # Receiver location risk
    features.append(1.0 if sender_country != receiver_country else 0.0)  # Cross-border
    features.append(np.random.uniform(0.0, 0.5))  # Distance from home (random for demo)
    features.append(max(sender_risk, receiver_risk))  # High-risk country
    features.append(0.8 if sender_country != receiver_country else 0.2)  # Sender-receiver distance
    
    # Channel features (16-19)
    channel_mapping = {"ATM": 1.0, "Online Banking": 0.2, "Mobile App": 0.1, "Branch": 0.0, "Phone": 0.5}
    device_mapping = {"Desktop": 0.3, "Mobile": 0.1, "Tablet": 0.2, "ATM": 1.0}
    
    features.append(channel_mapping.get(channel, 0.5))  # ATM vs Online
    features.append(device_mapping.get(device_type, 0.5))  # Mobile vs Desktop
    features.append(1.0 if new_device else 0.0)  # New device
    features.append(1.0 if vpn_detected else 0.0)  # VPN usage
    
    # Behavioral features (20-29)
    features.append(np.random.uniform(0.0, 0.4))  # Velocity score (random for demo)
    features.append(1.0 if amount % 100 == 0 and amount >= 1000 else 0.0)  # Round amount
    features.append(min(receiver_account_age / 10, 1.0))  # Receiver account age
    features.append(receiver_risk_score)  # Receiver risk score
    features.append(np.random.uniform(0.0, 0.8))  # Previous relationship (random for demo)
    features.append(1.0 if is_business_hours else 0.0)  # Business hours
    features.append(1.0 - abs(amount - 500) / 5000)  # Amount consistency (assuming $500 typical)
    features.append(np.random.uniform(0.5, 1.0))  # Type consistency (random for demo)
    features.append(np.random.uniform(0.0, 0.3))  # Failed attempts (random for demo)
    features.append(np.random.uniform(0.0, 0.2))  # Multiple recipients (random for demo)
    
    return features

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

def create_feature_importance_chart(transaction_features):
    """Create a chart showing feature values and their risk levels"""
    if not transaction_features:
        return None
    
    feature_data = []
    for i, value in enumerate(transaction_features):
        field_info = get_feature_info(i)
        feature_data.append({
            'Feature': field_info['name'][:25] + "..." if len(field_info['name']) > 25 else field_info['name'],
            'Value': value,
            'Category': field_info['category'],
            'Risk Level': 'High' if value > 0.7 else 'Medium' if value > 0.3 else 'Low'
        })
    
    df = pd.DataFrame(feature_data)
    
    # Create color mapping
    color_map = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
    
    fig = px.bar(
        df, 
        x='Value', 
        y='Feature',
        color='Risk Level',
        color_discrete_map=color_map,
        orientation='h',
        title="Transaction Feature Analysis",
        labels={'Value': 'Normalized Feature Value', 'Feature': 'Transaction Features'}
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=True
    )
    
    return fig
    """Create a comprehensive visualization of consortium results"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Individual Bank Scores',
            'Consensus vs Variance',
            'Risk Progression',
            'Confidence Analysis'
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Individual bank scores
    banks = list(result['individual_scores'].keys())
    scores = [float(score) for score in result['individual_scores'].values()]
    
    fig.add_trace(
        go.Bar(x=banks, y=scores, name="Bank Scores", marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']),
        row=1, col=1
    )
    
    # Consensus vs Variance
    fig.add_trace(
        go.Scatter(
            x=[result['consensus_score']], 
            y=[result['variance_score']], 
            mode='markers',
            marker=dict(size=15, color='red'),
            name="Consensus vs Variance"
        ),
        row=1, col=2
    )
    
    # Risk progression (simulated)
    risk_components = [
        result['consensus_score'],
        result['network_anomaly_score'],
        result['final_comparison_score']
    ]
    component_names = ['Consensus', 'Network Anomaly', 'Final Score']
    
    fig.add_trace(
        go.Scatter(
            x=component_names,
            y=risk_components,
            mode='lines+markers',
            name="Risk Progression",
            line=dict(width=3)
        ),
        row=2, col=1
    )
    
    # Confidence breakdown
    confidence_colors = {'high': '#4CAF50', 'medium': '#FF9800', 'low': '#F44336'}
    fig.add_trace(
        go.Pie(
            labels=['Confidence', 'Uncertainty'],
            values=[0.8 if result['confidence_level'] == 'high' else 0.6 if result['confidence_level'] == 'medium' else 0.4, 
                   0.2 if result['confidence_level'] == 'high' else 0.4 if result['confidence_level'] == 'medium' else 0.6],
            marker_colors=[confidence_colors.get(result['confidence_level'], '#FF9800'), '#E0E0E0'],
            name="Confidence"
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Consortium Analysis Dashboard")
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Consortium Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Privacy-Preserving Multi-Bank Fraud Intelligence System")
    
    # Sidebar
    st.sidebar.markdown("## 🏦 System Configuration")
    
    # Load models
    with st.spinner("Loading consortium models..."):
        banks = load_consortium_models()
    
    if not banks:
        st.error("❌ No trained models available. Please run the training script first.")
        st.code("python consortium_comparison_score_prototype.py train")
        return
    
    # Display bank information
    st.sidebar.markdown("### 📊 Available Banks")
    for bank_id, bank in banks.items():
        with st.sidebar.container():
            st.markdown(f"**{bank_id.upper()}**")
            st.caption(f"Confidence: {bank.model_confidence:.3f}")
            st.caption(f"Model: XGBoost")
    
    # Initialize consortium
    consortium = ConsortiumComparisonService()
    for bank_id, bank in banks.items():
        consortium.register_bank(bank_id, bank)
    
    st.sidebar.success(f"✅ {len(banks)} banks loaded successfully")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 🔍 Transaction Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Sample Transactions", "Real Transaction Details", "Advanced Features", "Random Generation"]
        )
        
        transaction_features = None
        
        if input_method == "Real Transaction Details":
            transaction_features = create_transaction_form()
            
        elif input_method == "Advanced Features":
            st.markdown("### Enter Transaction Features (30 features)")
            st.info("💡 **Feature Mapping Guide**: Hover over feature names to see what they represent")
            
            # Create a more user-friendly input interface with real field mappings
            with st.expander("💰 Transaction Amount & Timing (Features 0-9)", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    features_0_4 = []
                    for i in range(5):
                        field_info = get_feature_info(i)
                        val = st.number_input(
                            f"{field_info['name']}", 
                            value=0.5, min_value=0.0, max_value=1.0, step=0.01, 
                            key=f"feat_{i}",
                            help=field_info['description']
                        )
                        features_0_4.append(val)
                
                with col_b:
                    features_5_9 = []
                    for i in range(5, 10):
                        field_info = get_feature_info(i)
                        val = st.number_input(
                            f"{field_info['name']}", 
                            value=0.5, min_value=0.0, max_value=1.0, step=0.01, 
                            key=f"feat_{i}",
                            help=field_info['description']
                        )
                        features_5_9.append(val)
            
            with st.expander("🌍 Geographic & Channel Features (Features 10-19)"):
                col_c, col_d = st.columns(2)
                with col_c:
                    features_10_19 = []
                    for i in range(10, 20):
                        field_info = get_feature_info(i)
                        val = st.number_input(
                            f"{field_info['name']}", 
                            value=0.5, min_value=0.0, max_value=1.0, step=0.01, 
                            key=f"feat_{i}",
                            help=field_info['description']
                        )
                        features_10_19.append(val)
                
                with col_d:
                    features_20_29 = []
                    for i in range(20, 30):
                        field_info = get_feature_info(i)
                        val = st.number_input(
                            f"{field_info['name']}", 
                            value=0.5, min_value=0.0, max_value=1.0, step=0.01, 
                            key=f"feat_{i}",
                            help=field_info['description']
                        )
                        features_20_29.append(val)
            
            transaction_features = features_0_4 + features_5_9 + features_10_19 + features_20_29
            
        elif input_method == "Sample Transactions":
            samples = get_sample_transactions()
            selected_sample = st.selectbox(
                "Choose a sample transaction:",
                options=range(len(samples)),
                format_func=lambda x: samples[x][0]
            )
            transaction_features = samples[selected_sample][1]
            
            # Show sample details
            st.info(f"📝 Selected: {samples[selected_sample][0]}")
            
        else:  # Random Generation
            risk_level = st.selectbox("Risk Level:", ["Low", "Medium", "High"])
            if st.button("🎲 Generate Random Transaction"):
                np.random.seed(int(time.time()))
                if risk_level == "Low":
                    transaction_features = np.random.normal(0.3, 0.2, 30).tolist()
                elif risk_level == "Medium":
                    transaction_features = np.random.normal(0.6, 0.3, 30).tolist()
                else:
                    transaction_features = np.random.normal(0.9, 0.2, 30).tolist()
                
                st.session_state['random_features'] = transaction_features
            
            if 'random_features' in st.session_state:
                transaction_features = st.session_state['random_features']
                st.success(f"✅ Generated {risk_level.lower()} risk transaction")
    
    with col2:
        st.markdown("## 📊 Consortium Analysis Results")
        
        if transaction_features and st.button("🚀 Analyze Transaction", type="primary"):
            with st.spinner("Analyzing transaction across consortium..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Get consortium results
                result = consortium.generate_comparison_score(transaction_features)
                
                # Clear progress bar
                progress_bar.empty()
                
                # Display results
                st.markdown("### 🎯 Consortium Intelligence Analysis")
                
                # Risk level indicator
                final_score = result['final_comparison_score']
                recommendation = result['recommendation']
                variance = result['variance_score']
                flagging_banks = result.get('flagging_banks', [])
                
                # Enhanced recommendation display with explanation
                if recommendation == "approve":
                    if variance > 0.1:  # High disagreement
                        st.markdown(f'''
                        <div class="metric-card">
                        <h3>✅ APPROVED WITH INTELLIGENCE</h3>
                        <p><strong>Final Risk Score:</strong> {final_score:.3f}</p>
                        <p><strong>🔍 Consortium Insight:</strong> While majority of banks approve this transaction, 
                        <strong>significant disagreement</strong> (variance: {variance:.3f}) indicates specialized 
                        fraud patterns detected by {', '.join(flagging_banks) if flagging_banks else 'some banks'}.</p>
                        <p><strong>💡 Action:</strong> Transaction proceeds but flagged for investigation due to expert disagreement.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-transaction"><h3>✅ APPROVED</h3><p>Final Risk Score: {final_score:.3f}</p><p>Strong consensus - low risk transaction</p></div>', unsafe_allow_html=True)
                elif recommendation == "review":
                    st.markdown(f'<div class="metric-card"><h3>⚠️ REVIEW REQUIRED</h3><p>Final Risk Score: {final_score:.3f}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="fraud-alert"><h3>🚫 BLOCKED</h3><p>Final Risk Score: {final_score:.3f}</p></div>', unsafe_allow_html=True)
                
                # Key metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("Consensus Score", f"{result['consensus_score']:.3f}")
                
                with col_m2:
                    st.metric("Variance", f"{result['variance_score']:.3f}")
                
                with col_m3:
                    st.metric("Network Anomaly", f"{result['network_anomaly_score']:.3f}")
                
                with col_m4:
                    st.metric("Confidence", result['confidence_level'].upper())
                
                # Gauges
                st.markdown("### 📈 Risk Visualization")
                gauge_col1, gauge_col2 = st.columns(2)
                
                with gauge_col1:
                    st.plotly_chart(
                        create_score_gauge(final_score, "Final Risk Score"),
                        use_container_width=True
                    )
                
                with gauge_col2:
                    st.plotly_chart(
                        create_score_gauge(result['consensus_score'], "Consensus Score"),
                        use_container_width=True
                    )
                
                # Bank comparison with detailed explanations
                st.markdown("### 🏦 Individual Bank Analysis")
                st.plotly_chart(
                    create_bank_comparison_chart(result['individual_scores']),
                    use_container_width=True
                )
                
                # Detailed bank reasoning
                individual_scores = result['individual_scores']
                banks = list(individual_scores.keys())
                scores = [float(score) for score in individual_scores.values()]
                
                st.markdown("#### 🧠 Why Each Bank Made Their Decision:")
                
                for i, (bank, score) in enumerate(zip(banks, scores)):
                    if bank == 'bank_A':
                        specialty = "🏦 **Wire Transfer Specialist**"
                        if score < 0.3:
                            reasoning = f"**Score: {score:.3f} → APPROVE** ✅\n\n• Sees legitimate business customer (ABC Manufacturing)\n• $485K within normal business range\n• **Misses**: Email sophistication - focuses on transaction amounts/geography"
                            color = "safe-transaction"
                        else:
                            reasoning = f"**Score: {score:.3f} → BLOCK** 🚨\n\n• Detected high-value transaction anomaly\n• Geographic or amount-based risk factors triggered"
                            color = "fraud-alert"
                    elif bank == 'bank_B':
                        specialty = "🔍 **Identity Verification Expert**"
                        if score > 0.5:
                            reasoning = f"**Score: {score:.3f} → BLOCK** 🚨\n\n• **CAUGHT THE FRAUD!** Recipient account only 3 days old\n• Poor identity verification on receiver\n• New business entity with minimal documentation"
                            color = "fraud-alert"
                        else:
                            reasoning = f"**Score: {score:.3f} → APPROVE** ✅\n\n• Identity verification checks passed\n• Established account relationships"
                            color = "safe-transaction"
                    else:  # bank_C
                        specialty = "🌐 **Network Pattern Analyst**"
                        if score < 0.3:
                            reasoning = f"**Score: {score:.3f} → APPROVE** ✅\n\n• Case too subtle for network detection\n• No obvious cross-institutional patterns\n• **Misses**: This sophisticated individual case"
                            color = "safe-transaction"
                        else:
                            reasoning = f"**Score: {score:.3f} → BLOCK** 🚨\n\n• Detected network patterns across institutions\n• Email/communication anomalies flagged"
                            color = "fraud-alert"
                    
                    st.markdown(f'''
                    <div class="{color}">
                    <h4>{specialty} - {bank.upper()}</h4>
                    {reasoning}
                    </div>
                    ''', unsafe_allow_html=True)
                    st.markdown("")  # Add spacing
                
                # Feature importance visualization
                st.markdown("### 📊 Transaction Feature Analysis")
                feature_chart = create_feature_importance_chart(transaction_features)
                if feature_chart:
                    st.plotly_chart(feature_chart, use_container_width=True)
                
                # Feature explanation for the transaction
                st.markdown("### 🔍 Risk Factor Breakdown")
                if transaction_features:
                    # Group features by category for better understanding
                    categories = {}
                    for i, feature_val in enumerate(transaction_features):
                        field_info = get_feature_info(i)
                        category = field_info['category']
                        if category not in categories:
                            categories[category] = []
                        categories[category].append({
                            'name': field_info['name'],
                            'value': feature_val,
                            'description': field_info['description'],
                            'risk_level': 'High' if feature_val > 0.7 else 'Medium' if feature_val > 0.3 else 'Low'
                        })
                    
                    # Display categorized features with summary stats
                    col_cat1, col_cat2 = st.columns(2)
                    
                    categories_list = list(categories.items())
                    mid_point = len(categories_list) // 2
                    
                    with col_cat1:
                        for category, features in categories_list[:mid_point]:
                            with st.expander(f"📊 {category} Features ({len(features)} features)"):
                                high_risk_count = sum(1 for f in features if f['risk_level'] == 'High')
                                if high_risk_count > 0:
                                    st.warning(f"⚠️ {high_risk_count} high-risk factors detected")
                                
                                for feature in features:
                                    risk_color = "🔴" if feature['risk_level'] == 'High' else "🟡" if feature['risk_level'] == 'Medium' else "🟢"
                                    st.write(f"{risk_color} **{feature['name']}**: {feature['value']:.3f}")
                                    st.caption(f"{feature['description']} (Risk: {feature['risk_level']})")
                    
                    with col_cat2:
                        for category, features in categories_list[mid_point:]:
                            with st.expander(f"📊 {category} Features ({len(features)} features)"):
                                high_risk_count = sum(1 for f in features if f['risk_level'] == 'High')
                                if high_risk_count > 0:
                                    st.warning(f"⚠️ {high_risk_count} high-risk factors detected")
                                
                                for feature in features:
                                    risk_color = "🔴" if feature['risk_level'] == 'High' else "🟡" if feature['risk_level'] == 'Medium' else "🟢"
                                    st.write(f"{risk_color} **{feature['name']}**: {feature['value']:.3f}")
                                    st.caption(f"{feature['description']} (Risk: {feature['risk_level']})")
                
                # Detailed results
                with st.expander("📋 Detailed Technical Results", expanded=False):
                    st.json(result)
                
                # Flagging banks info
                if result['flagging_banks']:
                    st.markdown("### 🚩 Banks Flagging This Transaction")
                    for bank in result['flagging_banks']:
                        st.warning(f"🏦 {bank.upper()} flagged this transaction as high risk")
                else:
                    st.success("✅ No banks flagged this transaction as high risk")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔒 Privacy & Security")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.info("🏦 **Data Privacy**\nBank data never leaves premises")
    
    with col_info2:
        st.info("🔐 **Model Security**\nModel weights remain proprietary")
    
    with col_info3:
        st.info("🤝 **Collaborative Intelligence**\nShared insights without exposure")

if __name__ == "__main__":
    main()
