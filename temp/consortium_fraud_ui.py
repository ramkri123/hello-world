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

@st.cache_data
def get_sample_transactions():
    """Generate sample transactions for quick testing"""
    np.random.seed(42)
    samples = []
    
    # Low risk transaction
    low_risk = np.random.normal(0.3, 0.2, 30).tolist()
    samples.append(("Low Risk Transaction", low_risk))
    
    # Medium risk transaction
    medium_risk = np.random.normal(0.6, 0.3, 30).tolist()
    samples.append(("Medium Risk Transaction", medium_risk))
    
    # High risk transaction
    high_risk = np.random.normal(0.9, 0.2, 30).tolist()
    samples.append(("High Risk Transaction", high_risk))
    
    return samples

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

def create_consensus_visualization(result):
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
            ["Manual Input", "Sample Transactions", "Random Generation"]
        )
        
        transaction_features = None
        
        if input_method == "Manual Input":
            st.markdown("### Enter Transaction Features (30 features)")
            
            # Create a more user-friendly input interface
            with st.expander("💰 Transaction Amount & Basic Info", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    features_0_4 = []
                    for i in range(5):
                        val = st.number_input(f"Feature {i}", value=0.5, min_value=0.0, max_value=1.0, step=0.01, key=f"feat_{i}")
                        features_0_4.append(val)
                
                with col_b:
                    features_5_9 = []
                    for i in range(5, 10):
                        val = st.number_input(f"Feature {i}", value=0.5, min_value=0.0, max_value=1.0, step=0.01, key=f"feat_{i}")
                        features_5_9.append(val)
            
            with st.expander("📍 Location & Behavioral Features"):
                col_c, col_d = st.columns(2)
                with col_c:
                    features_10_19 = []
                    for i in range(10, 20):
                        val = st.number_input(f"Feature {i}", value=0.5, min_value=0.0, max_value=1.0, step=0.01, key=f"feat_{i}")
                        features_10_19.append(val)
                
                with col_d:
                    features_20_29 = []
                    for i in range(20, 30):
                        val = st.number_input(f"Feature {i}", value=0.5, min_value=0.0, max_value=1.0, step=0.01, key=f"feat_{i}")
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
                st.markdown("### 🎯 Final Assessment")
                
                # Risk level indicator
                final_score = result['final_comparison_score']
                recommendation = result['recommendation']
                
                if recommendation == "approve":
                    st.markdown(f'<div class="safe-transaction"><h3>✅ APPROVED</h3><p>Final Risk Score: {final_score:.3f}</p></div>', unsafe_allow_html=True)
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
                
                # Bank comparison
                st.markdown("### 🏦 Individual Bank Analysis")
                st.plotly_chart(
                    create_bank_comparison_chart(result['individual_scores']),
                    use_container_width=True
                )
                
                # Detailed results
                with st.expander("📋 Detailed Results", expanded=False):
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
