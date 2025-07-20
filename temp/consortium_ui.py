"""
Consortium Fraud Detection UI
Interactive Streamlit interface for testing the privacy-preserving fraud detection consortium
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os
import sys

# Import the consortium classes
from consortium_comparison_score_prototype import BankSimulator, ConsortiumComparisonService

# Page configuration
st.set_page_config(
    page_title="Consortium Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .bank-header {
        color: #2e8b57;
        font-weight: bold;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    .consensus-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_consortium_models():
    """Load pre-trained consortium models"""
    banks = {}
    bank_ids = ['bank_A', 'bank_B', 'bank_C']
    
    for bank_id in bank_ids:
        bank = BankSimulator(bank_id, f'{bank_id}_data.csv')
        if bank.load_model():
            banks[bank_id] = bank
        else:
            st.error(f"Failed to load model for {bank_id}. Please run training first.")
            return None
    
    # Initialize consortium
    consortium = ConsortiumComparisonService()
    for bank_id, bank in banks.items():
        consortium.register_bank(bank_id, bank)
    
    return consortium, banks

def generate_sample_transaction():
    """Generate a sample transaction for testing"""
    np.random.seed(None)  # Use current time for randomness
    return np.random.random(30).tolist()  # 30 features

def create_risk_gauge(score, title):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 1], 'tickformat': '.2%'},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def format_risk_level(score):
    """Format risk level with color coding"""
    if score < 0.3:
        return f'<span class="risk-low">LOW ({score:.1%})</span>'
    elif score < 0.7:
        return f'<span class="risk-medium">MEDIUM ({score:.1%})</span>'
    else:
        return f'<span class="risk-high">HIGH ({score:.1%})</span>'

def main():
    # Title and description
    st.markdown('<h1 class="main-header">🏦 Consortium Fraud Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Privacy-preserving fraud detection through collaborative intelligence<br>
            <em>Score Sharing Consortium Architecture</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading consortium models..."):
        result = load_consortium_models()
        if result is None:
            st.stop()
        consortium, banks = result
    
    # Sidebar for transaction input
    st.sidebar.header("🔍 Transaction Input")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Generate Sample", "Manual Input", "Upload CSV"]
    )
    
    transaction_data = None
    
    if input_method == "Generate Sample":
        st.sidebar.markdown("### Sample Transaction Generator")
        fraud_likelihood = st.sidebar.selectbox(
            "Sample type:",
            ["Random", "Low Risk", "Medium Risk", "High Risk"]
        )
        
        if st.sidebar.button("🎲 Generate New Transaction", type="primary"):
            if fraud_likelihood == "Low Risk":
                np.random.seed(42)
                transaction_data = (np.random.random(30) * 0.5).tolist()
            elif fraud_likelihood == "Medium Risk":
                np.random.seed(123)
                base = np.random.random(30)
                base[0:5] += 0.3  # Boost some features
                transaction_data = np.clip(base, 0, 1).tolist()
            elif fraud_likelihood == "High Risk":
                np.random.seed(456)
                base = np.random.random(30)
                base[0:10] += 0.5  # Boost many features
                transaction_data = np.clip(base, 0, 1).tolist()
            else:
                transaction_data = generate_sample_transaction()
    
    elif input_method == "Manual Input":
        st.sidebar.markdown("### Manual Feature Input")
        st.sidebar.markdown("*Enter values between 0 and 1*")
        
        transaction_data = []
        with st.sidebar.expander("Transaction Features", expanded=False):
            cols = st.columns(2)
            for i in range(30):
                col_idx = i % 2
                with cols[col_idx]:
                    value = st.number_input(
                        f"Feature {i+1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        key=f"feature_{i}"
                    )
                    transaction_data.append(value)
    
    elif input_method == "Upload CSV":
        st.sidebar.markdown("### CSV Upload")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) >= 30:
                transaction_data = df.iloc[0, :30].tolist()
                st.sidebar.success(f"Loaded transaction with {len(transaction_data)} features")
            else:
                st.sidebar.error(f"CSV must have at least 30 columns. Found {len(df.columns)}")
    
    # Main content area
    if transaction_data is not None:
        # Process transaction
        with st.spinner("Processing transaction through consortium..."):
            result = consortium.generate_comparison_score(transaction_data)
        
        # Display results in columns
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🎯 Consortium Analysis Results")
            
            # Consensus score - main metric
            consensus_score = result['consensus_score']
            st.markdown(f"""
            <div class="metric-card">
                <div class="consensus-score" style="color: {'#dc3545' if consensus_score > 0.7 else '#ffc107' if consensus_score > 0.3 else '#28a745'}">
                    {consensus_score:.1%}
                </div>
                <div style="text-align: center; margin-top: 0.5rem;">
                    <strong>Consensus Risk Score</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Individual bank scores
        st.markdown("### 🏛️ Individual Bank Assessments")
        
        bank_cols = st.columns(3)
        bank_names = ['Bank A', 'Bank B', 'Bank C']
        bank_ids = ['bank_A', 'bank_B', 'bank_C']
        
        for idx, (col, bank_name, bank_id) in enumerate(zip(bank_cols, bank_names, bank_ids)):
            with col:
                if bank_id in result['individual_scores']:
                    score = result['individual_scores'][bank_id]
                    assessment = result['individual_assessments'][bank_id]
                    
                    st.markdown(f'<h4 class="bank-header">{bank_name}</h4>', unsafe_allow_html=True)
                    
                    # Gauge chart
                    fig = create_risk_gauge(score, f"{bank_name} Risk Score")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Details
                    st.markdown(f"""
                    **Risk Level:** {format_risk_level(score)}  
                    **Model Confidence:** {assessment['model_confidence']:.1%}  
                    **Risk Bucket:** {assessment['risk_bucket'].upper()}
                    """, unsafe_allow_html=True)
        
        # Consortium metrics
        st.markdown("### 📊 Consortium Intelligence")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Variance Score",
                f"{result['variance_score']:.3f}",
                help="Measure of agreement between banks"
            )
        
        with col2:
            st.metric(
                "Network Anomaly",
                f"{result['network_anomaly_score']:.1%}",
                help="Cross-institutional anomaly detection"
            )
        
        with col3:
            st.metric(
                "Confidence Level",
                result['confidence_level'].upper(),
                help="Consortium confidence in the assessment"
            )
        
        with col4:
            st.metric(
                "Flagging Banks",
                f"{result['flagging_banks_count']}/3",
                help="Number of banks flagging as high risk"
            )
        
        # Final recommendation
        recommendation = result['recommendation']
        final_score = result['final_comparison_score']
        
        rec_color = {
            'approve': '#28a745',
            'review': '#ffc107', 
            'block': '#dc3545'
        }
        
        st.markdown("### 🎯 Final Recommendation")
        st.markdown(f"""
        <div style="background-color: {rec_color.get(recommendation, '#6c757d')}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
            <h2 style="margin: 0; color: white;">{recommendation.upper()}</h2>
            <p style="margin: 0.5rem 0 0 0; color: white;">Final Comparison Score: {final_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed analysis
        with st.expander("🔍 Detailed Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Individual Bank Scores:**")
                for bank_id, score in result['individual_scores'].items():
                    st.write(f"• {bank_id}: {score:.3f}")
                
                st.markdown("**Flagging Banks:**")
                if result['flagging_banks']:
                    for bank in result['flagging_banks']:
                        st.write(f"• {bank}")
                else:
                    st.write("• None")
            
            with col2:
                st.markdown("**Score Breakdown:**")
                st.write(f"• Consensus Score: {result['consensus_score']:.3f}")
                st.write(f"• Variance Score: {result['variance_score']:.3f}")
                st.write(f"• Network Anomaly: {result['network_anomaly_score']:.3f}")
                st.write(f"• Final Score: {result['final_comparison_score']:.3f}")
                
                st.markdown("**Transaction Features (first 10):**")
                for i, value in enumerate(transaction_data[:10]):
                    st.write(f"• Feature {i+1}: {value:.3f}")
        
        # Visualization
        st.markdown("### 📈 Score Visualization")
        
        # Create comparison chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Individual Bank Scores", "Consortium Metrics"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Bank scores
        bank_scores = [result['individual_scores'][bid] for bid in bank_ids]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig.add_trace(
            go.Bar(
                x=bank_names,
                y=bank_scores,
                name="Risk Scores",
                marker_color=colors,
                text=[f"{score:.1%}" for score in bank_scores],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Consortium metrics
        metrics = ['Consensus', 'Variance', 'Anomaly', 'Final']
        metric_values = [
            result['consensus_score'],
            result['variance_score'],
            result['network_anomaly_score'],
            result['final_comparison_score']
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=metric_values,
                name="Consortium Metrics",
                marker_color='#9467bd',
                text=[f"{val:.2f}" for val in metric_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Consortium Fraud Detection Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to the Consortium Fraud Detection System
        
        This interactive demo showcases the **Score Sharing Consortium** architecture for privacy-preserving fraud detection.
        
        ### 🔧 How it works:
        1. **Local Training**: Each bank trains models on their proprietary data
        2. **Local Inference**: Banks score transactions using their local models  
        3. **Score Sharing**: Only risk scores (0-1) are shared with the consortium
        4. **Intelligent Aggregation**: Consortium generates consensus and comparison scores
        
        ### 🛡️ Privacy Features:
        - ✅ Raw data never leaves bank premises
        - ✅ Model weights remain proprietary
        - ✅ Only aggregated risk scores are shared
        - ✅ Network intelligence without data exposure
        
        ### 📊 Available Models:
        """)
        
        # Display model information
        if 'consortium' in locals():
            for bank_id in ['bank_A', 'bank_B', 'bank_C']:
                if bank_id in banks:
                    bank = banks[bank_id]
                    st.markdown(f"""
                    **{bank_id.upper()}**: XGBoost Classifier  
                    *Model Confidence: {bank.model_confidence:.1%}*
                    """)
        
        st.markdown("""
        ### 🚀 Get Started:
        Use the sidebar to generate a sample transaction or input your own data to see the consortium analysis in action!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🔒 Privacy-Preserving Fraud Detection Consortium | 
        <a href="https://github.com" style="color: #1f77b4;">Documentation</a> | 
        Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
