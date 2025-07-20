#!/bin/bash

# Consortium Fraud Detection UI Launcher
# Starts the Streamlit app with secure private network binding

echo "🛡️ Starting Consortium Fraud Detection Dashboard..."
echo "📍 Network: Private network only (192.168.4.100:8501)"
echo "🔒 Security: No internet exposure"
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if models exist
if [ ! -d "models" ]; then
    echo "❌ No trained models found. Running training first..."
    python consortium_comparison_score_prototype.py train
    echo "✅ Training complete!"
    echo ""
fi

# Start Streamlit with private network binding
echo "🚀 Starting Streamlit dashboard..."
streamlit run consortium_fraud_ui.py --server.address 192.168.4.100

echo "🛑 Dashboard stopped."
