# 🛡️ Consortium Privacy-Preserving Fraud Detection System

A privacy-preserving fraud detection system that enables banks to collaborate on fraud detection without exposing sensitive data or model weights. This implementation demonstrates the **Score Sharing Consortium** architecture with an interactive web interface.

## 🧠 Agent Context & External References

### Core Documentation
- `fraud_scenarios_explained.md` - Detailed explanations of 6 major cross-institutional fraud scenarios
- `consortium_privacy_preserving_fraud_detection.md` - Technical architecture and implementation details
- `agent_context.json` - Structured project context for AI agents

### External Context Sources
```
External Banking Regulations: C:/path/to/external/banking_regulations/
External Fraud Patterns: C:/path/to/external/fraud_intelligence/
External Compliance: C:/path/to/external/compliance_frameworks/
External Research: C:/path/to/external/research_papers/
```

### Key Fraud Scenarios (Maximum Consortium Value)
1. **High-Value Wire Fraud** - Bank A specialization in $50M+ corporate transfers
2. **Synthetic Identity Networks** - Bank B expertise in identity verification
3. **Money Mule Networks** - Bank C velocity and timing analysis
4. **Business Email Compromise** - Cross-institutional pattern recognition
5. **Cross-Border Laundering** - Multi-country coordination detection
6. **Cryptocurrency Laundering** - Traditional-to-crypto conversion tracking

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [Usage Guide](#6-usage-guide)
   - 6.1 [Training Models](#61-training-models)
   - 6.2 [Running Inference](#62-running-inference)
   - 6.3 [Web Dashboard](#63-web-dashboard)
7. [Network Configuration](#7-network-configuration)
   - 7.1 [Private Network Only](#71-private-network-only)
   - 7.2 [Public Access](#72-public-access)
8. [API Reference](#8-api-reference)
9. [File Structure](#9-file-structure)
10. [Configuration](#10-configuration)
11. [Security Features](#11-security-features)
12. [Troubleshooting](#12-troubleshooting)
13. [Contributing](#13-contributing)
14. [License](#14-license)

---

## 1. Overview

This system implements a **Score Sharing Consortium** for fraud detection, where:

- 🏦 **Banks train models locally** using their proprietary data
- 🔒 **Model weights never leave** bank premises
- 📊 **Only risk scores are shared** for comparison
- 🤝 **Consortium aggregates scores** for enhanced detection
- 🛡️ **Complete privacy preservation** with collaborative intelligence

### Key Benefits

- ✅ **Maximum Privacy**: Raw data and models stay local
- ✅ **Network Intelligence**: Cross-bank anomaly detection
- ✅ **Regulatory Compliance**: Simplified legal framework
- ✅ **Minimal Governance**: Focus on score semantics only

---

## 2. Architecture

### Score Sharing Consortium Flow

```
Bank A: Local Training → Local Model → Local Inference → Risk Score → Consortium TEE
Bank B: Local Training → Local Model → Local Inference → Risk Score → Consortium TEE  
Bank C: Local Training → Local Model → Local Inference → Risk Score → Consortium TEE

Consortium TEE: Score Aggregation → Comparison Score → Consensus Alert → Banks
```

### Key Components

- **BankSimulator**: Simulates individual bank fraud detection models
- **ConsortiumComparisonService**: Aggregates scores and provides consensus analysis
- **Streamlit Dashboard**: Interactive web interface for transaction analysis

---

## 3. Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB RAM
- **Storage**: 500MB free space

### Required Python Packages

All dependencies are listed in `requirements.txt`:

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
streamlit>=1.47.0
plotly>=5.0.0
```

---

## 4. Installation

### 4.1 Clone or Download

Ensure you have all files in your working directory:

```
consortium_comparison_score_prototype.py
consortium_fraud_ui.py
consortium_privacy_preserving_fraud_detection.md
requirements.txt
start_dashboard.sh
.streamlit/config.toml
```

### 4.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 4.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.4 Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit, plotly; print('✅ All dependencies installed successfully')"
```

---

## 5. Quick Start

### 5.1 One-Command Start

```bash
# Make launch script executable (Linux/macOS only)
chmod +x start_dashboard.sh

# Start the complete system
./start_dashboard.sh
```

### 5.2 Manual Start

```bash
# Activate environment
source venv/bin/activate

# Train models (first time only)
python consortium_comparison_score_prototype.py train

# Start web dashboard
streamlit run consortium_fraud_ui.py
```

### 5.3 Access the Application

- **Local Access**: http://localhost:8501
- **Network Access**: http://192.168.4.100:8501 (private network only)

---

## 6. Usage Guide

### 6.1 Training Models

Train fraud detection models for all consortium banks:

```bash
# Train all models
python consortium_comparison_score_prototype.py train

# Check trained models
python consortium_comparison_score_prototype.py list
```

**Output Example:**
```
Consortium Model Training Mode
==============================

1. Creating synthetic bank data with diverse perspectives...
bank_A: 1000 transactions, 12.4% fraud rate
bank_B: 1000 transactions, 12.0% fraud rate
bank_C: 1000 transactions, 13.5% fraud rate

3. Training bank models on diverse data...
bank_A (xgboost): Training accuracy: 0.895
bank_B (xgboost): Training accuracy: 0.880
bank_C (xgboost): Training accuracy: 0.915

Training complete! Models saved and ready for inference.
```

### 6.2 Running Inference

Perform consortium fraud analysis using pre-trained models:

```bash
# Run inference only (requires trained models)
python consortium_comparison_score_prototype.py inference

# Run full demo (training + inference)
python consortium_comparison_score_prototype.py full
```

### 6.3 Web Dashboard

#### 6.3.1 Starting the Dashboard

```bash
# Start with private network binding (recommended)
streamlit run consortium_fraud_ui.py --server.address 192.168.4.100

# Or use the launch script
./start_dashboard.sh
```

#### 6.3.2 Dashboard Features

**🔍 Transaction Input Methods:**
- **Manual Input**: Enter all 30 transaction features manually
- **Sample Transactions**: Choose from pre-defined risk levels
- **Random Generation**: Generate transactions with specific characteristics

**📊 Analysis Results:**
- **Risk Gauges**: Visual risk score indicators
- **Bank Comparison**: Individual bank assessments
- **Consensus Analysis**: Aggregated consortium intelligence
- **Recommendation**: Approve/Review/Block decisions

**🏦 System Information:**
- **Bank Status**: Model confidence scores
- **Network Metrics**: Consensus and variance analysis
- **Privacy Indicators**: Data protection assurance

---

## 7. Network Configuration

### 7.1 Private Network Only (Recommended)

For maximum security, bind only to private network interface:

```bash
# Start with private binding
streamlit run consortium_fraud_ui.py --server.address 192.168.4.100
```

**Access URLs:**
- ✅ **Private Network**: http://192.168.4.100:8501
- ✅ **VS Code Tunnel**: http://localhost:8501
- ❌ **Internet**: Blocked

### 7.2 Public Access

⚠️ **Not recommended for production**

```bash
# Start with public binding (development only)
streamlit run consortium_fraud_ui.py --server.address 0.0.0.0
```

**Access URLs:**
- ✅ **Local**: http://localhost:8501
- ✅ **Network**: http://192.168.4.100:8501
- ⚠️ **Internet**: http://[PUBLIC_IP]:8501

---

## 8. API Reference

### 8.1 Command Line Interface

```bash
# Available commands
python consortium_comparison_score_prototype.py [mode]

# Modes:
train       # Train and save models only
inference   # Use pre-trained models for inference only
list        # List all saved models and information
full        # Complete demo with training and inference (default)
```

### 8.2 Core Classes

#### BankSimulator

```python
from consortium_comparison_score_prototype import BankSimulator

# Initialize bank
bank = BankSimulator(bank_id="bank_A", data_path="bank_A_data.csv")

# Train model
accuracy = bank.train_local_model(model_type='xgboost')

# Load existing model
bank.load_model()

# Predict risk score
result = bank.predict_risk_score(transaction_features)
```

#### ConsortiumComparisonService

```python
from consortium_comparison_score_prototype import ConsortiumComparisonService

# Initialize consortium
consortium = ConsortiumComparisonService()

# Register banks
consortium.register_bank("bank_A", bank_A)
consortium.register_bank("bank_B", bank_B)

# Analyze transaction
result = consortium.generate_comparison_score(transaction_features)
```

### 8.3 Result Format

```python
{
    'individual_scores': {'bank_A': 0.123, 'bank_B': 0.456, 'bank_C': 0.789},
    'consensus_score': 0.456,
    'variance_score': 0.123,
    'network_anomaly_score': 0.600,
    'final_comparison_score': 0.543,
    'confidence_level': 'high',
    'recommendation': 'review',
    'flagging_banks': ['bank_C'],
    'participating_banks': 3
}
```

---

## 9. File Structure

```
consortium-fraud-detection/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── consortium_comparison_score_prototype.py     # Core fraud detection logic
├── consortium_fraud_ui.py                      # Streamlit web interface
├── consortium_privacy_preserving_fraud_detection.md  # Architecture documentation
├── start_dashboard.sh                          # Launch script
├── .streamlit/
│   └── config.toml                             # Streamlit configuration
├── models/                                     # Trained model storage
│   ├── bank_A_model.pkl
│   ├── bank_A_metadata.json
│   ├── bank_B_model.pkl
│   ├── bank_B_metadata.json
│   ├── bank_C_model.pkl
│   └── bank_C_metadata.json
├── venv/                                       # Virtual environment
└── [bank_data].csv                            # Synthetic bank datasets
```

---

## 10. Configuration

### 10.1 Streamlit Configuration

Edit `.streamlit/config.toml`:

```toml
[server]
address = "192.168.4.100"  # Private network binding
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 10.2 Model Configuration

Models are automatically configured with:
- **Algorithm**: XGBoost (optimized for fraud detection)
- **Features**: 30 synthetic transaction features
- **Training**: 80/20 train-test split
- **Validation**: Stratified sampling for class balance

### 10.3 Environment Variables

```bash
# Optional: Set custom paths
export CONSORTIUM_MODEL_PATH="./models"
export CONSORTIUM_DATA_PATH="./data"
```

---

## 11. Security Features

### 11.1 Privacy Protection

- 🔒 **Data Isolation**: Bank data never leaves premises
- 🛡️ **Model Security**: Weights remain proprietary
- 📊 **Score-Only Sharing**: Minimal information exposure
- 🔐 **Network Binding**: Private network access only

### 11.2 Security Best Practices

1. **Network Security**:
   ```bash
   # Bind to private network only
   streamlit run consortium_fraud_ui.py --server.address 192.168.4.100
   ```

2. **File Permissions**:
   ```bash
   # Secure model files
   chmod 600 models/*.pkl
   chmod 600 models/*.json
   ```

3. **Virtual Environment**:
   ```bash
   # Use isolated Python environment
   source venv/bin/activate
   ```

### 11.3 Audit Trail

Models and transactions are logged with:
- **Timestamps**: All training and inference events
- **Model Metadata**: Training accuracy and confidence
- **Transaction History**: Analysis results and decisions

---

## 12. Troubleshooting

### 12.1 Common Issues

#### Missing Dependencies
```bash
# Error: ModuleNotFoundError
pip install -r requirements.txt
```

#### No Trained Models
```bash
# Error: No trained models found
python consortium_comparison_score_prototype.py train
```

#### Port Already in Use
```bash
# Error: Address already in use
pkill -f streamlit
# Then restart the application
```

#### Permission Denied
```bash
# Error: Permission denied for start_dashboard.sh
chmod +x start_dashboard.sh
```

### 12.2 Debugging

#### Enable Verbose Logging
```bash
# Set debug mode
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run consortium_fraud_ui.py
```

#### Check Network Binding
```bash
# Verify port binding
netstat -tlnp | grep 8501
```

#### Validate Models
```bash
# Check model status
python consortium_comparison_score_prototype.py list
```

### 12.3 Performance Issues

#### Memory Usage
- **Minimum**: 2GB RAM recommended
- **Models**: ~200KB per bank model
- **Data**: ~1MB per 1000 transactions

#### Startup Time
- **First Run**: ~30 seconds (includes training)
- **Subsequent**: ~5 seconds (loads existing models)

---

## 13. Contributing

### 13.1 Development Setup

```bash
# Clone repository
git clone [repository-url]
cd consortium-fraud-detection

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 pytest jupyter
```

### 13.2 Code Style

```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Run tests
pytest tests/
```

### 13.3 Adding Features

1. **Model Improvements**: Modify `BankSimulator` class
2. **UI Enhancements**: Update `consortium_fraud_ui.py`
3. **New Architectures**: Extend `ConsortiumComparisonService`

---

## 14. License

This project demonstrates privacy-preserving fraud detection concepts for educational and research purposes.

### 14.1 Usage Rights

- ✅ **Educational Use**: Free for learning and research
- ✅ **Academic Use**: Citation required
- ⚠️ **Commercial Use**: Contact authors for licensing

### 14.2 Disclaimer

This is a prototype implementation for demonstration purposes. For production use:

- Implement proper authentication and authorization
- Add comprehensive logging and monitoring
- Ensure compliance with financial regulations
- Conduct thorough security audits

---

## 📞 Support

For questions, issues, or contributions:

1. **Documentation**: Read the architecture guide in `consortium_privacy_preserving_fraud_detection.md`
2. **Issues**: Check the troubleshooting section above
3. **Features**: Review the API reference for customization options

---

## 🚀 Quick Reference

### Essential Commands
```bash
# Complete setup and start
./start_dashboard.sh

# Train models only
python consortium_comparison_score_prototype.py train

# Start dashboard (private network)
streamlit run consortium_fraud_ui.py --server.address 192.168.4.100

# Check system status
python consortium_comparison_score_prototype.py list
```

### Access URLs
- **Dashboard**: http://192.168.4.100:8501
- **VS Code**: http://localhost:8501

---

*Last updated: July 20, 2025*
