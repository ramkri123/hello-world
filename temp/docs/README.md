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

### Distributed HTTP-Based Consortium Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONSORTIUM HUB                               │
│                   (HTTP API Server)                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Flask HTTP Gateway (Port 8080)                             ││
│  │  • Participant registration & management                    ││
│  │  • Inference distribution & score collection               ││
│  │  • Consensus vs. divergence analysis                       ││
│  │  • Real-time session management                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                    ▲
                        HTTP Connections (Outbound Only)
                                    │
    ┌───────────────────┐          │          ┌───────────────────┐
    │  BANK A PROCESS   │──────────┼──────────│  BANK B PROCESS   │
    │ (HTTP Client Only)│          │          │ (HTTP Client Only)│
    │                   │          │          │                   │
    │ Wire Transfer     │          │          │ Identity          │
    │ Specialist        │          │          │ Verification      │
    │                   │          │          │ Expert            │
    │ • Local Model     │          │          │ • Local Model     │
    │ • HTTP Client     │          │          │ • HTTP Client     │
    │ • Specialized     │          │          │ • Specialized     │
    │   Logging         │          │          │   Logging         │
    └───────────────────┘          │          └───────────────────┘
                                   │
                        ┌───────────────────┐
                        │  BANK C PROCESS   │
                        │ (HTTP Client Only)│
                        │                   │
                        │ Network Pattern   │
                        │ Analyst           │
                        │                   │
                        │ • Local Model     │
                        │ • HTTP Client     │
                        │ • Specialized     │
                        │   Logging         │
                        └───────────────────┘
```

### Key Components

- **Consortium Hub** (`consortium_hub.py`): Central HTTP API server for participant coordination
- **Generic Bank Process** (`generic_bank_process.py`): **Primary implementation** - configurable bank process that can run as any bank
- **Bank Specializations** (`specializations/`): Optional custom business logic modules per bank
- **Distributed UI** (`distributed_consortium_ui.py`): Streamlit interface connecting via HTTP
- **Participant Node** (`participant_node.py`): Base class for bank process HTTP client functionality

### 🎯 **Primary Architecture (Recommended)**

**Generic Process Approach:**
```bash
python generic_bank_process.py --bank-id bank_A  # Wire Transfer Specialist
python generic_bank_process.py --bank-id bank_B  # Identity Verification Expert
python generic_bank_process.py --bank-id bank_C  # Network Pattern Analyst
```

### 🏗️ **Alternative Architecture (Legacy)**

For reference, the system also includes dedicated bank process files that demonstrate how each bank might implement their own specialized codebase in a real-world scenario:

- **Bank A Process** (`bank_A_process.py`): Wire transfer fraud specialist (dedicated implementation)
- **Bank B Process** (`bank_B_process.py`): Identity verification expert (dedicated implementation)  
- **Bank C Process** (`bank_C_process.py`): Network pattern analyst (dedicated implementation)

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
# 🏗️ DISTRIBUTED SYSTEM (Primary Implementation)
consortium_hub.py
participant_node.py  
generic_bank_process.py
distributed_consortium_ui.py
start_distributed_consortium.py
start_banks_separately.py
specializations/

# 🏛️ LEGACY COMPONENTS (For Reference/Alternative)
bank_A_process.py
bank_B_process.py
bank_C_process.py
consortium_comparison_score_prototype.py
consortium_fraud_ui.py
consortium_ui.py

# 📁 CONFIGURATION & DATA
requirements.txt
consortium_architecture.md
models/
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

---

## 5. Quick Start

### 5.1 Automated Distributed Launch

```bash
# Start all components of the distributed system
python start_distributed_consortium.py

# This starts:
# - Consortium Hub (port 8080)
# - Bank A Process (generic_bank_process.py --bank-id bank_A) 
# - Bank B Process (generic_bank_process.py --bank-id bank_B)
# - Bank C Process (generic_bank_process.py --bank-id bank_C)
# - Distributed UI (port 8501)
```

### 5.2 Manual Component Start

```bash
# 1. Start consortium hub first
python consortium_hub.py

# 2. Start individual bank processes (separate terminals)
python generic_bank_process.py --bank-id bank_A
python generic_bank_process.py --bank-id bank_B  
python generic_bank_process.py --bank-id bank_C

# 3. Start distributed UI
streamlit run distributed_consortium_ui.py
```

### 5.3 Individual Bank Launch

```bash
# Launch banks individually with custom launcher
python start_banks_separately.py --bank A  # Start only Bank A
python start_banks_separately.py --bank all  # Start all banks
```

### 5.4 Access the Application

- **Distributed UI**: http://localhost:8501
- **Consortium Hub API**: http://localhost:8080

### 5.5 Architecture Note

**🔒 Zero-Trust Security Model:**
- Banks run as **HTTP clients only** - no inbound ports
- All connections are **outbound-only** from bank premises
- Consortium hub receives connections but banks never expose services
- Complete **firewall-friendly** architecture

---

## 6. Usage Guide

### 6.1 Distributed System Operation

The system operates as **separate Python processes** for each bank:

```bash
# Check consortium hub health
curl http://localhost:8080/health

# View registered participants
curl http://localhost:8080/health | python -m json.tool
```

**Expected Output:**
```json
{
  "status": "healthy",
  "participants": 3,
  "active_sessions": 0,
  "timestamp": "2025-07-20T16:33:14.160975"
}
```

### 6.2 Bank Process Management

Each bank runs as an **independent Python process** with specialized logging:

```bash
# Bank A: Wire Transfer Specialist
[BANK_A] 2025-07-20 16:33:05,123 - INFO - 🏦 Starting Bank A - Wire Transfer Specialist
[BANK_A] 2025-07-20 16:33:05,124 - INFO - ✅ Successfully registered with consortium

# Bank B: Identity Verification Expert  
[BANK_B] 2025-07-20 16:33:06,125 - INFO - 🔍 Starting Bank B - Identity Verification Expert
[BANK_B] 2025-07-20 16:33:06,126 - INFO - ✅ Successfully registered with consortium

# Bank C: Network Pattern Analyst
[BANK_C] 2025-07-20 16:33:07,127 - INFO - 🌐 Starting Bank C - Network Pattern Analyst  
[BANK_C] 2025-07-20 16:33:07,128 - INFO - ✅ Successfully registered with consortium
```

### 6.3 Distributed Web Dashboard

#### 6.3.1 Starting the Dashboard

```bash
# Start distributed UI (connects to consortium hub via HTTP)
streamlit run distributed_consortium_ui.py

# Access distributed interface
open http://localhost:8501
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

### 8.1 Consortium Hub HTTP API

The consortium hub exposes a REST API for participant management and inference coordination:

```bash
# Base URL
CONSORTIUM_HUB = "http://localhost:8080"
```

#### 8.1.1 Health Check
```http
GET /health
Content-Type: application/json

Response:
{
  "status": "healthy",
  "participants": 3,
  "active_sessions": 0,
  "timestamp": "2025-07-20T16:33:14.160975"
}
```

#### 8.1.2 Participant Registration
```http
POST /register
Content-Type: application/json

Request:
{
  "node_id": "bank_A",
  "specialty": "wire_transfer_specialist", 
  "endpoint": "http://localhost:8081",
  "geolocation": "US-East"
}

Response:
{
  "status": "registered",
  "session_token": "abc123...",
  "participants": 3
}
```

#### 8.1.3 Inference Request
```http
POST /inference
Content-Type: application/json

Request:
{
  "use_case": "wire_transfer_fraud",
  "features": [0.35, 0.45, 0.75, ...]
}

Response:
{
  "session_id": "sess_123",
  "status": "distributed",
  "participants_notified": 3
}
```

#### 8.1.4 Score Submission
```http
POST /score
Content-Type: application/json

Request:
{
  "session_id": "sess_123",
  "participant_id": "bank_A",
  "risk_score": 0.85,
  "confidence": 0.95
}

Response:
{
  "status": "recorded",
  "session_status": "collecting"
}
```

#### 8.1.5 Results Retrieval
```http
GET /results/{session_id}
Content-Type: application/json

Response:
{
  "session_id": "sess_123",
  "status": "completed",
  "comparison_score": 0.823,
  "consensus": "STRONG_CONSENSUS",
  "scores": {
    "bank_A": {"risk_score": 0.85, "confidence": 0.95},
    "bank_B": {"risk_score": 0.82, "confidence": 0.90},
    "bank_C": {"risk_score": 0.81, "confidence": 0.88}
  }
}
```

### 8.2 Bank Process Management

#### 8.2.1 Generic Bank Process

```bash
# Generic bank process (Primary approach)
python generic_bank_process.py --bank-id bank_A --consortium-url http://localhost:8080
python generic_bank_process.py --bank-id bank_B --consortium-url http://localhost:8080
python generic_bank_process.py --bank-id bank_C --consortium-url http://localhost:8080

# Alternative: Legacy dedicated bank scripts
python bank_A_process.py --consortium-url http://localhost:8080
python bank_B_process.py --consortium-url http://localhost:8080
python bank_C_process.py --consortium-url http://localhost:8080
```

#### 8.2.2 ParticipantNode Configuration

```python
from participant_node import ParticipantNode, NodeConfig

# Configure bank node using generic process
config = NodeConfig(
    node_id="bank_A",
    specialty="wire_transfer_specialist",
    consortium_url="http://localhost:8080",
    model_path="models/bank_A_model.pkl",
    geolocation="US-East"
)

# Create and start node
node = ParticipantNode(config)
if node.register_with_consortium():
    node.start_polling()
```

#### 8.2.3 Bank Specialization

```python
# Optional: Create custom specialization module
# File: specializations/bank_A_specialization.py

def customize_node(node):
    """Add custom business logic to the bank node"""
    original_inference = node.process_inference
    
    def enhanced_inference(features):
        # Custom wire transfer analysis
        result = original_inference(features)
        # Apply bank-specific enhancements
        return result
    
    node.process_inference = enhanced_inference
```

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
├── README.md                                    # This documentation
├── requirements.txt                             # Python dependencies
├── consortium_architecture.md                  # Distributed architecture specification
│
├── 🏗️ DISTRIBUTED SYSTEM COMPONENTS (Primary)
├── consortium_hub.py                           # Central HTTP API server (port 8080)
├── participant_node.py                         # Base class for bank HTTP clients
├── generic_bank_process.py                     # **Primary** - Configurable bank process
├── distributed_consortium_ui.py                # Streamlit UI connecting via HTTP (port 8501)
├── specializations/                            # Optional bank-specific customizations
│   └── bank_A_specialization.py               # Example: Wire transfer specialization
│
├── 🚀 SYSTEM LAUNCHERS
├── start_distributed_consortium.py             # Automated full system launcher
├── start_banks_separately.py                   # Manual individual bank launcher
│
├── 🏛️ LEGACY COMPONENTS (Alternative Implementation)
├── bank_A_process.py                           # Bank A - Dedicated implementation
├── bank_B_process.py                           # Bank B - Dedicated implementation
├── bank_C_process.py                           # Bank C - Dedicated implementation
├── consortium_comparison_score_prototype.py     # Original single-process implementation
├── consortium_fraud_ui.py                      # Original single-process UI
├── consortium_ui.py                            # Alternative single-process UI
│
├── 📁 DATA & MODELS
├── models/                                     # Trained model storage
│   ├── bank_A_model.pkl                       # Bank A specialized model
│   ├── bank_A_metadata.json                   # Bank A model metadata
│   ├── bank_B_model.pkl                       # Bank B specialized model  
│   ├── bank_B_metadata.json                   # Bank B model metadata
│   ├── bank_C_model.pkl                       # Bank C specialized model
│   ├── bank_C_metadata.json                   # Bank C model metadata
│   ├── test_bank_model.pkl                    # Test bank model
│   └── test_bank_metadata.json                # Test bank metadata
├── bank_A_data.csv                            # Bank A synthetic dataset
├── bank_B_data.csv                            # Bank B synthetic dataset
├── bank_C_data.csv                            # Bank C synthetic dataset
└── test_bank_data.csv                         # Test bank dataset
```

### 🎯 **Key Architecture Files**

- **`consortium_hub.py`** - Central coordinator running Flask HTTP API
- **`generic_bank_process.py`** - **Primary** configurable bank process (replaces separate bank files)
- **`participant_node.py`** - HTTP client base class for bank connectivity
- **`specializations/`** - Optional bank-specific business logic modules
- **`distributed_consortium_ui.py`** - UI connecting to hub via HTTP requests
- **`start_distributed_consortium.py`** - One-command launcher for full system

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

1. **Bank Process Enhancements**: Modify individual `bank_*_process.py` files for specialized behavior
2. **Consortium Hub Extensions**: Update `consortium_hub.py` for new API endpoints or aggregation logic
3. **UI Improvements**: Enhance `distributed_consortium_ui.py` for better visualization
4. **Participant Node Features**: Extend `participant_node.py` base class for new capabilities
5. **System Orchestration**: Improve `start_distributed_consortium.py` for better process management

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

### Essential Commands - Distributed System
```bash
# 🚀 Start complete distributed system
python start_distributed_consortium.py

# 🏗️ Manual component startup
python consortium_hub.py                         # Start hub first
python generic_bank_process.py --bank-id bank_A  # Start Bank A
python generic_bank_process.py --bank-id bank_B  # Start Bank B  
python generic_bank_process.py --bank-id bank_C  # Start Bank C
streamlit run distributed_consortium_ui.py       # Start UI

# 🔧 Individual bank management
python start_banks_separately.py --bank A   # Start specific bank
python start_banks_separately.py --bank all # Start all banks

# 🔍 System health checks
curl http://localhost:8080/health            # Check consortium hub
curl http://localhost:8080/health | python -m json.tool  # Pretty JSON
```

### Access URLs - Distributed Architecture
- **Distributed UI**: http://localhost:8501
- **Consortium Hub**: http://localhost:8080

### Process Architecture - Zero-Trust Model
```
✅ Each bank runs as separate Python process
✅ HTTP client connections only (outbound-only)
✅ No inbound ports on bank infrastructure
✅ Consortium hub coordinates all participants
✅ Specialized logging per bank process
✅ Firewall-friendly architecture
```

---

*Last updated: July 20, 2025 - Distributed Process Architecture*
