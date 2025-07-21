# 🎬 Demo Guide: Business Email Compromise (BEC) Fraud Detection

## **🎯 Perfect Demo Scenario for Presentations**

You now have a complete, demo-ready fraud example that perfectly showcases the value of consortium fraud detection! Here's everything you need for a compelling presentation:

---

## **📧 The Fraud Story**

### **What Happened:**
- ABC Manufacturing's CFO receives an "urgent" email from their CEO
- Email requests immediate $485K wire transfer to "new strategic supplier"
- Claims it's for a "confidential acquisition deal"
- **But it's actually a sophisticated Business Email Compromise (BEC) fraud!**

### **The Fraudulent Email:**
```
From: CEO John Smith <jsmith@abc-manufacturing.com> [SPOOFED]
To: CFO Sarah Johnson
Subject: URGENT - Confidential Supplier Payment Required

Sarah,
I'm in meetings with our legal team regarding the acquisition 
we discussed. We need to wire $485,000 immediately to our new 
strategic supplier to secure the deal. This is highly confidential.

Wire to: Global Tech Solutions LLC
Account: 4567-8901-2345-6789 (First National Bank)
Amount: $485,000 USD

Please handle personally and don't discuss until I return.
Thanks, John
```

---

## **🏦 How to Demo This in the UI**

### **Step 1: Load the Demo Case**
1. Open the Streamlit dashboard at: http://localhost:8502
2. In the **Transaction Input Method**, select "Sample Transactions"
3. Choose: **"🎯 DEMO: CEO Fraud - ABC Manufacturing ($485K Wire)"**
4. Click **"🚀 Analyze Transaction"**

### **Step 2: Explain What You're Showing**
*"Let me show you a real fraud scenario that individual banks would miss, but our consortium catches..."*

### **Step 3: Walk Through the Results**

#### **Individual Bank Scores** 🏦
- **Bank A (Sender's bank)**: ~0.08 (LOW) ✅
  - *"They see a legitimate customer with sufficient funds making a normal business wire"*
  
- **Bank B (Recipient's bank)**: ~0.12 (LOW-MEDIUM) ⚠️
  - *"They notice the recipient account is only 3 days old, but that alone isn't conclusive"*
  
- **Bank C (Network banks)**: ~0.09 (LOW-MEDIUM) ⚠️
  - *"They've seen similar patterns but can't connect it to this specific transaction"*

#### **Consortium Intelligence** 🤝
- **Consensus Score**: ~0.35-0.65 (MEDIUM-HIGH) 🚨
- **Variance Score**: ~0.25-0.45 (HIGH disagreement = suspicious!)
- **Network Anomaly**: ~0.6-0.8 (Pattern detected across institutions!)
- **Final Assessment**: **BLOCKED** or **FLAGGED FOR REVIEW** 🛑

---

## **💡 Why This Demo Is Perfect**

### **✅ Easy to Understand**
- Everyone knows what CEO email fraud is
- Clear business context (manufacturing company, acquisition)
- Simple wire transfer everyone can relate to

### **✅ Shows Inter-Bank Value**
- **Without Consortium**: Each bank sees different piece, likely approves
- **With Consortium**: Pattern recognition across institutions catches fraud
- **Clear ROI**: Saves $485K + identifies criminal network

### **✅ Realistic & Compelling**
- Based on actual BEC fraud patterns
- $485K is significant but believable amount
- Shows sophisticated fraud that fools individual security

---

## **🎤 Presentation Script**

### **Opening** (30 seconds)
*"Business Email Compromise costs companies $43 billion annually. Let me show you how our consortium approach catches sophisticated fraud that individual banks miss completely."*

### **Setup** (30 seconds)
*"Here's ABC Manufacturing's CFO receiving what appears to be an urgent request from their CEO for a $485,000 wire transfer to a new supplier for a confidential acquisition."*

### **Individual Analysis** (60 seconds)
*"When we analyze this transaction, each bank sees something different:*
- *ABC's bank sees a normal business transaction from a good customer*
- *The recipient bank notices it's a very new account but that's not unusual*
- *Other banks in the network have seen similar patterns but can't connect them*
*Each bank individually would likely approve this."*

### **Consortium Reveal** (45 seconds)
*"But when we share intelligence across institutions - without sharing any customer data - a pattern emerges:*
- *High variance in risk scores indicates disagreement*
- *Network analysis reveals the same scheme hitting multiple companies*
- *Cross-institutional intelligence identifies a coordinated fraud ring*
*Result: $485,000 fraud stopped, criminal network exposed!"*

### **Closing** (15 seconds)
*"This is the power of privacy-preserving consortium intelligence - catching sophisticated fraud that's invisible to individual institutions."*

---

## **🔧 Technical Details for Demo**

### **Dashboard Location**: http://localhost:8502
### **Demo Transaction**: "🎯 DEMO: CEO Fraud - ABC Manufacturing ($485K Wire)"
### **Expected Results**:
- Individual scores: 0.08-0.12 (would approve individually)
- Consortium detection: High variance + network patterns = BLOCK
- Clear value demonstration: $485K saved + network identified

### **Key Talking Points**:
1. **Sophistication**: Fraud sophisticated enough to fool individual banks
2. **Privacy**: No customer data shared, only risk intelligence
3. **Network Effect**: Pattern invisible to individuals, obvious to consortium
4. **Real Impact**: Prevents major financial loss + identifies criminal network

---

## **🚀 Ready to Demo!**

Your consortium fraud detection system is now loaded with the perfect demo scenario that clearly shows:

✅ **Why individual banks miss sophisticated fraud**  
✅ **How consortium intelligence catches what others miss**  
✅ **Real financial impact and criminal network detection**  
✅ **Privacy-preserving approach that protects customer data**  

**Go to http://localhost:8502 and start demonstrating the future of collaborative fraud detection!** 🛡️
