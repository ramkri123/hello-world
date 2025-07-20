# 🎯 Demo Fraud Transaction - Business Email Compromise (BEC)

## **Perfect for UI Demo - Easy to Explain & Shows Inter-Bank Value**

### **📧 The Fraud Email**
```
From: CEO John Smith <jsmith@abc-manufacturing.com> [SPOOFED EMAIL]
To: CFO Sarah Johnson <sjohnson@abc-manufacturing.com>
Subject: URGENT - Confidential Supplier Payment Required

Sarah,

I'm in meetings with our legal team regarding the acquisition we discussed. 
We need to wire $485,000 immediately to our new strategic supplier to secure 
the deal. This is highly confidential - please process ASAP.

Wire Details:
- Amount: $485,000 USD
- Recipient: Global Tech Solutions LLC
- Account: 4567-8901-2345-6789 (First National Bank)
- Routing: 021000021
- Purpose: Strategic acquisition deposit

Please handle this personally and don't discuss with anyone until I return.

Thanks,
John
```

### **🏦 What Each Bank Sees Individually**

#### **Bank A (ABC Manufacturing's Bank)**
```
✅ Customer Analysis: ABC Manufacturing (customer since 2018)
✅ Account Balance: $2.3M (sufficient funds)
✅ Amount: $485K (within normal business range)
✅ Purpose: "Strategic acquisition" (sounds legitimate)
✅ Authorization: From CEO's usual device
📊 Risk Assessment: LOW RISK ✅
```

#### **Bank B (Recipient Bank - First National)**
```
🚨 Account Analysis: Global Tech Solutions LLC
⚠️ Account Age: 3 days old (just opened!)
⚠️ Business Type: Generic "tech solutions"
⚠️ Documentation: Minimal business verification
🚨 First Large Incoming Wire: $485K
📊 Risk Assessment: SUSPICIOUS ⚠️
```

#### **Bank C (Other Banks in Network)**
```
🚨 Pattern Recognition: Same scheme detected
🚨 Similar Targets: 5 other companies this week
🚨 Identical Language: "confidential acquisition"
🚨 Same Recipient Pattern: New LLC accounts
🚨 Timing: All on Friday afternoons
📊 Risk Assessment: CONFIRMED FRAUD 🚨
```

### **🤝 Consortium Intelligence**
```
Individual Scores:
- Bank A: 0.01 (very low - legitimate customer)
- Bank B: 0.09 (moderate - suspicious recipient)  
- Bank C: 0.06 (low - but recognizes pattern)

Consortium Analysis:
- Variance: High (different banks see different risks)
- Network Anomaly: Detected across institutions
- Final Assessment: BLOCKED 🛑
```

### **💡 Why This is Perfect for Demo**

#### **Easy to Explain** ✅
- Everyone understands email fraud
- Clear business context (CEO/CFO roles)
- Simple wire transfer scenario
- Obvious red flags when pointed out

#### **Shows Inter-Bank Value** ✅
- **Bank A alone**: Would approve (legitimate customer)
- **Bank B alone**: Might question but unsure
- **Bank C alone**: Sees pattern but not this specific case
- **Consortium together**: Catches the fraud! 🎯

#### **Visual Demo Elements** ✅
- Show the fraudulent email
- Display each bank's individual analysis
- Reveal the consortium insight
- Demonstrate pattern recognition across institutions

### **🎬 Demo Script**

1. **Setup**: "Let me show you a real fraud scenario..."
2. **Show Email**: "CFO receives this urgent request from CEO"
3. **Bank A View**: "ABC's bank sees normal business transaction"
4. **Bank B View**: "Recipient bank notices brand new account"
5. **Bank C View**: "Other banks recognize same pattern elsewhere"
6. **Consortium Result**: "Together they catch what none could alone!"
7. **Impact**: "Saved $485K and identified criminal network"

### **🔍 Technical Details for System**

```python
# Transaction Features (normalized 0-1 scale)
demo_features = [
    0.75,  # Amount: $485K (high but business-reasonable)
    0.85,  # Urgency: Friday 4:47 PM rush request
    0.25,  # Sender Risk: Low (legitimate business)
    0.95,  # Receiver Risk: Very high (3-day-old account)
    0.90,  # Email Fraud: Spoofed CEO communication
    0.85,  # Network Pattern: Same scheme at other banks
    0.95,  # Urgency Indicators: "URGENT", "confidential"
    0.70,  # Timing: End of business day
    # ... (22 more features)
]
```

### **📈 Expected Consortium Results**

- **Consensus Score**: 0.3-0.6 (moderate agreement)
- **Variance Score**: 0.2-0.4 (high - different perspectives)
- **Network Anomaly**: 0.7+ (cross-institutional pattern)
- **Final Score**: 0.5-0.8 (flagged for review/block)
- **Recommendation**: BLOCK or REVIEW
- **Consortium Value**: HIGH (pattern invisible to individual banks)

This scenario perfectly demonstrates why banks need consortium intelligence - the fraud is sophisticated enough to fool individual institutions but obvious when banks share intelligence! 🛡️
