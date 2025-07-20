# Cross-Institutional Fraud Scenarios Explained

## Overview
This document explains the six major fraud scenarios where consortium collaboration provides maximum value compared to individual bank detection.

---

## 1. High-Value Wire Transfer Fraud ($50M Corporate)

### **The Scenario**
Criminals compromise legitimate corporate accounts and initiate massive wire transfers to offshore accounts. They use sophisticated social engineering to bypass internal controls.

### **Why Individual Banks Miss It**
- Smaller banks rarely see $50M+ transactions, so their models aren't trained for this scale
- The accounts are legitimately opened, so identity checks pass
- Transaction appears normal for large corporations

### **How Consortium Adds Value**
- **Bank A specializes** in high-value transaction analysis (serves major corporations)
- **Banks B & C** provide validation that the pattern is unusual for typical customers
- **Cross-validation** confirms whether large transfers are normal for this customer type
- **Result**: Bank A catches what others miss, consortium prevents false negatives

### **Feature Pattern**
- `feature_0` (Amount): 0.95 (extremely high)
- `feature_10` (Geographic): 0.88 (offshore destination)
- `feature_15` (Identity): 0.25 (legitimate account, low identity risk)

---

## 2. Synthetic Identity Fraud (Coordinated Ring)

### **The Scenario**
Criminal network creates hundreds of fake identities using real SSNs combined with fake names and addresses. They build credit history across multiple banks before "busting out."

### **Why Individual Banks Miss It**
- Each bank only sees a fragment of the identity information
- Individual applications appear legitimate
- Credit building phase looks like normal customer behavior

### **How Consortium Adds Value**
- **Bank B specializes** in identity verification and inconsistency detection
- **Cross-institutional view** reveals the same SSN used with different names
- **Pattern recognition** identifies coordinated application timing
- **Network analysis** shows interconnected addresses and phone numbers

### **Feature Pattern**
- `feature_0` (Amount): 0.45 (moderate, not suspicious by amount)
- `feature_15` (Identity): 0.92 (extreme identity inconsistency)
- `feature_20` (Account Age): 0.85 (suspicious new account patterns)

---

## 3. Money Mule Network (Human Trafficking)

### **The Scenario**
Human trafficking organizations recruit victims to rapidly move money across multiple banks. Each victim receives funds and immediately transfers to the next person in the chain.

### **Why Individual Banks Miss It**
- Individual transfers appear legitimate (person-to-person)
- Amounts stay under reporting thresholds
- Victims are real people with legitimate accounts

### **How Consortium Adds Value**
- **Bank C specializes** in velocity and timing analysis
- **Cross-bank timing** reveals impossible coordination patterns
- **Network mapping** shows the same recipients across institutions
- **Velocity aggregation** reveals total flow exceeds individual bank limits

### **Feature Pattern**
- `feature_5` (Velocity): 0.95 (extreme speed of consecutive transfers)
- `feature_25` (Network): 0.90 (strong connectivity patterns)
- `feature_0` (Amount): 0.55 (moderate amounts, under thresholds)

---

## 4. Business Email Compromise (CEO Fraud)

### **The Scenario**
Criminals impersonate company executives to request urgent wire transfers to "confidential suppliers." They research companies and time attacks during executive travel.

### **Why Individual Banks Miss It**
- Requests come from legitimate business accounts
- Companies often have complex supplier relationships
- Individual banks can't verify across corporate banking relationships

### **How Consortium Adds Value**
- **Pattern recognition** across multiple institutions simultaneously
- **Timing correlation** shows attacks hitting multiple companies at once
- **Behavioral analysis** reveals deviation from normal corporate patterns
- **Cross-validation** of supplier legitimacy across bank networks

### **Feature Pattern**
- `feature_0` (Amount): 0.78 (high but reasonable for business)
- `feature_18` (Business): 0.85 (corporate account patterns)
- `feature_22` (Communication): 0.80 (email anomalies)

---

## 5. Cross-Border Laundering (Trade-Based)

### **The Scenario**
International drug cartels use fake trade invoices to justify moving money across borders. They inflate invoice values and use shell companies in multiple countries.

### **Why Individual Banks Miss It**
- Trade finance appears legitimate to individual banks
- Invoice amounts seem reasonable for international trade
- Shell companies have proper documentation

### **How Consortium Adds Value**
- **Multi-country coordination** reveals impossible trade patterns
- **Invoice correlation** shows inflated values across institutions
- **Timing analysis** reveals coordinated money movements
- **Currency pattern** detection across multiple conversion points

### **Feature Pattern**
- `feature_10` (Geographic): 0.95 (multiple high-risk countries)
- `feature_25` (Network): 0.88 (shell company connections)
- `feature_12` (Currency): 0.82 (suspicious conversion patterns)

---

## 6. Cryptocurrency Laundering (Ransomware)

### **The Scenario**
Ransomware groups convert stolen funds to privacy coins through multiple traditional banks before moving to crypto exchanges. They use mixing services to obscure trails.

### **Why Individual Banks Miss It**
- Crypto exchange activity appears normal to individual banks
- Conversion amounts spread across multiple institutions
- Individual transactions look like legitimate crypto trading

### **How Consortium Adds Value**
- **Coordinated conversion tracking** across traditional banking
- **Timing correlation** reveals simultaneous crypto purchases
- **Volume aggregation** shows total conversion exceeds normal patterns
- **Mixing service detection** through cross-institutional flow analysis

### **Feature Pattern**
- `feature_14` (Crypto Exchange): 0.90 (crypto exchange indicators)
- `feature_5` (Velocity): 0.85 (rapid conversion speed)
- `feature_26` (Privacy): 0.80 (privacy coin patterns)

---

## Consortium Value Analysis

### **Maximum Value Indicators**

1. **High Variance + High Max Score**: One bank's specialty catches what others miss
2. **High Consensus + Low Variance**: All banks agree on clear fraud pattern
3. **High Network Anomaly**: Cross-institutional coordination detected
4. **Specialized Detection**: Bank expertise reveals hidden patterns

### **Scoring Interpretation**

- **Consensus Score > 0.8**: Strong agreement across all banks
- **Variance Score > 0.3**: Significant disagreement, suggests specialization
- **Network Anomaly > 0.7**: Clear cross-institutional fraud pattern
- **Final Score**: Weighted combination considering confidence and patterns

### **Real-World Impact**

These scenarios represent billions in annual fraud losses that individual banks cannot effectively detect alone. Consortium collaboration provides:

- **Earlier Detection**: Days to hours improvement
- **Higher Accuracy**: Reduced false positives through cross-validation
- **Network Intelligence**: Patterns invisible to individual institutions
- **Specialized Expertise**: Each bank's strengths benefit the entire network
