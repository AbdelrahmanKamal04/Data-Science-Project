# Credit Card Fraud Detection System

A complete **end-to-end data science project** for detecting fraudulent credit card transactions using machine learning, feature engineering, and interactive visualization.

---

## Overview

Financial fraud is a critical challenge in modern digital systems. This project builds a **robust fraud detection pipeline** that:

- Identifies fraudulent transactions
- Handles severe class imbalance
- Engineers meaningful behavioral features
- Compares multiple machine learning models
- Provides an **interactive Streamlit dashboard** for stakeholders

---

## Dataset Description

| Column | Type | Description |
|------|------|------------|
| transaction_id | Integer | Unique transaction ID |
| amount | Float | Transaction value |
| transaction_hour | Integer | Hour of transaction (0–23) |
| merchant_category | Categorical | Merchant type |
| foreign_transaction | Binary | Domestic (0) / Foreign (1) |
| location_mismatch | Binary | Location anomaly flag |
| device_trust_score | Integer | Device reliability score |
| velocity_last_24h | Integer | Transactions in last 24h |
| cardholder_age | Integer | Customer age |
| is_fraud | Binary | Target variable |

---

## Key Insights from EDA

### Class Imbalance
- Fraud cases ≈ **4–5% only**
- Requires resampling (SMOTE)

---

### Strong Fraud Indicators

| Feature | Insight |
|--------|--------|
| foreign_transaction | Strongest fraud signal |
| location_mismatch | Major geographic anomaly indicator |
| velocity_last_24h | Fraud often occurs in bursts |
| device_trust_score | Lower trust → higher fraud risk |
| transaction_hour | Fraud peaks at night |

---

### Behavioral Patterns

- Fraud occurs more during **late-night hours (0–5 AM)**
- Fraudsters perform **rapid transactions (velocity spikes)**
- Foreign + location mismatch = **high-risk scenario**

---

## Data Processing Pipeline

### 1. Data Cleaning
- Removed duplicates  
- Validated data types  
- Standardized schema  

---

### 2. Train / Validation / Test Split
- Stratified split: **70 / 15 / 15**
- Preserves fraud distribution  

---

### 3. Feature Engineering (Train-Driven)

All features are derived **ONLY from training data** to prevent data leakage.

#### Engineered Features

| Feature | Purpose |
|--------|--------|
| log_amount | Normalize skewed transaction values |
| is_night_transaction | Capture late-night fraud activity |
| velocity_per_hour | Detect burst behavior |
| high_risk_abroad | Combine geo risk signals |
| is_high_velocity_low_trust | Risk interaction |
| relevant_amount | Contextual anomaly detection |

---

### 4. Encoding
- One-Hot Encoding for `merchant_category`

---

### 5. Scaling
- StandardScaler applied to numerical features  

---

### 6. Class Balancing
- SMOTE applied **only on training data**

---

## Feature Space (After Processing)

Total features increased from **10 → 16**

Includes:
- Behavioral indicators  
- Risk flags  
- Encoded categorical features  

---

## Models Trained

1. Logistic Regression (Baseline)  
2. Random Forest  
3. XGBoost  
4. LightGBM  
5. Gradient Boosting  

---

## Model Performance

### Best Model: **LightGBM**

**Why?**
- Handles non-linearity well  
- Captures feature interactions  
- Performs best on imbalanced structured data  

---

### Evaluation Strategy

Instead of accuracy alone, we focus on:

- **Recall** → Catch as much fraud as possible  
- **Precision** → Reduce false alarms  
- **F1 Score** → Balance both  
- **ROC-AUC / PR-AUC**

---

## EDA & Visualization

The project includes:

- Univariate analysis  
- Bivariate comparisons (Original vs SMOTE)  
- Correlation heatmaps  
- Temporal fraud trends  
- Feature interaction analysis  

---

## Interactive Dashboard

Built using **Streamlit**, the dashboard provides:

- Dataset overview  
- Fraud distribution insights  
- Feature comparisons (Original vs SMOTE)  
- Correlation & temporal analysis  

### Run Dashboard

```bash
streamlit run dashboard/app.py
