# FinSight

## AI-Powered Real-Time FinTech Fraud Detection & Risk Decisioning Platform

FinSight is an end-to-end fraud detection platform designed to detect suspicious financial transactions in real time.

The system combines:

- Machine Learning
- Feature Engineering
- XGBoost
- Business Risk Decisioning
- FastAPI
- PostgreSQL
- Docker
- Power BI
- Automated Testing

The platform converts a transaction into a fraud probability and then makes an operational decision:

**ALLOW → REVIEW → BLOCK**

---

## Problem Statement

Financial institutions process a large number of transactions every day.

A fraud detection system must:

- detect suspicious transactions
- minimize false positives
- handle highly imbalanced fraud data
- make decisions quickly
- store prediction history
- provide business-level monitoring and analytics

FinSight was built to simulate this type of real-world fraud decisioning workflow.

---

## Solution Architecture

```text
                         ┌─────────────────────────┐
                         │       Financial         │
                         │       Transaction       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Pydantic Validation   │
                         │       FastAPI API       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Real-Time Feature      │
                         │      Engineering        │
                         │                         │
                         │  Amount Features        │
                         │  Balance Features       │
                         │  Transaction Type       │
                         │  Time Features          │
                         │  High-Value Flags       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       XGBoost            │
                         │    Fraud Detection      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Fraud Probability     │
                         │       0.00 → 1.00       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Decision Policy     │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              ┌──────────┐     ┌───────────┐     ┌──────────┐
              │  ALLOW   │     │  REVIEW   │     │  BLOCK   │
              └────┬─────┘     └─────┬─────┘     └────┬─────┘
                   │                 │                 │
                   └─────────────────┼─────────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │      PostgreSQL         │
                         │   Prediction History    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       Power BI          │
                         │    Fraud Dashboard     │
                         └─────────────────────────┘
                    

                    ## Machine Learning Pipeline

```text
Raw PaySim Data
      │
      ▼
Data Profiling
      │
      ▼
Data Cleaning
      │
      ▼
Business Validation
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Feature Audit
      │
      ▼
Real-Time Feature Selection
      │
      ▼
Chronological Train / Validation / Test Split
      │
      ├─────────────────────┐
      ▼                     ▼
Logistic Regression      XGBoost
Baseline
                              │
                              ▼
                       Threshold Optimization
                              │
                              ▼
                       Final Model Evaluation
                              │
                              ▼
                         Risk Decisioning






That tells a very important story:

> You didn't just train XGBoost.

You built a complete ML lifecycle.

---

# 🚀 NEXT: Add a Model Performance Section

Your README currently contains the major metrics, but let's make the model evaluation much more convincing.

Add:

```markdown
## Final Model Performance

The final XGBoost model was selected and evaluated using a chronological validation and test strategy.

### Unseen Test Performance

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9992 |
| PR-AUC | 0.9058 |
| Precision | 95.74% |
| Recall | 69.75% |
| F1 Score | 0.8070 |
| Flagged Transactions | 3,099 |
| Investigation Workload | 0.24% |
| Fraud Value Capture | 96.81% |

### Confusion Matrix

```text
                    Predicted
                  Legitimate   Fraud
Actual Legitimate  1,268,138    132
Actual Fraud           1,287  2,967





This is much stronger than simply saying:

> “My model has 99.92% ROC-AUC.”

Because recruiters working with fraud detection know that a high ROC-AUC alone doesn't tell the complete story.

---

# 🚨 Add the Business Impact Section

This is where FinSight becomes more interesting.

Create:

```markdown
## Business Impact

FinSight is designed around the operational reality of fraud detection.

The goal is not simply to maximize model accuracy. The goal is to identify fraudulent transactions while keeping the investigation workload manageable.

At the selected operating threshold:

- 0.24% of transactions were flagged for investigation/blocking.
- 95.74% precision was achieved for the fraud decision.
- 69.75% of fraud cases were captured.
- 96.81% of fraudulent transaction value was captured.

This demonstrates the trade-off between fraud detection performance and operational investigation capacity.




## Why Chronological Splitting?

Random train/test splitting can allow future transaction patterns to appear in the training data.

FinSight instead uses chronological splitting to simulate a more realistic scenario:

```text
Past Transactions
       ↓
Training
       ↓
Validation
       ↓
Future Transactions
       ↓
Final Test




That sentence about distribution shift is valuable because you actually observed it.

---

# 🔥 Add the Real-Time vs Post-Transaction Feature Design

This is one of the strongest engineering decisions you've made.

Add:

```markdown
## Real-Time vs Post-Transaction Features

Fraud detection systems must distinguish between information available when a transaction decision is being made and information that only becomes available after the transaction.

### Real-Time Features

The real-time XGBoost model uses features available before or during transaction authorization, including:

- transaction type
- transaction amount
- origin balance
- destination balance available before transaction
- transaction time
- hour
- day
- high-value indicator
- zero-amount indicator

### Post-Transaction Features

Additional balance consistency features were investigated for post-transaction monitoring, including:

- origin balance change
- origin balance error
- destination balance change
- destination balance error
- absolute balance errors

These were deliberately excluded from the real-time model because they depend on information generated after the transaction.



## Testing

FinSight uses Pytest for automated API and model testing.

Current test coverage includes:

- Root endpoint
- Health endpoint
- Model information endpoint
- ML prediction functionality

Run:

```bash
pytest tests/ -v



---

# 🐳 NEXT: Document Docker Properly

Your README should tell another developer exactly how to run the project.

Add:

```markdown
## Running with Docker

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FinSight

## Power BI Dashboard

FinSight includes an interactive Power BI dashboard for monitoring:

- Transaction volume
- Fraud decisions
- Risk levels
- Transaction types
- Fraud probability
- Decision trends
- Interactive filtering

![FinSight Dashboard](dashboard/dashboard_overview.png)



## API Documentation

FinSight exposes the fraud detection model through FastAPI.

![FastAPI Documentation](docs/api_docs.png)



