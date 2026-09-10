# FinSight: Real-Time FinTech Fraud Detection & Risk Decisioning Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-EB5424?style=flat)](https://xgboost.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

FinSight is a production-grade, end-to-end fraud mitigation and real-time risk decisioning engine designed for high-throughput payment architectures. 

Operating on a benchmark dataset of **6.36 million financial transactions (PaySim)**, FinSight addresses extreme class imbalance, enforces strict temporal feature hygiene to eliminate lookahead bias, optimizes decision boundaries under operational SLA limits, and serves inferences via a containerized **FastAPI** microservice backed by **PostgreSQL** audit logging and **Power BI** risk observability.

---

## Executive Summary & Core Metrics

Traditional rule engines either generate crippling false positive rates or miss multi-variable synthetic fraud networks entirely. FinSight replaces static rules with an optimized, leakage-free Gradient Boosted Tree architecture paired with a three-tier operational risk policy:

$$\text{Transaction Ingestion} \longrightarrow \hat{p}(\text{Fraud}) \longrightarrow \begin{cases} \mathbf{ALLOW} & \hat{p} < \tau_{\text{review}} \\ \mathbf{REVIEW} & \tau_{\text{review}} \le \hat{p} < \tau_{\text{block}} \\ \mathbf{BLOCK} & \hat{p} \ge \tau_{\text{block}} \end{cases}$$

### Key Engineering Benchmarks (Unseen Chronological Test Set)

* **Precision:** **95.74%** (Eliminates genuine transaction disruption).
* **Fraud Value Capture:** **96.81%** of exposed financial loss neutralized.
* **Operational Workload:** Flagged rate contained to **0.24%** of total transaction volume.
* **PR-AUC:** **0.9058** (ROC-AUC: **0.9992**).
* **Serving Latency:** P95 $< 18\text{ ms}$ single-transaction inference budget.

---

## Architecture Overview

```text
                     ┌─────────────────────────────────────────┐
                     │       Incoming Transaction Event        │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │          FastAPI Microservice           │
                     │    (Pydantic Strict Schemas & SLA)      │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │      Zero-Lookahead Feature Layer       │
                     │  - Behavioral Velocity & Delta Spikes   │
                     │  - Temporal Cyclical Components         │
                     │  - Pre-Transaction Balance Ratios       │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │          XGBoost Risk Engine            │
                     │       Calibrated Score Generation       │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │    Dynamic Policy Decision Matrix       │
                     └────┬──────────────────┬─────────────────┘
                          │                  │
        ┌─────────────────┴────────┐         │         ┌─────────────────┴────────┐
        ▼                          ▼         ▼         ▼                          ▼
  ┌───────────┐              ┌───────────────────┐              ┌───────────┐
  │   ALLOW   │              │   MANUAL REVIEW   │              │   BLOCK   │
  │ (Low Risk)│              │   (Medium Risk)   │              │(High Risk)│
  └─────┬─────┘              └─────────┬─────────┘              └─────┬─────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                     ┌─────────────────────────────────────────┐
                     │        PostgreSQL Audit Ledger          │
                     │       Append-Only Risk Decisions        │
                     └─────────────────┬───────────────────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────────────┐
                     │     Power BI Operational Telemetry      │
                     │      Real-Time Fraud & Triage SLA       │
                     └─────────────────────────────────────────┘



Raw Ingestion (PaySim 6.36M)
  │
  ├──► Strict Temporal Sorting
  │
  ├──► Chronological Train / Val / Test Partitioning (Zero Lookahead)
  │
  ├──► Feature Engineering & Target Audit (Exclusion of Post-Transaction Signals)
  │
  ├──► Baseline Benchmarking: Class-Weighted Logistic Regression
  │
  ├──► Production Architecture: Scaled XGBoost
  │
  ├──► Operational Threshold & Cost-Curve Optimization (Argmin Loss)
  │
  └──► Serialization (xgboost_final.joblib) & FastAPI Endpoint Deployment



  PREDICTED
                       Legitimate      Fraud
ACTUAL Legitimate     1,268,138        132       <-- 99.98% True Negative Rate
       Fraud              1,287      2,967       <-- High-Value Catch Rate





FinSight/
├── api/                        # FastAPI application & route controllers
│   ├── main.py                 # Endpoint routing & lifespan handlers
│   └── schemas.py              # Pydantic contract definitions
├── dashboard/                  # Business intelligence assets
│   └── dashboard_overview.png  # Power BI interface preview
├── data/                       # Local data mount (git-ignored)
│   └── .gitkeep
├── docs/                       # Architectural diagrams & specifications
│   └── api_docs.png
├── models/                     # Serialized production models
│   └── xgboost_final.joblib    # Frozen model weights (1.82 MB)
├── notebooks/                  # Experimental audit trail
│   ├── 01_data_understanding.ipynb
│   └── 02_feature_analysis.ipynb
├── reports/                    # Reproducibility benchmarks & audits
│   ├── feature_audit.csv
│   ├── policy_comparison.csv
│   └── risk_strategy_comparison.csv
├── src/                        # Core system modules
│   ├── features/               # Real-time feature pipelines
│   ├── models/                 # Model builders & calibrators
│   └── risk/                   # Decision engine & threshold rules
├── tests/                      # Automated test suite
│   └── test_api.py
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile                  # Production container definition
├── requirements.txt            # Pinned dependency manifest
└── README.md




