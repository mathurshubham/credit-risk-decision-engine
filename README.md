
# Project Alpha: FinTech Credit Risk Decision Engine

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-Astral-purple.svg)](https://github.com/astral-sh/uv)
[![ML Library](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Framework](https://img.shields.io/badge/FastAPI-High%20Performance-green.svg)](https://fastapi.tiangolo.com/)
[![Deployment](https://img.shields.io/badge/Google_Cloud-Run-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)

---

## 🚀 Live Demo

The project is deployed serverless on **Google Cloud Run**. You can test the machine learning model immediately without installing anything.

**👉 [Access Live API Documentation (Swagger UI)](https://credit-risk-engine-644458477502.us-central1.run.app/docs)**

---

## 1. Project Overview

**Project Alpha** is a production-grade machine learning system designed to assess credit risk in a regulated FinTech environment. Unlike standard academic projects that assume clean data, this engine is engineered to ingest **"dirty" raw banking data**, clean it automatically, and predict a borrower's creditworthiness (`Good`, `Standard`, or `Poor`).

### 1.1. The Business Challenge
Financial institutions lose millions annually due to:
1.  **Default Risk:** Approving borrowers who cannot repay.
2.  **Operational Inefficiency:** Manual review of messy data.
3.  **Regulatory Compliance:** The "Black Box" problem—decisions must be explainable.

### 1.2. The Solution
This API provides a real-time `credit_score` and `risk_level`, complete with explainability features.
*   **Robustness:** Handles data anomalies (e.g., negative ages, typos in income) via a custom cleaning pipeline.
*   **Performance:** Uses **XGBoost** for state-of-the-art tabular classification.
*   **Transparency:** Integrates **SHAP (SHapley Additive exPlanations)** to explain *why* a specific applicant was rejected or approved.

---

## 2. Technical Architecture & Stack

This project adopts a modern "Senior Engineer" stack, moving away from legacy tools to maximize performance and reproducibility.

*   **Language:** Python 3.10
*   **Dependency Management:** **`uv`** (A high-performance Rust-based replacement for pip/poetry).
*   **Machine Learning:**
    *   **XGBoost:** Champion model for classification.
    *   **Scikit-Learn Pipelines:** For encapsulating cleaning and preprocessing logic.
    *   **SHAP:** For model interpretability.
*   **API Framework:** **FastAPI** (Async, Type-safe inputs via Pydantic).
*   **Containerization:** Multi-stage **Docker** builds and **Docker Compose**.
*   **Cloud:** **Google Cloud Run** (Serverless container deployment).

---

## 3. Engineering Workflow

### 3.1. The "Dirty Data" Pipeline
Real-world data is messy. We engineered custom Scikit-Learn transformers to handle specific anomalies found during EDA:
*   **`RegexCleaner`:** Fixes typo-ridden numerical columns (e.g., converting `"23_"` or `"19,000.00_"` to floats).
*   **`OutlierCapper`:** Handles physically impossible values (e.g., `Age: -500` or `Num_Bank_Accounts: 1798`) via Winsorization.
*   **`MissingValueImputer`:** Implements Grouped Median Imputation based on Occupation.

### 3.2. Financial Feature Engineering
We enriched the dataset with domain-specific ratios:
*   **DTI (Debt-to-Income):** A critical indicator of leverage.
*   **Credit Utilization Proxy:** Estimates how maxed out a borrower's cards are.
*   **Income Stability:** Flags discrepancies between annual and monthly reported income.

---

## 4. API Usage & Testing

You can test the API against the live cloud deployment or run it locally.

### Option A: Test Live (Cloud)
Copy and paste this command into your terminal to hit the production endpoint:

```bash
curl -X 'POST' \
  'https://credit-risk-engine-644458477502.us-central1.run.app/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "Age": 28,
  "Occupation": "Teacher",
  "Annual_Income": 15000.0,
  "Monthly_Inhand_Salary": 1200.0,
  "Num_Bank_Accounts": 8,
  "Num_Credit_Card": 10,
  "Interest_Rate": 25,
  "Num_of_Loan": 9,
  "Delay_from_due_date": 20,
  "Num_of_Delayed_Payment": 15,
  "Changed_Credit_Limit": 5.0,
  "Num_Credit_Inquiries": 10,
  "Credit_Mix": "Bad",
  "Outstanding_Debt": 4000.0,
  "Credit_Utilization_Ratio": 90.0,
  "Payment_of_Min_Amount": "Yes",
  "Total_EMI_per_month": 500.0,
  "Amount_invested_monthly": 0.0,
  "Payment_Behaviour": "Low_spent_Small_value_payments",
  "Monthly_Balance": 100.0
}'
```

**Expected Response:**
```json
{
  "credit_score": "Poor",
  "probability": {
    "Good": 0.01,
    "Standard": 0.15,
    "Poor": 0.84
  },
  "risk_level": "High"
}
```

### Option B: Run Locally (Docker)
To run the container on your own machine:

```bash
# 1. Build and Start the Service
docker compose up --build

# 2. Access the API at http://localhost:9696/docs
```

---

## 5. Development Setup

If you wish to contribute or modify the code, we use `uv` for lightning-fast dependency management.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync Dependencies (Creates a virtualenv exactly matching the lock file)
uv sync

# 3. Train the Model (Reproducible Pipeline)
uv run python -m project_alpha.train

# 4. Start the API in Dev Mode
uv run uvicorn project_alpha.app:app --reload
```

---

## 6. Project Structure

```
project_alpha/
├── data/                   # Raw data (simulating a data warehouse dump)
├── notebooks/
│   ├── 01_data_profiling.ipynb   # EDA proving data is "dirty"
│   └── 02_modeling.ipynb         # Model Training & SHAP Analysis
├── src/
│   └── project_alpha/
│       ├── cleaning.py     # Custom Regex & Outlier transformers
│       ├── features.py     # Financial Ratio logic
│       ├── train.py        # Reproducible Training Script
│       └── app.py          # FastAPI Application
├── Dockerfile              # Multi-stage build optimized for uv
├── docker-compose.yml      # Orchestration
├── pyproject.toml          # Project metadata
├── uv.lock                 # Strict dependency locking
└── model.joblib            # Serialized Pipeline (Cleaners + XGBoost)
```

## 7. Course Deliverables Checklist

-   [x] **`README.md`:** This file.
-   [x] **Data:** `data/raw_data.csv` included.
-   [x] **Notebooks:** EDA (`01`) and Model Training/SHAP (`02`).
-   [x] **Scripts:** `train.py` (pipeline training) and `app.py` (inference).
-   [x] **Dependencies:** Managed via `pyproject.toml` and `uv.lock`.
-   [x] **Containerization:** `Dockerfile` and `docker-compose.yml`.
-   [x] **Cloud Deployment:** Live on Google Cloud Run with public URL provided.