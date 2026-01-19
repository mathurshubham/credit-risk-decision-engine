
# Project Alpha: FinTech Credit Risk Decision Engine

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-Astral-purple.svg)](https://github.com/astral-sh/uv)
[![ML Library](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/SHAP-Explainability-orange.svg)](https://shap.readthedocs.io/)
[![Framework](https://img.shields.io/badge/FastAPI-High%20Performance-green.svg)](https://fastapi.tiangolo.com/)
[![Containerization](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

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

## 4. How to Run Locally

You can run this project using **Docker Compose** (Recommended) or the **`uv`** package manager.

### Option A: Docker Compose (Easiest)
This spins up the API in a containerized environment identical to production.

```bash
# 1. Build and Start the Service
docker compose up --build

# The API is now live at http://localhost:9696
```

### Option B: Local Development with `uv`
If you want to train the model or edit code:

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync Dependencies (Creates a virtualenv exactly matching the lock file)
uv sync

# 3. Train the Model (Reproducible Pipeline)
uv run python -m project_alpha.train

# 4. Start the API
uv run uvicorn project_alpha.app:app --reload
```

---

## 5. API Usage & Testing

Once the service is running (on port `9696` via Docker), you can test it.

### 1. Interactive Docs (Swagger UI)
Visit: **[http://localhost:9696/docs](http://localhost:9696/docs)**

### 2. Test via cURL
Copy the command below to test a **High Risk** scenario (Low Income, High Debt):

```bash
curl -X 'POST' \
  'http://127.0.0.1:9696/predict' \
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
-   [x] **Data:** `data/raw_data.csv` included (Git LFS or small enough sample).
-   [x] **Notebooks:** EDA (`01`) and Model Training/SHAP (`02`).
-   [x] **Scripts:** `train.py` (pipeline training) and `app.py` (inference).
-   [x] **Dependencies:** Managed via `pyproject.toml` and `uv.lock`.
-   [x] **Containerization:** `Dockerfile` and `docker-compose.yml`.
-   [x] **Cloud/Local Deployment:** Instructions provided for local Docker run (reproducible).