
# Cloud Deployment Guide (Google Cloud Run)

This document details the steps to deploy **Project Alpha: Credit Risk Engine** to **Google Cloud Run**, a serverless platform that automatically scales containers.

The service is currently live at:
**[https://credit-risk-engine-644458477502.us-central1.run.app/docs](https://credit-risk-engine-644458477502.us-central1.run.app/docs)**

## Prerequisites

1.  **Google Cloud Account** with billing enabled.
2.  **Google Cloud SDK (`gcloud` CLI)** installed and authorized (`gcloud auth login`).
3.  **Docker** installed locally (optional, for local testing before push).

## Deployment Steps

### 1. Project Configuration

Initialize the `gcloud` environment and set the active project.

```bash
# Set your GCP Project ID
gcloud config set project predictive-maint-api-2025

# Enable necessary APIs (if not already enabled)
gcloud services enable artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com
```

### 2. Create Artifact Registry

Create a private Docker repository to store the container images in the `us-central1` region.

```bash
gcloud artifacts repositories create alpha-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Project Alpha Credit Risk Repo"
```

### 3. Build & Push Docker Image

We use **Google Cloud Build** to build the image remotely. This zips the local source code, sends it to Google Cloud, builds the Docker container (installing `uv` dependencies), and stores it in the Artifact Registry.

Run this command from the project root:

```bash
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/predictive-maint-api-2025/alpha-repo/credit-risk-service:v1 .
```

### 4. Deploy to Cloud Run

Deploy the container as a serverless microservice. We explicitly set the port to `8000` because our FastAPI application (via Uvicorn) listens on that port.

```bash
gcloud run deploy credit-risk-engine \
    --image=us-central1-docker.pkg.dev/predictive-maint-api-2025/alpha-repo/credit-risk-service:v1 \
    --platform=managed \
    --region=us-central1 \
    --allow-unauthenticated \
    --port=8000 \
    --memory=1Gi
```

*   `--allow-unauthenticated`: Makes the API publicly accessible (required for portfolio demonstration).
*   `--port=8000`: Matches the `EXPOSE 8000` instruction in our Dockerfile.
*   `--memory=1Gi`: Allocates sufficient RAM for XGBoost and Pandas operations.

---

## Testing the Production Service

### 1. Interactive Documentation (Swagger UI)
You can test the API directly in your browser using the auto-generated Swagger UI:
👉 **[Open Swagger UI](https://credit-risk-engine-644458477502.us-central1.run.app/docs)**

### 2. Test via cURL
Run the following command in your terminal to get a real credit risk prediction from the cloud:

```bash
curl -X 'POST' \
  'https://credit-risk-engine-644458477502.us-central1.run.app/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "Age": 28,
  "Occupation": "Teacher",
  "Annual_Income": 50000.0,
  "Monthly_Inhand_Salary": 3000.0,
  "Num_Bank_Accounts": 2,
  "Num_Credit_Card": 4,
  "Interest_Rate": 10,
  "Num_of_Loan": 1,
  "Delay_from_due_date": 2,
  "Num_of_Delayed_Payment": 1,
  "Changed_Credit_Limit": 5.0,
  "Num_Credit_Inquiries": 1,
  "Credit_Mix": "Standard",
  "Outstanding_Debt": 1200.0,
  "Credit_Utilization_Ratio": 35.0,
  "Payment_of_Min_Amount": "No",
  "Total_EMI_per_month": 50.0,
  "Amount_invested_monthly": 100.0,
  "Payment_Behaviour": "Low_spent_Small_value_payments",
  "Monthly_Balance": 400.0
}'
```

**Expected Response:**
```json
{
  "credit_score": "Standard",
  "probability": {
    "Good": 0.118,
    "Standard": 0.832,
    "Poor": 0.05
  },
  "risk_level": "Medium"
}
```

---

## Teardown (Cleanup)

To avoid incurring ongoing charges for storage or compute, verify the resources and remove them when finished.

### 1. Verify Active Resources
Check what is currently running or stored:
```bash
gcloud run services list
gcloud artifacts repositories list
```

### 2. Delete Resources
Run these commands to permanently delete the deployment and the images:

```bash
# 1. Delete the Cloud Run Service
gcloud run services delete credit-risk-engine --region=us-central1 --quiet

# 2. Delete the Container Images Repository
gcloud artifacts repositories delete alpha-repo --location=us-central1 --quiet
```