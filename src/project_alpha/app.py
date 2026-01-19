import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Define the path to the model
MODEL_PATH = "model.joblib"
pipeline = None

# ==========================================
# 1. LIFESPAN (Startup/Shutdown)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
    yield
    # Clean up on shutdown (if needed)
    pipeline = None

# Initialize FastAPI
app = FastAPI(title="Project Alpha: Credit Risk Engine", lifespan=lifespan)

# ==========================================
# 2. DATA CONTRACT (Pydantic)
# ==========================================
class CreditApplication(BaseModel):
    # We use basic types (float/int/str) but the pipeline can handle strings for numbers 
    # thanks to your RegexCleaner.
    Age: float = Field(..., example=34)
    Occupation: str = Field(..., example="Engineer")
    Annual_Income: float = Field(..., example=65000.0)
    Monthly_Inhand_Salary: float = Field(..., example=4500.0)
    Num_Bank_Accounts: int = Field(..., example=4)
    Num_Credit_Card: int = Field(..., example=3)
    Interest_Rate: int = Field(..., example=15)
    Num_of_Loan: int = Field(..., example=2)
    Delay_from_due_date: int = Field(..., example=5)
    Num_of_Delayed_Payment: int = Field(..., example=1)
    Changed_Credit_Limit: float = Field(..., example=1200.0)
    Num_Credit_Inquiries: int = Field(..., example=4)
    Credit_Mix: str = Field(..., example="Good")
    Outstanding_Debt: float = Field(..., example=800.0)
    Credit_Utilization_Ratio: float = Field(..., example=30.0)
    Payment_of_Min_Amount: str = Field(..., example="No")
    Total_EMI_per_month: float = Field(..., example=150.0)
    Amount_invested_monthly: float = Field(..., example=80.0)
    Payment_Behaviour: str = Field(..., example="High_spent_Small_value_payments")
    Monthly_Balance: float = Field(..., example=350.0)

# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict")
def predict(application: CreditApplication):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Convert Pydantic object to Pandas DataFrame
        # We wrap it in a list [application.dict()] to make it a single-row DataFrame
        data = application.model_dump()
        df = pd.DataFrame([data])
        
        # 2. Predict
        # The pipeline handles all cleaning/encoding automatically
        prediction_idx = pipeline.predict(df)[0]
        probs = pipeline.predict_proba(df)[0]
        
        # 3. Map result back to readable string
        # Recall mapping: Good->0, Standard->1, Poor->2
        class_map = {0: 'Good', 1: 'Standard', 2: 'Poor'}
        result = class_map[prediction_idx]
        
        # 4. Construct Response
        return {
            "credit_score": result,
            "probability": {
                "Good": round(float(probs[0]), 3),
                "Standard": round(float(probs[1]), 3),
                "Poor": round(float(probs[2]), 3)
            },
            "risk_level": "High" if result == 'Poor' else "Low" if result == 'Good' else "Medium"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))