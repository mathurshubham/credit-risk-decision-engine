import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Import our custom modules
from project_alpha.cleaning import RegexCleaner, OutlierCapper
from project_alpha.features import MissingValueImputer, FeatureEngineer

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "data/raw_data.csv"
MODEL_PATH = "model.joblib"
RANDOM_STATE = 42

DIRTY_NUMERIC_COLS = [
    'Age', 'Annual_Income', 'Num_of_Loan', 'Num_Bank_Accounts', 
    'Num_Credit_Card', 'Interest_Rate', 'Delay_from_due_date', 
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Monthly_Inhand_Salary', 
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
]

CATEGORICAL_COLS = [
    'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
]

def train():
    print("🚀 Starting Project Alpha Training Pipeline...")
    
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. Target Preprocessing
    target_map = {'Good': 0, 'Standard': 1, 'Poor': 2}
    print("Mapping Target: Good->0, Standard->1, Poor->2")
    df['target'] = df['Credit_Score'].map(target_map)
    df = df.dropna(subset=['target'])
    
    # 3. Drop Unused/Messy Columns
    drop_cols = [
        'Credit_Score', 'target', 'ID', 'Customer_ID', 'Name', 'SSN', 'Month',
        'Type_of_Loan', 'Credit_History_Age'
    ]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['target']

    print(f"Feature columns: {list(X.columns)}")

    # 4. Stratified Split
    print("Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 5. Construct Pipeline
    print("Constructing Scikit-Learn Pipeline...")
    
    # Preprocessor for Categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL_COLS)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Main Pipeline (Flattened Structure)
    model_pipeline = Pipeline([
        # --- Cleaning Steps ---
        ('regex', RegexCleaner(columns=DIRTY_NUMERIC_COLS)),
        ('outliers', OutlierCapper()),
        ('imputer', MissingValueImputer()),
        ('features', FeatureEngineer()),
        
        # --- Encoding ---
        ('preprocessor', preprocessor),
        
        # --- Model ---
        ('classifier', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # 6. Training
    print("🧠 Training XGBoost Model...")
    model_pipeline.fit(X_train, y_train)

    # 7. Evaluation
    print("Evaluating Model Performance...")
    # This will now work because 'model_pipeline' is a single flat chain
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good', 'Standard', 'Poor']))

    # 8. Serialization
    print(f"💾 Saving reproducible pipeline to {MODEL_PATH}...")
    joblib.dump(model_pipeline, MODEL_PATH)
    print("Training Complete.")

if __name__ == "__main__":
    train()