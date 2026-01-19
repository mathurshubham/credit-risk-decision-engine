import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Implements Grouped Median Imputation as per PDF Section 3.2.
    
    Logic:
    - Fills missing 'Monthly_Balance' based on the median of the user's 'Occupation'.
    - If Occupation is missing or new, falls back to the global median.
    """
    def __init__(self):
        self.group_medians = {}
        self.global_median = 0.0

    def fit(self, X, y=None):
        # Check if necessary columns exist
        if 'Monthly_Balance' in X.columns and 'Occupation' in X.columns:
            # 1. Calculate Global Median (Fallback)
            self.global_median = X['Monthly_Balance'].median()
            
            # 2. Calculate Grouped Medians by Occupation
            # We group by Occupation and take the median of Monthly_Balance
            self.group_medians = X.groupby('Occupation')['Monthly_Balance'].median().to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        if 'Monthly_Balance' in X_copy.columns and 'Occupation' in X_copy.columns:
            def fill_balance(row):
                if pd.notnull(row['Monthly_Balance']):
                    return row['Monthly_Balance']
                
                # Look up occupation median
                group_val = self.group_medians.get(row['Occupation'], self.global_median)
                
                # SAFETY FIX: If the group median itself is NaN (e.g., new/empty group), use global
                if pd.isna(group_val):
                    return self.global_median
                return group_val

            X_copy['Monthly_Balance'] = X_copy.apply(fill_balance, axis=1)
            
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs Financial Ratios as per PDF Section 3.3.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Ensure inputs are numeric before calculation (Safety check)
        numeric_cols = ['Annual_Income', 'Outstanding_Debt', 'Num_Credit_Card', 'Monthly_Inhand_Salary']
        for col in numeric_cols:
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0)

        # 1. Debt-to-Income Ratio (DTI)
        # Formula: Outstanding_Debt / Annual_Income
        # We add +1.0 to denominator to avoid DivisionByZero errors
        if 'Outstanding_Debt' in X_copy.columns and 'Annual_Income' in X_copy.columns:
            X_copy['DTI_Ratio'] = X_copy['Outstanding_Debt'] / (X_copy['Annual_Income'] + 1.0)

        # 2. Credit Utilization Proxy
        # PDF Strategy: Outstanding_Debt / (Num_Credit_Card * Proxy_Limit)
        # We assume a standard limit proxy of $5000 per card if limit is unknown
        if 'Outstanding_Debt' in X_copy.columns and 'Num_Credit_Card' in X_copy.columns:
            proxy_limit = X_copy['Num_Credit_Card'] * 5000
            X_copy['Utilization_Proxy'] = X_copy['Outstanding_Debt'] / (proxy_limit + 1.0)

        # 3. Income Stability (Visual check mostly, but we can engineer a ratio)
        # Monthly Salary * 12 vs Annual Income
        if 'Annual_Income' in X_copy.columns and 'Monthly_Inhand_Salary' in X_copy.columns:
            expected_annual = X_copy['Monthly_Inhand_Salary'] * 12
            # Ratio of reported annual income to calculated annual income
            # Values close to 1.0 indicate consistency. Values far off indicate potential fraud/error.
            X_copy['Income_Stability'] = abs(X_copy['Annual_Income'] - expected_annual) / (X_copy['Annual_Income'] + 1.0)

        return X_copy