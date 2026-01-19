import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

class RegexCleaner(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to clean 'dirty' string columns containing
    special characters (e.g., '23_', '1000.00_') and convert them to numeric.
    """
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in self.columns:
            if col in X_copy.columns:
                # 1. Convert to string to handle mixed types
                # 2. Regex: Keep only digits, dots, and minus signs
                X_copy[col] = X_copy[col].astype(str).apply(
                    lambda x: re.sub(r'[^\d.-]', '', x)
                )
                
                # 3. Handle empty strings resulting from regex (e.g., if value was just "_")
                X_copy[col] = X_copy[col].replace('', np.nan)
                
                # 4. Convert to float
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                
        return X_copy

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Handles physically impossible outliers via Winsorization or Capping.
    
    Logic based on PDF Section 3.2:
    - Age: Values < 18 or > 100 are replaced with the Median (as they are likely errors).
    - Bank Accounts: Values > 20 are capped at 20 (conservative upper bound for consumers).
    """
    def __init__(self):
        self.age_min = 18
        self.age_max = 100
        self.max_accounts = 20
        self.median_age = 30 # Default fallback, will be learned in fit()

    def fit(self, X, y=None):
        # Learn the median age from valid data only
        if 'Age' in X.columns:
            # We assume RegexCleaner has already run, so Age is float
            # Filter for valid range to calculate a true median
            valid_ages = X['Age'][
                (X['Age'] >= self.age_min) & 
                (X['Age'] <= self.age_max)
            ]
            if not valid_ages.empty:
                self.median_age = valid_ages.median()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Handle Age Outliers (Impute with Median)
        if 'Age' in X_copy.columns:
            mask = (X_copy['Age'] < self.age_min) | (X_copy['Age'] > self.age_max)
            X_copy.loc[mask, 'Age'] = self.median_age
            
        # 2. Handle Bank Account Outliers (Hard Cap)
        if 'Num_Bank_Accounts' in X_copy.columns:
            X_copy.loc[X_copy['Num_Bank_Accounts'] > self.max_accounts, 'Num_Bank_Accounts'] = self.max_accounts
            # Also handle negative accounts if any
            X_copy.loc[X_copy['Num_Bank_Accounts'] < 0, 'Num_Bank_Accounts'] = 0

        return X_copy