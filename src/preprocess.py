"""
Preprocessing and Feature Engineering Pipeline (INR Dataset)
Handles data splitting, feature engineering, transformation, and serialization.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to compute domain-specific real estate features."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        
        # Room ratios and metrics
        total_rooms = X_df["Bedrooms"] + X_df["Bathrooms"]
        X_df["Total_Rooms"] = total_rooms
        X_df["Bed_to_Bath_Ratio"] = X_df["Bedrooms"] / np.maximum(X_df["Bathrooms"], 1)
        X_df["Sqft_per_Room"] = X_df["Area_sqft"] / np.maximum(total_rooms, 1)
        
        # Age and luxury flags
        X_df["Is_New_Construction"] = (X_df["House_Age_years"] <= 5).astype(int)
        X_df["Is_Luxury_Sqft"] = (X_df["Area_sqft"] >= 2500).astype(int)
        
        return X_df

def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Builds a scikit-learn ColumnTransformer for numerical scaling and categorical encoding."""
    
    all_num_cols = numerical_cols + [
        "Total_Rooms", "Bed_to_Bath_Ratio", "Sqft_per_Room", 
        "Is_New_Construction", "Is_Luxury_Sqft"
    ]
    
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, all_num_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )
    
    full_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

def prepare_and_split_data(raw_data_path: str, processed_dir: str, test_size: float = 0.2, random_state: int = 42):
    raw_path = Path(raw_data_path)
    proc_dir = Path(processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(raw_path)
    X = df.drop(columns=["Price_INR"])
    y = df["Price_INR"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(proc_dir / "train.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)
    
    print(f"Data split successfully:")
    print(f"Training set: {train_df.shape} -> saved to {proc_dir / 'train.csv'}")
    print(f"Testing set: {test_df.shape} -> saved to {proc_dir / 'test.csv'}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_and_split_data(
        raw_data_path="D:/House_Price_Prediction/data/raw/housing_data.csv",
        processed_dir="D:/House_Price_Prediction/data/processed"
    )
