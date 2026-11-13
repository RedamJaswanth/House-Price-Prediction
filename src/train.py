"""
Model Training, Cross-Validation, and Comparison Module (INR Currency)
Trains Linear Regression, Ridge, Random Forest, Gradient Boosting, and XGBoost on INR dataset.
Evaluates MAE, RMSE, R2, saves best model, and generates evaluation artifacts.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from preprocess import create_preprocessing_pipeline

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "MAE_INR": round(float(mae), 2),
        "RMSE_INR": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "MAPE_pct": round(float(mape), 2)
    }

def train_and_evaluate(processed_dir: str, models_dir: str, reports_dir: str):
    proc_path = Path(processed_dir)
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    figures_path = reports_path / "figures"
    
    models_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Load Train and Test
    train_df = pd.read_csv(proc_path / "train.csv")
    test_df = pd.read_csv(proc_path / "test.csv")
    
    X_train = train_df.drop(columns=["Price_INR"])
    y_train = train_df["Price_INR"]
    X_test = test_df.drop(columns=["Price_INR"])
    y_test = test_df["Price_INR"]
    
    numerical_cols = [
        "Area_sqft", "Bedrooms", "Bathrooms", "Floors", 
        "House_Age_years", "Parking_Spaces", "Distance_City_Center_km", "School_Rating"
    ]
    categorical_cols = ["Location", "Furnishing"]
    
    # Build Preprocessing Pipeline
    prep_pipeline = create_preprocessing_pipeline(numerical_cols, categorical_cols)
    print("Fitting preprocessing pipeline on INR dataset...")
    X_train_transformed = prep_pipeline.fit_transform(X_train)
    X_test_transformed = prep_pipeline.transform(X_test)
    
    # Save Preprocessor
    joblib.dump(prep_pipeline, models_path / "preprocessor.pkl")
    print(f"Preprocessor saved to: {models_path / 'preprocessor.pkl'}")
    
    # Define Candidate Models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=250, learning_rate=0.06, max_depth=5, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    }
    
    results = {}
    fitted_models = {}
    test_predictions = {}
    
    print("\n--- Training & Benchmarking Models (INR) ---")
    for name, model in candidate_models.items():
        print(f"Evaluating {name}...")
        
        # 5-Fold Cross-Validation on Training set
        cv_scores = cross_val_score(model, X_train_transformed, y_train, cv=5, scoring='r2', n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        
        # Fit on entire training set
        model.fit(X_train_transformed, y_train)
        fitted_models[name] = model
        
        # Predict on Test set
        y_pred = model.predict(X_test_transformed)
        test_predictions[name] = y_pred
        
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["CV_R2_Mean"] = round(cv_mean, 4)
        metrics["CV_R2_Std"] = round(cv_std, 4)
        
        results[name] = metrics
        print(f"  Test R2: {metrics['R2']:.4f} | RMSE: INR {metrics['RMSE_INR']:,.0f} | MAE: INR {metrics['MAE_INR']:,.0f} | CV R2: {cv_mean:.4f}")

    # Identify Best Model by R2 Score
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_model = fitted_models[best_model_name]
    best_preds = test_predictions[best_model_name]
    print(f"\n==========================================")
    print(f"WINNER: {best_model_name} with Test R2 = {results[best_model_name]['R2']}")
    print(f"==========================================")
    
    # Save Best Model
    joblib.dump(best_model, models_path / "best_model.pkl")
    print(f"Best model saved to: {models_path / 'best_model.pkl'}")
    
    # Save Metrics Leaderboard to JSON
    summary_data = {
        "best_model": best_model_name,
        "currency": "INR",
        "leaderboard": results,
        "features": {
            "numerical": numerical_cols,
            "categorical": categorical_cols
        }
    }
    with open(models_path / "metrics_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)
        
    # Generate Visualizations (in Crores / Lakhs)
    # 1. Actual vs Predicted (in ₹ Crores)
    plt.figure(figsize=(9, 7))
    plt.scatter(y_test / 1e7, best_preds / 1e7, alpha=0.4, color="#1f77b4", edgecolor='none')
    min_val = min(y_test.min(), best_preds.min()) / 1e7
    max_val = max(y_test.max(), best_preds.max()) / 1e7
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Valuation Line (y = x)")
    plt.title(f"Actual vs. Predicted House Prices ({best_model_name})", fontsize=13, fontweight='bold')
    plt.xlabel("Actual Price (Crores INR)", fontsize=11)
    plt.ylabel("Predicted Price (Crores INR)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_path / "actual_vs_predicted.png", dpi=200)
    plt.close()
    
    # 2. Residual Distribution (in ₹ Lakhs)
    residuals_lakhs = (y_test - best_preds) / 1e5
    plt.figure(figsize=(9, 6))
    sns.histplot(residuals_lakhs, kde=True, color="#2ca02c", bins=35)
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.title(f"Residuals Error Distribution in Lakhs INR ({best_model_name})", fontsize=13, fontweight='bold')
    plt.xlabel("Residual Error (Actual - Predicted) in Lakhs INR", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_path / "residuals_distribution.png", dpi=200)
    plt.close()
    
    # 3. Model Benchmark Bar Chart
    plt.figure(figsize=(10, 5))
    model_names = list(results.keys())
    r2_scores = [results[m]["R2"] for m in model_names]
    bars = plt.barh(model_names, r2_scores, color="#3470a3")
    plt.xlim(0.8, 1.0)
    plt.xlabel("Test R² Score", fontsize=11, fontweight='bold')
    plt.title("Model Performance Comparison on INR Pricing (R² Score)", fontsize=13, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        plt.text(w - 0.015, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_path / "model_comparison_r2.png", dpi=200)
    plt.close()
    
    print("All training evaluation figures successfully saved!")
    return results

if __name__ == "__main__":
    train_and_evaluate(
        processed_dir="D:/House_Price_Prediction/data/processed",
        models_dir="D:/House_Price_Prediction/models",
        reports_dir="D:/House_Price_Prediction/reports"
    )
