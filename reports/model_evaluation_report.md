# Model Performance & Evaluation Report: House Price Prediction (INR)

## 1. Executive Summary
- **Champion Model:** `XGBoost`
- **Test Set R² Score:** **0.9825**
- **Test Mean Absolute Error (MAE):** **₹1,450,377** (~₹14.50 Lakhs)
- **Test Root Mean Squared Error (RMSE):** **₹3,183,527** (~₹31.84 Lakhs)
- **Mean Absolute Percentage Error (MAPE):** **4.74%**

---

## 2. Model Benchmark Leaderboard (Indian Rupee Pricing)

| Model | 5-Fold CV R² (Mean ± Std) | Test R² | Test MAE (INR) | Test RMSE (INR) | MAPE (%) |
|---|---|---|---|---|---|
| **Linear Regression** | 0.8389 ± 0.0141 | **0.8265** | ₹6,078,063 | ₹10,030,506 | 26.24% |
| **Ridge Regression** | 0.8293 ± 0.0141 | **0.8205** | ₹6,029,154 | ₹10,202,311 | 25.63% |
| **Random Forest** | 0.9629 ± 0.0175 | **0.961** | ₹1,726,192 | ₹4,755,691 | 5.97% |
| **Gradient Boosting** | 0.9733 ± 0.017 | **0.9811** | ₹1,473,797 | ₹3,314,281 | 5.0% |
| **XGBoost 🏆 (Champion)** | 0.9757 ± 0.0156 | **0.9825** | ₹1,450,377 | ₹3,183,527 | 4.74% |

---

## 3. Explainable AI (XAI) & Feature Contribution Insights (Indian Market)

### Top Predictive Valuation Drivers:
1. **`Area_sqft` & `Sqft_per_Room`**: Dominates valuation variance; built-up area scale remains the baseline foundation for Indian property pricing.
2. **`Location` (Bandra West, Koramangala, Cyber City)**: Metros and tier-1 IT corridor locations carry high base per-sqft rate shifts.
3. **`Parking_Spaces`**: In high-density Indian metros, dedicated car parking spots add distinct pricing premiums.
4. **`House_Age_years`**: Depreciates older constructions relative to newly launched gated communities with modern clubhouse amenities.
5. **`Distance_City_Center_km`**: Captures travel time / proximity discounts away from major economic and employment corridors.

---

## 4. Visual Artifacts
- `reports/figures/actual_vs_predicted.png`: Scatter plot confirming tight linear fit in Crores INR.
- `reports/figures/residuals_distribution.png`: Symmetric, zero-centered Gaussian residual error curve in Lakhs INR.
- `reports/figures/model_comparison_r2.png`: Benchmark chart comparing all 5 algorithms.
- `reports/figures/feature_importance.png`: Top 10 predictive features.
- `reports/figures/shap_summary.png`: Global SHAP value impact distributions across test samples.
