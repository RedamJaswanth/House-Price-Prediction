---
title: Indian Real Estate Valuation & House Price Prediction
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
license: mit
---

# 🏠 Indian Real Estate Valuation & House Price Prediction System (INR ₹)

An end-to-end Data Science and Machine Learning regression system calibrated for the Indian metropolitan housing market. It predicts property market valuations in **Indian Rupees (INR ₹)** based on key architectural, geographical, and community infrastructure characteristics.

---

## 📌 Project Overview
- **Project Domain:** Indian Real Estate Analytics & Property Valuation (PropTech)
- **Problem Type:** Supervised Machine Learning (Regression)
- **Currency:** **Indian Rupee (INR ₹)** — formatted in **Lakhs (L)** and **Crores (Cr)**
- **Champion Algorithm:** **XGBoost Regressor (Test $R^2 = 0.9825$, $\text{MAPE} = 4.32\%$)**
- **User Interface:** Streamlit Interactive Valuation Dashboard with Home Loan EMI calculator
- **Backend Service:** FastAPI Production REST API with Pydantic validation
- **Model Explainability:** SHAP (Shapley Additive Explanations) & Gini/Gain Feature Importance

---

## 🏙️ Key Locations Covered (Indian Metros)
- **Bandra West (Mumbai)** — Premium luxury tier (~₹32,000 / sqft)
- **Koramangala (Bengaluru)** — Prime IT / tech executive hub (~₹15,000 / sqft)
- **Cyber City (Gurugram)** — Corporate & commercial corridor (~₹13,500 / sqft)
- **Whitefield (Bengaluru)** — Major IT park residential corridor (~₹9,500 / sqft)
- **Gachibowli (Hyderabad)** — HITEC city residential belt (~₹9,000 / sqft)
- **Hinjewadi (Pune)** — IT hub residential sector (~₹7,500 / sqft)
- **Noida Sector 62** — NCR commercial & residential node (~₹6,800 / sqft)

---

## 📊 Model Benchmark Leaderboard (INR Pricing)

Evaluated on **1,200 held-out test properties** with 5-fold cross-validation:

| Model | 5-Fold CV $R^2$ | Test $R^2$ | Test MAE (INR) | Test RMSE (INR) | MAPE |
|---|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression (Baseline)** | 0.8389 | 0.8265 | ₹60.78 Lakhs | ₹1.00 Crore | 18.25% |
| **Ridge Regression** | 0.8293 | 0.8205 | ₹60.29 Lakhs | ₹1.02 Crore | 17.80% |
| **Random Forest Regressor** | 0.9629 | 0.9610 | ₹17.26 Lakhs | ₹47.55 Lakhs | 5.86% |
| **Gradient Boosting Regressor** | 0.9733 | 0.9811 | ₹14.73 Lakhs | ₹33.14 Lakhs | 4.41% |
| **XGBoost Regressor 🏆** | **0.9757** | **0.9825** | **₹14.50 Lakhs** | **₹31.83 Lakhs** | **4.32%** |

---

## 🧠 Explainable AI (SHAP) Insights for Indian Housing
1. **Built-up Area (`Area_sqft`)**: Dominant value driver explaining over 65% of price variance.
2. **Location Multipliers**: Micro-markets like *Bandra West (Mumbai)* and *Koramangala (Bengaluru)* create massive positive SHAP shifts into multi-crore valuations.
3. **Dedicated Parking Spaces**: In Indian metros where street parking is heavily congested, each dedicated parking slot adds ₹3.5L – ₹7L in capital value.
4. **Distance to IT Parks (`Distance_City_Center_km`)**: High traffic congestion in cities like Bengaluru, Gurugram, and Mumbai creates steep commute penalties for peripheral properties.

---

## 📁 Repository Structure

```
D:\House_Price_Prediction\
├── data\
│   ├── raw\
│   │   └── housing_data.csv          # 6,000 benchmark property records in INR
│   └── processed\
│       ├── train.csv                 # 80% training set (4,800 records)
│       └── test.csv                  # 20% test set (1,200 records)
├── models\
│   ├── best_model.pkl                # Serialized XGBoost champion model
│   ├── preprocessor.pkl              # Scikit-Learn feature transformer pipeline
│   └── metrics_summary.json          # Benchmark evaluation metrics
├── reports\
│   ├── eda_report.md                 # Statistical findings report in INR
│   ├── model_evaluation_report.md    # Model performance & SHAP analysis
│   └── figures\                      # High-resolution publication plots
│       ├── actual_vs_predicted.png
│       ├── area_vs_price_by_location.png
│       ├── correlation_heatmap.png
│       ├── feature_boxplots.png
│       ├── feature_importance.png
│       ├── model_comparison_r2.png
│       ├── price_distribution.png
│       ├── residuals_distribution.png
│       └── shap_summary.png
├── src\
│   ├── data_generator.py             # Generates benchmark dataset in INR (Lakhs/Crores)
│   ├── eda.py                        # EDA charts & markdown reporter
│   ├── preprocess.py                 # Pipeline: feature engineering & column transforms
│   ├── train.py                      # Multi-model training, 5-fold CV, model serialization
│   ├── evaluate.py                   # SHAP explainability & feature importances
│   └── predict.py                    # Inference engine with Lakhs/Crores formatter
├── tests\
│   └── test_api.py                   # Automated API integration tests
├── app.py                            # Interactive Streamlit Web App with EMI calculator
├── api.py                            # FastAPI production REST service
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── INTERVIEW_GUIDE.md                # Data Science interview Q&A and pitch guide
```

---

## 🚀 How to Run the Project

### 1. Launch the Streamlit Interactive Dashboard
```bash
cd D:\House_Price_Prediction
streamlit run app.py
```
*Open in browser at: `http://localhost:8501`*

### 2. Launch the FastAPI REST Service
```bash
cd D:\House_Price_Prediction
uvicorn api:app --reload --port 8000
```
*Interactive Swagger Documentation: `http://localhost:8000/docs`*

### 3. Sample Prediction Request via cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "location": "Koramangala (Bengaluru)",
       "area_sqft": 1850,
       "bedrooms": 3,
       "bathrooms": 2,
       "floors": 2,
       "house_age_years": 4,
       "parking_spaces": 2,
       "distance_city_center_km": 3.5,
       "school_rating": 8,
       "furnishing": "Fully-Furnished"
     }'
```
Response:
```json
{
  "currency": "INR",
  "estimated_price_inr": 30507910.0,
  "price_formatted": "Rs. 3.05 Crores (Rs. 30,507,910)",
  "price_per_sqft_inr": 16490.76,
  "confidence_interval_95": {
    "lower_inr": 29287593.6,
    "upper_inr": 31728226.4,
    "range_formatted": "Rs. 2.93 Crores (Rs. 29,287,594) - Rs. 3.17 Crores (Rs. 31,728,226)"
  },
  "property_summary": {
    "location": "Koramangala (Bengaluru)",
    "area_sqft": 1850.0,
    "bedrooms": 3,
    "bathrooms": 2,
    "furnishing": "Fully-Furnished"
  }
}
```
