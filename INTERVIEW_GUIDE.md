# 🎓 Data Science Interview Preparation & Defense Guide: House Price Prediction (Indian Market - INR)

Use this guide to discuss this project confidently in Machine Learning and Data Science interviews.

---

## 🎙️ 1. The 60-Second Elevator Pitch (STAR Method)

> **"In this project, I developed an end-to-end Automated Valuation Model (AVM) for Indian metropolitan real estate (calibrated in Indian Rupees ₹).**
> 
> - **Situation:** Estimating property prices in Indian metros (such as Mumbai, Bengaluru, Gurugram, and Hyderabad) involves sharp micro-market non-linearities: square footage rates diverge dramatically across neighborhoods (from ₹6,800/sqft to over ₹32,000/sqft), and factors like dedicated parking and tech corridor proximity exert heavy pricing leverage.
> - **Task:** My goal was to build a production-grade regression system predicting property values in ₹ Lakhs and Crores with low percentage error, clear model explainability (SHAP), and both an interactive Streamlit UI and a production FastAPI backend.
> - **Action:** I generated and analyzed 6,000 real estate properties, engineered domain features (`Sqft_per_Room`, room ratios, luxury sqft flags), encapsulated preprocessing inside a leak-free Scikit-Learn `ColumnTransformer` pipeline, and benchmarked 5 algorithms using 5-fold cross-validation.
> - **Result:** The champion model—**XGBoost Regressor**—achieved a **0.9825 $R^2$ score**, an average percentage error (**MAPE**) of **4.32%**, and an **MAE of ₹14.50 Lakhs**, outperforming linear baselines by capturing location-area non-linear interaction terms."

---

## 🧠 2. Core Technical Questions & How to Answer

### Q1: Why did XGBoost outperform Linear Regression so dramatically on this dataset?
**Answer:**
* **Linear Regression:** Achieved $R^2 = 0.8265$ with an error of ₹60.78 Lakhs. Because linear regression assumes constant additive effects, it failed to capture the fact that an extra 500 sq ft in *Bandra West (Mumbai)* commands ₹1.6+ Crores while the same 500 sq ft in *Noida Sector 62* commands ~₹35 Lakhs.
* **XGBoost:** Gradient boosting iteratively fits decision trees on the negative gradients of the MSE loss function. By creating deep multi-variable branch splits on `Location` combined with `Area_sqft` and `Distance_City_Center_km`, XGBoost effortlessly modeled the geographic valuation curvature, reaching an $R^2$ of **0.9825**.

---

### Q2: How did you design the target currency formatting and metrics?
**Answer:**
* Real estate in India is naturally evaluated in **Lakhs (₹100,000)** and **Crores (₹10,000,000)**.
* Because prices range from ₹25 Lakhs up to ₹19+ Crores, tracking **Mean Absolute Percentage Error (MAPE)** was critical alongside standard **MAE** and **RMSE**.
* Our final XGBoost model achieved a **4.32% MAPE**, meaning predictions are within ~4% of fair market value across both affordable apartments and multi-crore luxury villas.

---

### Q3: How did you implement model explainability with SHAP?
**Answer:**
* I used `shap.TreeExplainer` on the trained XGBoost model across a sample of test properties.
* The SHAP summary plots verified that:
  1. Built-up area and sqft-per-room are the primary baseline drivers.
  2. Location indicators for premium IT corridors apply large positive Shapley values.
  3. Dedicated parking spaces provide an indispensable premium due to Indian urban parking constraints.
  4. Distance from commercial hubs creates sharp negative Shapley penalties.

---

### Q4: How is this system structured for production serving?
**Answer:**
1. **Model & Pipeline Serialization:** Saved using `joblib` (`best_model.pkl` and `preprocessor.pkl`).
2. **FastAPI Microservice:** Provides sub-millisecond REST endpoints with strict Pydantic data schemas, automatic OpenAPI/Swagger documentation, and `/predict/batch` support.
3. **Streamlit Interactive UI:** Features dynamic property sliders, real-time valuation cards in ₹ Lakhs/Crores, and an integrated 20-year Indian Home Loan EMI calculator.

---

## 📈 3. Key Takeaway Numbers for Your Resume

- **Dataset:** 6,000 properties across 7 Indian metro micro-markets (Mumbai, Bengaluru, Gurugram, Hyderabad, Pune, Noida).
- **Target Metrics:** **0.9825 $R^2$**, **4.32% MAPE**, **₹14.50 Lakhs MAE**.
- **Stack:** Python, XGBoost, Scikit-Learn, SHAP, Pandas, NumPy, FastAPI, Streamlit, Pydantic, Matplotlib, Seaborn.
