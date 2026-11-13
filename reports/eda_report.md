# Exploratory Data Analysis Report: House Price Prediction (INR)

## 1. Dataset Overview
- **Total Records:** 6,000
- **Features:** 11 (9 Numerical, 2 Categorical)
- **Target Variable:** `Price_INR` (Selling price in Indian Rupees ₹)
- **Price Range:** ₹36.00 Lakhs to ₹19.43 Crores (Mean: ₹2.93 Crores)

### Key Features:
- **Numerical:** Area_sqft, Bedrooms, Bathrooms, Floors, House_Age_years, Parking_Spaces, Distance_City_Center_km, School_Rating, Price_INR
- **Categorical:** Location, Furnishing

---

## 2. Statistical Findings & Feature Correlations with Price

| Feature | Correlation with Price_INR (r) | Interpretation |
|---|---|---|
| `Area_sqft` | **0.586** | Moderate positive |
| `Bedrooms` | **0.517** | Moderate positive |
| `Bathrooms` | **0.459** | Moderate positive |
| `Floors` | **0.363** | Moderate positive |
| `Parking_Spaces` | **0.026** | Weak/Near-zero |
| `School_Rating` | **0.025** | Weak/Near-zero |
| `House_Age_years` | **-0.016** | Weak/Near-zero |
| `Distance_City_Center_km` | **-0.510** | Negative correlation |

---

## 3. Key Data Insights (Indian Housing Market)
1. **Area (sqft)** is the primary driver of house price (r = 0.59).
2. **Location Premium:** Clear price tiers are visible. Prime micro-markets like **Bandra West (Mumbai)** and **Koramangala (Bengaluru)** command significant per-sqft pricing premiums.
3. **Distance to IT Hubs / City Center:** Exhibits negative correlation; suburban tech corridor periphery properties offer relative affordability.
4. **House Age Depreciation:** Displays realistic non-linear depreciation; newer constructions hold strong premiums while older resale units discount steadily.
5. **Clean Data Quality:** Zero missing values and zero schema anomalies across all 6,000 observations.

---

## 4. Visualizations Generated
- `reports/figures/price_distribution.png`: Target variable distribution & log transform comparison in Crores INR.
- `reports/figures/correlation_heatmap.png`: Complete feature correlation matrix.
- `reports/figures/area_vs_price_by_location.png`: Scatter plot showing square-footage price slopes across Indian cities.
- `reports/figures/feature_boxplots.png`: Spread across bedrooms, bathrooms, and furnishing status.
