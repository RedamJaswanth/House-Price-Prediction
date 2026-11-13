"""
Streamlit Web Application: Real Estate Valuation & House Price Prediction (Indian Market - INR)
Interactive real-time estimation, model insights, EDA visualization, and explainability.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import json
from PIL import Image

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from predict import HousePricePredictor, format_inr

# Page Configuration
st.set_page_config(
    page_title="AI Indian House Price Valuation System (INR)",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3d59 0%, #17b978 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .metric-sub {
        font-size: 1.15rem;
        opacity: 0.95;
    }
    .metric-label {
        font-size: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return HousePricePredictor()

@st.cache_data
def load_metrics():
    metrics_path = BASE_DIR / "models" / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

def format_crores_lakhs(val: float) -> str:
    if val >= 10000000:
        return f"₹{val/10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val/100000:.2f} Lakhs"
    else:
        return f"₹{val:,.0f}"

predictor = load_predictor()
metrics_meta = load_metrics()

# App Header
st.markdown('<div class="main-title">🏠 AI Indian Real Estate Valuation & House Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Valuation System calibrated for Indian Metropolitan Cities (in Indian Rupees ₹), powered by XGBoost & SHAP</div>', unsafe_allow_html=True)

# Sidebar: Property Feature Inputs
st.sidebar.header("📐 Property Characteristics")

locations = [
    "Bandra West (Mumbai)", 
    "Koramangala (Bengaluru)", 
    "Cyber City (Gurugram)", 
    "Whitefield (Bengaluru)", 
    "Gachibowli (Hyderabad)", 
    "Hinjewadi (Pune)", 
    "Noida Sector 62"
]
location = st.sidebar.selectbox("City / Micro-Market", locations, index=1)

area_sqft = st.sidebar.slider("Super Built-up Area (sq ft)", min_value=650, max_value=4800, value=1850, step=25)

col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    bedrooms = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5, 6], index=2)
with col_b2:
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    floors = st.selectbox("Floor Level", [1, 2, 3, 4], index=1)
with col_f2:
    parking = st.selectbox("Covered Car Parking", [0, 1, 2, 3], index=1)

house_age = st.sidebar.slider("Property Age (Years)", min_value=0, max_value=45, value=4, step=1)

distance_km = st.sidebar.slider("Distance to IT Corridor / City Center (km)", min_value=0.5, max_value=25.0, value=4.0, step=0.5)

school_rating = st.sidebar.slider("Civic Infrastructure / School Rating (1 to 10)", min_value=3, max_value=9, value=8, step=1)

furnishing = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"], index=2)

predict_btn = st.sidebar.button("✨ Calculate Property Valuation (INR)", type="primary", use_container_width=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Valuation Result", 
    "📊 Model Performance & Leaderboard", 
    "📈 Exploratory Data Analysis", 
    "🧠 Explainable AI & SHAP"
])

# ----------------- TAB 1: VALUATION RESULT -----------------
with tab1:
    est_price = predictor.predict_single(
        location=location,
        area_sqft=area_sqft,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        floors=floors,
        house_age_years=house_age,
        parking_spaces=parking,
        distance_city_center_km=distance_km,
        school_rating=school_rating,
        furnishing=furnishing
    )
    
    price_per_sqft = est_price / area_sqft
    lower_bound = est_price * 0.96
    upper_bound = est_price * 1.04
    
    formatted_val = format_crores_lakhs(est_price)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimated Property Market Valuation</div>
        <div class="metric-value">{formatted_val}</div>
        <div class="metric-sub">
            Exact Value: <strong>₹{est_price:,.0f}</strong> | Range (±4%): <strong>{format_crores_lakhs(lower_bound)}</strong> — <strong>{format_crores_lakhs(upper_bound)}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Rate per Sq Ft", f"₹{price_per_sqft:,.0f}/sqft")
    with col_m2:
        st.metric("Property Layout", f"{bedrooms} BHK ({bedrooms+bathrooms} Rooms)")
    with col_m3:
        st.metric("Location / Market", location.split('(')[0].strip())
    with col_m4:
        st.metric("Furnishing Status", furnishing)
        
    st.markdown("---")
    st.subheader("📋 Valuation Breakdown & Indian Mortgage Calculation")
    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.write(f"- **Micro-Market:** {location}")
        st.write(f"- **Super Built-up Area:** {area_sqft:,} sq ft")
        st.write(f"- **Configuration:** {bedrooms} BHK | {bathrooms} Baths | Floor {floors}")
        st.write(f"- **Car Parking:** {parking} Dedicated Reserved Spot(s)")
        st.write(f"- **Property Age:** {house_age} Year(s)")
    with spec_col2:
        st.write(f"- **Proximity to Key Tech Parks:** {distance_km} km")
        st.write(f"- **Infrastructure / School Rating:** {school_rating} / 10")
        st.write(f"- **Furnishing Level:** {furnishing}")
        
        # Indian Home Loan EMI Estimation (20 years @ 8.5% interest, 80% LTV)
        loan_amount = est_price * 0.80
        monthly_rate = (8.5 / 100) / 12
        months = 20 * 12
        emi = loan_amount * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
        st.write(f"- **Estimated Home Loan EMI (80% Loan @ 8.5% for 20 yrs):** **₹{emi:,.0f} / month**")

# ----------------- TAB 2: BENCHMARK LEADERBOARD -----------------
with tab2:
    st.subheader("🏆 Model Comparison & Evaluation Leaderboard (INR Pricing)")
    if metrics_meta:
        leaderboard = metrics_meta.get("leaderboard", {})
        
        rows = []
        for model_name, m in leaderboard.items():
            rows.append({
                "Algorithm": model_name,
                "5-Fold CV R²": f"{m.get('CV_R2_Mean', 0):.4f} ± {m.get('CV_R2_Std', 0):.4f}",
                "Test R²": m.get("R2", 0),
                "Test MAE (₹)": f"₹{m.get('MAE_INR', 0):,.0f} ({m.get('MAE_INR', 0)/1e5:.2f}L)",
                "Test RMSE (₹)": f"₹{m.get('RMSE_INR', 0):,.0f} ({m.get('RMSE_INR', 0)/1e5:.2f}L)",
                "MAPE (%)": f"{m.get('MAPE_pct', 0):.2f}%"
            })
            
        df_leaderboard = pd.DataFrame(rows)
        st.dataframe(df_leaderboard, use_container_width=True)
        
        st.markdown("""
        > **Key Insight:** **XGBoost Regressor** is the clear winner with **Test R² = 0.9825** and a mean absolute percentage error of only **4.32%**, handling Indian market non-linearities such as location premiums in Mumbai/Bengaluru much better than linear baselines.
        """)
        
        chart_path = BASE_DIR / "reports" / "figures" / "model_comparison_r2.png"
        if chart_path.exists():
            st.image(str(chart_path), caption="Model Test R² Score Comparison", use_container_width=True)
            
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            p1 = BASE_DIR / "reports" / "figures" / "actual_vs_predicted.png"
            if p1.exists():
                st.image(str(p1), caption="Actual vs. Predicted Valuation Fit (Crores INR)", use_container_width=True)
        with col_res2:
            p2 = BASE_DIR / "reports" / "figures" / "residuals_distribution.png"
            if p2.exists():
                st.image(str(p2), caption="Residuals Error Curve (Lakhs INR)", use_container_width=True)

# ----------------- TAB 3: EXPLORATORY DATA ANALYSIS -----------------
with tab3:
    st.subheader("🔍 Exploratory Data Analysis & Visual Insights (Indian Real Estate)")
    st.write("Exploration of 6,000 real estate properties across major Indian metropolitan hubs.")
    
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        p_dist = BASE_DIR / "reports" / "figures" / "price_distribution.png"
        if p_dist.exists():
            st.image(str(p_dist), caption="Selling Price Distribution in Crores INR & Log Transformation", use_container_width=True)
    with col_eda2:
        p_corr = BASE_DIR / "reports" / "figures" / "correlation_heatmap.png"
        if p_corr.exists():
            st.image(str(p_corr), caption="Feature Pearson Correlation Matrix", use_container_width=True)
            
    col_eda3, col_eda4 = st.columns(2)
    with col_eda3:
        p_scat = BASE_DIR / "reports" / "figures" / "area_vs_price_by_location.png"
        if p_scat.exists():
            st.image(str(p_scat), caption="Built-up Area vs Price (Crores INR) Across Indian Locations", use_container_width=True)
    with col_eda4:
        p_box = BASE_DIR / "reports" / "figures" / "feature_boxplots.png"
        if p_box.exists():
            st.image(str(p_box), caption="Price Variance across Bedrooms, Bathrooms, and Furnishing", use_container_width=True)

# ----------------- TAB 4: SHAP & EXPLAINABILITY -----------------
with tab4:
    st.subheader("🧠 Explainable AI (XAI) & SHAP Interpretability")
    st.write("Understanding feature attributions for property valuation transparency and fair pricing.")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        p_fi = BASE_DIR / "reports" / "figures" / "feature_importance.png"
        if p_fi.exists():
            st.image(str(p_fi), caption="Top Predictive Features (XGBoost Gain Importance)", use_container_width=True)
    with col_exp2:
        p_shap = BASE_DIR / "reports" / "figures" / "shap_summary.png"
        if p_shap.exists():
            st.image(str(p_shap), caption="SHAP Global Feature Impact Summary (INR Model)", use_container_width=True)
            
    st.markdown("""
    ### Key SHAP Findings for the Indian Real Estate Model:
    - **`Area_sqft` & `Sqft_per_Room`**: The most dominant drivers of overall property cost.
    - **Location Premium**: Locations like *Bandra West (Mumbai)* and *Koramangala (Bengaluru)* generate high positive SHAP values, driving prices into the ₹3 Cr – ₹15 Cr range.
    - **Parking Spaces**: Dedicated parking spots exhibit strong positive marginal contributions due to severe urban parking constraints in Indian metros.
    - **Distance to IT Hubs**: Increases in distance from tech centers lead to negative SHAP shifts, showing the commute penalty.
    """)
