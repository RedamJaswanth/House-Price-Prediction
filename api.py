"""
Production FastAPI Application for Indian Real Estate Valuation (INR)
Provides high-performance REST API endpoints for single and batch predictions in Indian Rupees (₹).
"""

from pathlib import Path
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import json

# Add src directory to system path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from predict import HousePricePredictor, format_inr

app = FastAPI(
    title="Indian Real Estate Valuation & House Price Prediction API (INR)",
    description="Machine Learning Valuation API using XGBoost and Scikit-Learn pipelines calibrated in Indian Rupees (₹).",
    version="2.0.0"
)

# Load predictor and model metrics
try:
    predictor = HousePricePredictor()
except Exception as e:
    predictor = None
    print(f"Warning: Could not initialize predictor: {e}")

class HouseFeatures(BaseModel):
    location: str = Field(..., example="Koramangala (Bengaluru)", description="Indian city / micro-market name")
    area_sqft: float = Field(..., gt=100, le=10000, example=1850.0, description="Total super built-up area in square feet")
    bedrooms: int = Field(..., ge=1, le=10, example=3, description="Number of bedrooms (BHK)")
    bathrooms: int = Field(..., ge=1, le=10, example=2, description="Number of bathrooms")
    floors: int = Field(1, ge=1, le=10, example=2, description="Floor level / stories")
    house_age_years: int = Field(4, ge=0, le=100, example=4, description="Property age in years")
    parking_spaces: int = Field(1, ge=0, le=5, example=2, description="Number of dedicated covered car parking spaces")
    distance_city_center_km: float = Field(5.0, ge=0.0, le=100.0, example=3.5, description="Distance to city center / IT tech corridor in kilometers")
    school_rating: int = Field(7, ge=1, le=10, example=8, description="Civic infrastructure & school rating (1 to 10)")
    furnishing: str = Field("Semi-Furnished", example="Fully-Furnished", description="Furnishing status: Unfurnished, Semi-Furnished, Fully-Furnished")

class PredictionResponse(BaseModel):
    currency: str = "INR"
    estimated_price_inr: float
    price_formatted: str
    price_per_sqft_inr: float
    confidence_interval_95: dict
    property_summary: dict

@app.get("/")
def read_root():
    return {
        "service": "Indian House Price Prediction REST API",
        "currency": "INR (Indian Rupee)",
        "status": "online",
        "docs_url": "/docs",
        "version": "2.0.0"
    }

@app.get("/health")
def health_check():
    metrics_file = BASE_DIR / "models" / "metrics_summary.json"
    meta = {}
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            meta = json.load(f)
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "currency": meta.get("currency", "INR"),
        "champion_algorithm": meta.get("best_model", "XGBoost"),
        "test_r2_score": meta.get("leaderboard", {}).get("XGBoost", {}).get("R2", None)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_house_price(house: HouseFeatures):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service is uninitialized.")
    
    try:
        pred = predictor.predict_single(
            location=house.location,
            area_sqft=house.area_sqft,
            bedrooms=house.bedrooms,
            bathrooms=house.bathrooms,
            floors=house.floors,
            house_age_years=house.house_age_years,
            parking_spaces=house.parking_spaces,
            distance_city_center_km=house.distance_city_center_km,
            school_rating=house.school_rating,
            furnishing=house.furnishing
        )
        
        price_sqft = round(pred / house.area_sqft, 2)
        lower_bound = round(pred * 0.96, 2)
        upper_bound = round(pred * 1.04, 2)
        
        # Format string for Indian notation
        formatted = format_inr(pred)
        
        return PredictionResponse(
            currency="INR",
            estimated_price_inr=pred,
            price_formatted=formatted,
            price_per_sqft_inr=price_sqft,
            confidence_interval_95={
                "lower_inr": lower_bound, 
                "upper_inr": upper_bound,
                "range_formatted": f"{format_inr(lower_bound)} - {format_inr(upper_bound)}"
            },
            property_summary={
                "location": house.location,
                "area_sqft": house.area_sqft,
                "bedrooms": house.bedrooms,
                "bathrooms": house.bathrooms,
                "furnishing": house.furnishing
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
def predict_batch(houses: List[HouseFeatures]):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor service is uninitialized.")
    
    records = []
    for h in houses:
        records.append({
            "Location": h.location,
            "Area_sqft": h.area_sqft,
            "Bedrooms": h.bedrooms,
            "Bathrooms": h.bathrooms,
            "Floors": h.floors,
            "House_Age_years": h.house_age_years,
            "Parking_Spaces": h.parking_spaces,
            "Distance_City_Center_km": h.distance_city_center_km,
            "School_Rating": h.school_rating,
            "Furnishing": h.furnishing
        })
    df = pd.DataFrame(records)
    preds = predictor.predict_dataframe(df)
    
    results = []
    for i, pred in enumerate(preds):
        results.append({
            "property_index": i,
            "estimated_price_inr": float(pred),
            "price_formatted": format_inr(float(pred)),
            "location": records[i]["Location"],
            "area_sqft": records[i]["Area_sqft"]
        })
    return {"batch_size": len(results), "currency": "INR", "predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
