"""
Inference Module for House Price Prediction (INR Currency)
Provides single-sample and batch prediction functions with input validation and Indian currency formatting (Lakhs / Crores).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Need to import FeatureEngineer so unpickling works seamlessly
from preprocess import FeatureEngineer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

def format_inr(amount: float) -> str:
    """Formats numeric price into Indian numbering format (Lakhs / Crores)."""
    if amount >= 10000000: # 1 Crore
        cr = amount / 10000000
        return f"Rs. {cr:.2f} Crores (Rs. {amount:,.0f})"
    elif amount >= 100000: # 1 Lakh
        lakhs = amount / 100000
        return f"Rs. {lakhs:.2f} Lakhs (Rs. {amount:,.0f})"
    else:
        return f"Rs. {amount:,.0f}"

class HousePricePredictor:
    def __init__(self, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH):
        if not Path(model_path).exists() or not Path(preprocessor_path).exists():
            raise FileNotFoundError("Model or preprocessor artifacts not found. Please train models first.")
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict_single(self, 
                       location: str, 
                       area_sqft: float, 
                       bedrooms: int, 
                       bathrooms: int, 
                       floors: int = 1,
                       house_age_years: int = 5, 
                       parking_spaces: int = 1, 
                       distance_city_center_km: float = 5.0, 
                       school_rating: int = 7, 
                       furnishing: str = "Semi-Furnished") -> float:
        
        input_data = pd.DataFrame([{
            "Location": location,
            "Area_sqft": area_sqft,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "House_Age_years": house_age_years,
            "Parking_Spaces": parking_spaces,
            "Distance_City_Center_km": distance_city_center_km,
            "School_Rating": school_rating,
            "Furnishing": furnishing
        }])
        
        transformed = self.preprocessor.transform(input_data)
        prediction = float(self.model.predict(transformed)[0])
        return round(max(0.0, prediction), 2)

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(df)
        preds = self.model.predict(transformed)
        return np.maximum(0.0, np.round(preds, 2))

if __name__ == "__main__":
    predictor = HousePricePredictor()
    
    # Test sample in Bangalore (Koramangala)
    sample_house = {
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
    }
    
    pred_val = predictor.predict_single(**sample_house)
    print("\n--- Indian Real Estate Inference Test ---")
    print(f"Sample House Characteristics: {sample_house}")
    print(f"Estimated Market Valuation: {format_inr(pred_val)}")
