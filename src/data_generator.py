"""
Data Generation Module for House Price Prediction (Indian Real Estate Market - INR)
Generates a realistic housing dataset with Indian metropolitan locations, realistic per-sqft pricing (in ₹), non-linearities, and market dynamics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_housing_data(n_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    locations = [
        "Bandra West (Mumbai)", 
        "Koramangala (Bengaluru)", 
        "Cyber City (Gurugram)", 
        "Whitefield (Bengaluru)", 
        "Gachibowli (Hyderabad)", 
        "Hinjewadi (Pune)", 
        "Noida Sector 62"
    ]
    
    # Base rate in ₹ (INR) per sqft
    location_rate_per_sqft = {
        "Bandra West (Mumbai)": 32000,
        "Koramangala (Bengaluru)": 15000,
        "Cyber City (Gurugram)": 13500,
        "Whitefield (Bengaluru)": 9500,
        "Gachibowli (Hyderabad)": 9000,
        "Hinjewadi (Pune)": 7500,
        "Noida Sector 62": 6800
    }
    
    furnishing_status = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
    furnishing_bonus_inr = {
        "Unfurnished": 0,
        "Semi-Furnished": 350000,    # ₹3.5 Lakhs
        "Fully-Furnished": 850000     # ₹8.5 Lakhs
    }

    # Location distribution
    loc_choices = np.random.choice(
        locations, 
        size=n_samples, 
        p=[0.14, 0.16, 0.16, 0.18, 0.15, 0.11, 0.10]
    )
    
    # Built-up Area between 650 sqft and 4500 sqft
    area = np.clip(np.random.lognormal(mean=7.5, sigma=0.45, size=n_samples), 650, 4800).astype(int)
    
    # Bedrooms correlated with area
    bedrooms = np.clip(np.round((area / 750) + np.random.normal(0, 0.6, n_samples)), 1, 6).astype(int)
    
    # Bathrooms correlated with bedrooms
    bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.5, n_samples)), 1, 5).astype(int)
    
    # Age of property (0 to 45 years)
    house_age = np.random.randint(0, 45, size=n_samples)
    
    # Dedicated parking spots (0 to 3)
    parking = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.15, 0.50, 0.28, 0.07])
    
    # Floor level / stories (1 to 4)
    floors = np.where(
        area > 2200, 
        np.random.choice([2, 3, 4], size=n_samples, p=[0.50, 0.35, 0.15]), 
        np.random.choice([1, 2], size=n_samples, p=[0.75, 0.25])
    )
    
    # Distance to city center / IT tech corridor (km)
    dist_base = {
        "Bandra West (Mumbai)": (2.0, 1.2),
        "Koramangala (Bengaluru)": (4.0, 1.5),
        "Cyber City (Gurugram)": (5.0, 2.0),
        "Whitefield (Bengaluru)": (14.0, 3.5),
        "Gachibowli (Hyderabad)": (12.0, 3.0),
        "Hinjewadi (Pune)": (15.0, 4.0),
        "Noida Sector 62": (16.0, 4.5)
    }
    distance_km = np.array([
        max(0.5, round(np.random.normal(dist_base[loc][0], dist_base[loc][1]), 1))
        for loc in loc_choices
    ])
    
    # Local school / infrastructure rating (3 to 10)
    school_rating = np.random.randint(3, 10, size=n_samples)
    
    # Furnishing
    furnishing = np.random.choice(furnishing_status, size=n_samples, p=[0.35, 0.45, 0.20])
    
    # Price Calculation in INR (₹)
    prices = []
    for i in range(n_samples):
        loc = loc_choices[i]
        sqft = area[i]
        beds = bedrooms[i]
        baths = bathrooms[i]
        age = house_age[i]
        park = parking[i]
        flr = floors[i]
        dist = distance_km[i]
        sch = school_rating[i]
        furn = furnishing[i]
        
        # Base price = Area * Location Rate
        base_rate = location_rate_per_sqft[loc]
        price = sqft * base_rate
        
        # Bedroom & Bathroom value
        price += beds * 280000   # ₹2.8 Lakhs per bedroom
        price += baths * 180000   # ₹1.8 Lakhs per bathroom
        
        # Parking amenity bonus
        price += park * 350000   # ₹3.5 Lakhs per car park
        
        # Floor level bonus (e.g. penthouses / upper levels)
        price += (flr - 1) * 150000
        
        # School / civic infrastructure bonus
        price += (sch - 5) * 120000
        
        # Furnishing addition
        price += furnishing_bonus_inr[furn]
        
        # Distance discount (proximity to IT Hubs / Downtowns commands a premium)
        price -= dist * 45000
        
        # Age depreciation: newer constructions have premium, older properties depreciate
        depreciation = (age ** 0.85) * 45000
        price -= depreciation
        
        # Luxury multiplier for large properties in prime areas
        if sqft > 2500 and loc in ["Bandra West (Mumbai)", "Koramangala (Bengaluru)"]:
            price *= 1.18
            
        # Add realistic real estate market variance (~4% Gaussian noise)
        noise = np.random.normal(0, price * 0.04)
        final_price = max(2500000, price + noise) # Min price ₹25 Lakhs
        prices.append(round(final_price, -3)) # Round to nearest ₹1,000
        
    df = pd.DataFrame({
        "Location": loc_choices,
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "House_Age_years": house_age,
        "Parking_Spaces": parking,
        "Distance_City_Center_km": distance_km,
        "School_Rating": school_rating,
        "Furnishing": furnishing,
        "Price_INR": prices
    })
    
    return df

if __name__ == "__main__":
    out_dir = Path("D:/House_Price_Prediction/data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "housing_data.csv"
    
    print("Generating synthetic real estate dataset with INR pricing...")
    df = generate_housing_data(n_samples=6000, random_state=42)
    df.to_csv(out_file, index=False)
    print(f"Dataset saved to: {out_file}")
    print(f"Shape: {df.shape}")
    print("\nSample records:")
    print(df.head())
    print("\nPrice summary in INR:")
    min_val = df['Price_INR'].min()
    mean_val = df['Price_INR'].mean()
    max_val = df['Price_INR'].max()
    print(f"Min: INR {min_val:,.0f} ({min_val/1e5:.2f} Lakhs)")
    print(f"Mean: INR {mean_val:,.0f} ({mean_val/1e7:.2f} Crores)")
    print(f"Max: INR {max_val:,.0f} ({max_val/1e7:.2f} Crores)")
