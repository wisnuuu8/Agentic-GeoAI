from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Fertilizer Recommendation API")

# PERBAIKAN JALUR FILE: Mengarah ke folder 'models'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "fertilizer_rf_model.pkl")
columns_path = os.path.join(BASE_DIR, "..", "models", "model_columns.pkl")

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

# (Lanjutan kode di bawahnya tetap sama persis seperti sebelumnya)
class FarmInput(BaseModel):
    Temperature: float
    Moisture: float
    Rainfall: float = 0.0
    PH: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Carbon: float = 0.0

@app.get("/")
def home():
    return {"message": "Fertilizer AI API is running successfully!"}

@app.post("/predict")
def predict_fertilizer(data: FarmInput):
    try:
        mapped_data = {
            'Soil_Type': 'Loamy',
            'Crop_Type': 'Rice',
            'Season': 'Kharif',
            'Soil_pH': data.PH,                     
            'Nitrogen_Level': data.Nitrogen,        
            'Phosphorus_Level': data.Phosphorous,   
            'Potassium_Level': data.Potassium,      
            'Soil_Moisture': data.Moisture,         
            'Temperature': data.Temperature,
            'Rainfall': data.Rainfall,
        }
        
        df_input = pd.DataFrame([mapped_data])
        df_encoded = pd.get_dummies(df_input)
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        prediction = model.predict(df_final)
        
        return {
            "status": "success",
            "recommended_fertilizer": prediction[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))