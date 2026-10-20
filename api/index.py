from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Fertilizer Recommendation API")

# Load model dan kolom saat server menyala
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "fertilizer_rf_model.pkl")
columns_path = os.path.join(BASE_DIR, "..", "model_columns.pkl")

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

# Definisikan format input HANYA berdasarkan parameter angka dari Anda
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
        # Mapping parameter angka, dan berikan nilai default untuk kategori (misal: 'Loamy', 'Wheat', 'Kharif')
        # agar struktur 39 kolom One-Hot Encoding tetap terpenuhi dengan baik.
        mapped_data = {
            'Soil_Type': 'Loamy',          # Nilai default kategori tanah
            'Crop_Type': 'Rice',          # Nilai default jenis tanaman
            'Season': 'Kharif',           # Nilai default musim
            'Soil_pH': data.PH,                     
            'Nitrogen_Level': data.Nitrogen,        
            'Phosphorus_Level': data.Phosphorous,   
            'Potassium_Level': data.Potassium,      
            'Soil_Moisture': data.Moisture,         
            'Temperature': data.Temperature,
            'Rainfall': data.Rainfall,
        }
        
        # 1. Ubah dictionary menjadi DataFrame
        df_input = pd.DataFrame([mapped_data])
        
        # 2. Lakukan One-Hot Encoding
        df_encoded = pd.get_dummies(df_input)
        
        # 3. Reindex agar kolomnya pas 39 buah sesuai model_columns.pkl
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 4. Lakukan Prediksi
        prediction = model.predict(df_final)
        
        return {
            "status": "success",
            "recommended_fertilizer": prediction[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))