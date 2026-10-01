from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd
import os

app = FastAPI(title="API ML on Vercel")

# Load model Random Forest dari root directory
model_path = os.path.join(os.path.dirname(__file__), "../model_rf.pkl")
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None

class DataLahan(BaseModel):
    Temperature: float
    Moisture: float
    Rainfall: float
    PH: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Carbon: float

    # Validator pintar: otomatis mengonversi string angka dari Power Automate menjadi float
    @field_validator('*', mode='before')
    @classmethod
    def parse_numeric(cls, v):
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                raise ValueError(f"Nilai '{v}' gagal dikonversi menjadi angka.")
        return v

@app.get("/")
def home():
    return {
        "pesan": "Selamat datang di API ML!",
        "status": "Server Aktif dan Siap Digunakan"
    }

@app.post("/prediksi")
def prediksi_pupuk(data: DataLahan):
    if model is None:
        raise HTTPException(status_code=500, detail="File model ML tidak ditemukan di server.")
    
    # Menggunakan kompatibilitas model_dump (Pydantic v2) atau dict (Pydantic v1)
    input_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Eksekusi prediksi model Random Forest
    hasil_prediksi = model.predict(input_df)
    
    return {
        "status": "sukses",
        "rekomendasi_pupuk": str(hasil_prediksi[0])
    }