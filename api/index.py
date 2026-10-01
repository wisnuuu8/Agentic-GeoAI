from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="API ML on Vercel")

# Load model Random Forest dari root directory
model_path = os.path.join(os.path.dirname(__file__), "../model_rf.pkl")
model = joblib.load(model_path)

class DataLahan(BaseModel):
    Temperature: float
    Moisture: float
    Rainfall: float
    PH: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Carbon: float

@app.post("/prediksi")
def prediksi_pupuk(data: DataLahan):
    input_df = pd.DataFrame([data.dict()])
    hasil_prediksi = model.predict(input_df)
    
    return {
        "status": "sukses",
        "rekomendasi_pupuk": hasil_prediksi[0]
    }