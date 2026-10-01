# api_model.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Inisialisasi Aplikasi API
app = FastAPI(title="API ML", description="API untuk Rekomendasi Pupuk berbasis Random Forest")

# 2. Load Otak Model (File .pkl Ndan)
print("Loading model...")
model = joblib.load('model_rf.pkl')
print("Model berhasil di-load!")

# 3. Format Input Data (Harus sama persis dengan 8 kolcom fitur training tadi)
class DataLahan(BaseModel):
    Temperature: float
    Moisture: float
    Rainfall: float
    PH: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Carbon: float

# 4. Membuat Endpoint POST untuk Prediksi
@app.post("/prediksi")
def prediksi_pupuk(data: DataLahan):
    # Mengubah data dari format JSON/Pydantic menjadi DataFrame Pandas 
    # agar dikenali oleh model Random Forest
    input_df = pd.DataFrame([data.dict()])
    
    # Model menebak hasilnya
    hasil_prediksi = model.predict(input_df)
    
    # Mengembalikan hasil tebakan ke dalam format JSON (bisa dibaca Power Automate)
    return {
        "status": "sukses",
        "rekomendasi_pupuk": hasil_prediksi[0]
    }