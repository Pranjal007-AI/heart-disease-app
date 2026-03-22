from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(title="Heart Disease Prediction API — KNN Model", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "columns.pkl"), "rb") as f:
    COLUMNS = pickle.load(f)

with open(os.path.join(BASE_DIR, "Scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "heart_disease_knn.pkl"), "rb") as f:
    model = pickle.load(f)

print(f"All artifacts loaded! Columns: {COLUMNS}")


class HeartInput(BaseModel):
    Age:            int   = Field(..., ge=1,    le=120,  example=50)
    RestingBP:      int   = Field(..., ge=50,   le=250,  example=130)
    Cholesterol:    int   = Field(..., ge=0,    le=700,  example=245)
    FastingBS:      int   = Field(..., ge=0,    le=1,    example=0)
    MaxHR:          int   = Field(..., ge=60,   le=250,  example=150)
    Oldpeak:        float = Field(..., ge=-5.0, le=10.0, example=1.5)
    Sex:            str   = Field(..., example="M",      description="M or F")
    ChestPainType:  str   = Field(..., example="ATA",    description="ATA | NAP | TA | ASY")
    RestingECG:     str   = Field(..., example="Normal", description="Normal | ST | LVH")
    ExerciseAngina: str   = Field(..., example="N",      description="Y or N")
    ST_Slope:       str   = Field(..., example="Up",     description="Up | Flat | Down")


class PredictionResponse(BaseModel):
    prediction:    int
    result:        str
    probability:   float
    risk_level:    str
    message:       str


def encode_input(data: HeartInput) -> np.ndarray:
    row = {
        "Age":               data.Age,
        "RestingBP":         data.RestingBP,
        "Cholesterol":       data.Cholesterol,
        "FastingBS":         data.FastingBS,
        "MaxHR":             data.MaxHR,
        "Oldpeak":           data.Oldpeak,
        "Sex_M":             1 if data.Sex.upper() == "M" else 0,
        "ChestPainType_ATA": 1 if data.ChestPainType.upper() == "ATA" else 0,
        "ChestPainType_NAP": 1 if data.ChestPainType.upper() == "NAP" else 0,
        "ChestPainType_TA":  1 if data.ChestPainType.upper() == "TA"  else 0,
        "RestingECG_Normal": 1 if data.RestingECG == "Normal" else 0,
        "RestingECG_ST":     1 if data.RestingECG == "ST"     else 0,
        "ExerciseAngina_Y":  1 if data.ExerciseAngina.upper() == "Y" else 0,
        "ST_Slope_Flat":     1 if data.ST_Slope == "Flat" else 0,
        "ST_Slope_Up":       1 if data.ST_Slope == "Up"   else 0,
    }
    return np.array([[row[col] for col in COLUMNS]], dtype=float)


@app.get("/")
def root():
    return {"message": "Heart Disease KNN API is running!", "features": len(COLUMNS)}

@app.get("/health")
def health():
    return {"status": "ok", "model": "KNN", "n_features": len(COLUMNS)}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartInput):
    try:
        X_raw    = encode_input(data)
        X_scaled = scaler.transform(X_raw)
        pred     = int(model.predict(X_scaled)[0])

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_scaled)[0][pred])
        else:
            prob = 1.0 if pred == 1 else 0.0

        prob_pct = round(prob * 100, 1)

        if pred == 1:
            result     = "Heart Disease Detected"
            risk_level = "High Risk" if prob >= 0.75 else "Moderate Risk"
            message    = "Aapke parameters cardiac risk indicate kar rahe hain. Kripya ek qualified cardiologist se milein aur further tests karaaein."
        else:
            result     = "No Heart Disease Detected"
            risk_level = "Low Risk" if prob >= 0.75 else "Borderline"
            message    = "Aapke parameters normal range mein hain. Swasth jeevanashaili banaye rakhein — regular exercise, balanced diet aur routine checkups."

        return PredictionResponse(
            prediction=pred, result=result,
            probability=prob_pct, risk_level=risk_level, message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
