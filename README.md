# 🫀 Cardiac Risk Predictor
### by Pranjal Parashar


## 🔗 Live Demo
👉 Try the Web App **(https://cardiac-risk-predictor-pranjal.netlify.app)**

> Backend API: https://heart-disease-app-1974.onrender.com





> KNN-based Heart Disease Prediction Web App — FastAPI Backend + 3D HTML Frontend

---

## 🧠 Project Overview

This is Ai powered Webapp that's work on a  **K-Nearest Neighbors (KNN)** machine learning model use that use to predict the 
heart disease .

The user enters their health parameters using sliders and dropdowns, and the model instantly predicts whether they have heart disease or not — along with a confidence score.

---

## ⚙️ Tech Stack

| Layer | Technology | work |
|-------|-----------|------|
| Frontend | HTML + CSS + JavaScript | 3D UI, sliders, animated heart, result display |
| Backend | FastAPI (Python) | REST API, model loading, prediction |
| ML Model | KNN (scikit-learn) | Heart disease binary classification |
| Deployment | Render.com + Netlify | Free cloud hosting — live link |

---

## 📁 Folder Structure

```
heart-disease-app/
├── backend/
│   ├── main.py                  ← FastAPI server (API endpoints)
│   ├── requirements.txt         ← Python dependencies
│   ├── columns.pkl              ← Feature order (15 features)
│   ├── Scaler.pkl               ← StandardScaler (trained)
│   └── heart_disease_knn.pkl    ← Trained KNN model
└── frontend/
    └── index.html               ← Poora UI (3D heart, sliders, form)
```

---

## 📊 Model Features (15 Total)

| Feature | Input Type | Description |
|---------|-----------|-------------|
| Age | Slider (1–100) | Patient ki umar (years) |
| RestingBP | Slider (80–220) | Resting blood pressure (mm Hg) |
| Cholesterol | Slider (0–600) | Serum cholesterol (mg/dl) |
| FastingBS | Dropdown | Fasting blood sugar > 120 mg/dl (0/1) |
| MaxHR | Slider (60–220) | Maximum heart rate achieved (bpm) |
| Oldpeak | Slider (-2 to 8) | ST depression (exercise vs rest) |
| Sex | Dropdown | M (Male) ya F (Female) |
| ChestPainType | Dropdown | ASY / ATA / NAP / TA |
| RestingECG | Dropdown | Normal / ST / LVH |
| ExerciseAngina | Dropdown | Y (Yes) ya N (No) |
| ST_Slope | Dropdown | Up / Flat / Down |

> **Note:** Categorical fields are internally one-hot encoded — the backend handles this automatically.

---

## 🖥️ Local Setup

### Step 1 — DO CLONE
```bash
git clone https://github.com/APKA_USERNAME/heart-disease-app.git
cd heart-disease-app
```

### Step 2 — Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
```

### Step 3 — Start The Server
```bash
uvicorn main:app --reload --port 8000
```
✅ API Docs: http://127.0.0.1:8000/docs

### Step 4 — Now Open The Front-End`
Open `frontend/index.html` in your browser by double-clicking it.

---

## 🌐 Generate the Live Link (Free Deployment)

```
GitHub → Render (backend) → index.html update → Netlify (frontend) = 🔗 LIVE LINK
```

### 🔵 Step 1 — Open the Github
```bash
git init
git add .
git commit -m "heart disease predictor by Pranjal Parashar"
git remote add origin https://github.com/APKA_USERNAME/heart-disease-app.git
git push -u origin main
```

### 🟠 Step 2 — Deploy On Render.com
1. [render.com](https://render.com) → GitHub sign up
2. **New +** → **Web Service** → connect repo
3. :fill the settings

| Setting | Value |
|---------|-------|
| Root Directory | `backend` |
| Runtime | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

4. Deploy this → milega: `https://heart-disease-xxxx.onrender.com` ✅

### 🟡 Step 3 — Update the URL in index.html
`frontend/index.html` Update this:
```js
// Ye line dhundho aur Render URL daal do
const API = "https://heart-disease-xxxx.onrender.com";
```
Then push:
```bash
git add . && git commit -m "update API URL" && git push
```

### 🟢 Step 4 — Deploy the Front-End on the Netlify
1. [netlify.com](https://netlify.com) → Sign Up
2. Dashboard  `frontend/` folder **drag & drop** do
3. IN 30 second ready:

🔗 **`https://random-name.netlify.app`** — that's your live link!

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/predict` | Prediction endpoint |

### POST `/predict` — Request
```json
{
  "Age": 52, "Sex": "M", "RestingBP": 130,
  "Cholesterol": 245, "FastingBS": 0, "MaxHR": 150,
  "ChestPainType": "ASY", "RestingECG": "Normal",
  "ExerciseAngina": "N", "Oldpeak": 1.5, "ST_Slope": "Up"
}
```

### Response
```json
{
  "prediction": 1,
  "result": "Heart Disease Detected",
  "probability": 87.3,
  "risk_level": "High Risk",
  "message": "Kripya turant doctor se milein..."
}


---

*Made with ❤️ by Pranjal Parashar*
