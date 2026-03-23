from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "columns.pkl"), "rb") as f:
    COLUMNS = pickle.load(f)
with open(os.path.join(BASE_DIR, "Scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, "heart_disease_knn.pkl"), "rb") as f:
    model = pickle.load(f)

print(f"All artifacts loaded! Columns: {COLUMNS}")


def encode_input(data):
    row = {
        "Age":               int(data["Age"]),
        "RestingBP":         int(data["RestingBP"]),
        "Cholesterol":       int(data["Cholesterol"]),
        "FastingBS":         int(data["FastingBS"]),
        "MaxHR":             int(data["MaxHR"]),
        "Oldpeak":           float(data["Oldpeak"]),
        "Sex_M":             1 if str(data["Sex"]).upper() == "M" else 0,
        "ChestPainType_ATA": 1 if str(data["ChestPainType"]).upper() == "ATA" else 0,
        "ChestPainType_NAP": 1 if str(data["ChestPainType"]).upper() == "NAP" else 0,
        "ChestPainType_TA":  1 if str(data["ChestPainType"]).upper() == "TA"  else 0,
        "RestingECG_Normal": 1 if data["RestingECG"] == "Normal" else 0,
        "RestingECG_ST":     1 if data["RestingECG"] == "ST"     else 0,
        "ExerciseAngina_Y":  1 if str(data["ExerciseAngina"]).upper() == "Y" else 0,
        "ST_Slope_Flat":     1 if data["ST_Slope"] == "Flat" else 0,
        "ST_Slope_Up":       1 if data["ST_Slope"] == "Up"   else 0,
    }
    return np.array([[row[col] for col in COLUMNS]], dtype=float)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        body = json.dumps({"status": "ok", "message": "Heart Disease API running!"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
                X_raw = encode_input(data)
                X_scaled = scaler.transform(X_raw)
                pred = int(model.predict(X_scaled)[0])
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_scaled)[0][pred])
                else:
                    prob = 1.0 if pred == 1 else 0.0
                prob_pct = round(prob * 100, 1)
                if pred == 1:
                    result = "Heart Disease Detected"
                    risk_level = "High Risk" if prob >= 0.75 else "Moderate Risk"
                    message = "Aapke parameters cardiac risk indicate kar rahe hain. Kripya ek qualified cardiologist se milein."
                else:
                    result = "No Heart Disease Detected"
                    risk_level = "Low Risk" if prob >= 0.75 else "Borderline"
                    message = "Aapke parameters normal range mein hain. Swasth jeevanashaili banaye rakhein."
                body = json.dumps({"prediction": pred, "result": result, "probability": prob_pct, "risk_level": risk_level, "message": message}).encode()
                self.send_response(200)
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
        else:
            body = json.dumps({"error": "Not found"}).encode()
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_cors()
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Server starting on port {port}...")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
