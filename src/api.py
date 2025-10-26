from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="MLOps FastAPI Model")

model = joblib.load("src/model.pkl")

@app.get("/")
def home():
    return {"message": "API MLOps is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = int(model.predict(df)[0])
    return {"prediction": prediction}
