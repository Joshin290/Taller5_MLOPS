from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_NAME = "Penguins_RF"
MODEL_VERSION = "7"
# Ruta donde estará guardado el modelo dentro del contenedor Docker
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Penguins Classification API",
    description="""
API para predecir la especie de un pingüino usando un modelo Random Forest local.
""",
    version="1.0.0"
)


# ============================================================
# CARGAR MODELO LOCALMENTE
# ============================================================

model = None


@app.on_event("startup")
def load_model():
    global model

    print("======================================")
    print("CARGANDO MODELO DESDE ARCHIVO LOCAL")
    print("======================================")
    print(f"Modelo: {MODEL_NAME}")
    print(f"Ruta: {MODEL_PATH}")

    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = None

    print("======================================")


# ============================================================
# MODELO DE ENTRADA
# ============================================================

class PenguinInput(BaseModel):
    island: str
    bill_length_mm: float | None = None
    bill_depth_mm: float | None = None
    flipper_length_mm: float | None = None
    body_mass_g: float | None = None
    sex: str | None = None


# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Penguins Classification API",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    if model is None:
        return {
            "status": "error",
            "model_loaded": False
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post("/predict")
def predict(data: PenguinInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está cargado"
        )

    try:
        input_data = pd.DataFrame([{
            "bill_length_mm": data.bill_length_mm,
            "bill_depth_mm": data.bill_depth_mm,
            "flipper_length_mm": data.flipper_length_mm,
            "body_mass_g": data.body_mass_g,
            "island": data.island,
            "sex": data.sex
        }])

        prediction = model.predict(input_data)

        return {
            "prediction": str(prediction[0]),
            "model": MODEL_NAME,
            "version": MODEL_VERSION
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )