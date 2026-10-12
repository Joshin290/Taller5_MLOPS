import mlflow
import joblib

# Configura tu URI de MLflow (si estás corriendo MLflow localmente o mediante puerto reenviado)
mlflow.set_tracking_uri("http://localhost:5000") 

MODEL_NAME = "Penguins_RF"
MODEL_VERSION = "1"

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
print(f"Descargando modelo desde: {model_uri}")

# Carga el modelo desde MLflow (lo descarga desde MinIO por debajo)
model = mlflow.pyfunc.load_model(model_uri)

# Extrae el modelo subyacente de Scikit-Learn y guárdalo localmente
joblib.dump(model.unwrap_python_model(), "api/model.pkl")
print("¡Modelo guardado localmente como model.pkl exitosamente!")

