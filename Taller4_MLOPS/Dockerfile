FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias necesarias (incluyendo joblib y mlflow)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    mlflow==2.22.0 \
    scikit-learn==1.6.1 \
    numpy==1.26.4 \
    pandas==2.0.3 

# Copiar todo el proyecto al contenedor
COPY api/model/ /app/model/

COPY api/main.py /app/main.py

ENV MODEL_PATH=/app/model

# Ejecutar el script para descargar el modelo desde MinIO/MLflow durante el build

EXPOSE 8010

# Ejecutar la API apuntando al archivo main dentro de la carpeta api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]