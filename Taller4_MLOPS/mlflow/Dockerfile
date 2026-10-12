FROM python:3.9-slim

# ==========================================================
# Variables de MinIO / S3
# ==========================================================
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
    AWS_ACCESS_KEY_ID=admin \
    AWS_SECRET_ACCESS_KEY=supersecret \
    AWS_DEFAULT_REGION=us-east-1

# ==========================================================
# Instalar MLflow y dependencias
# ==========================================================
RUN pip install --no-cache-dir \
    mlflow \
    boto3 \
    psycopg2-binary

# ==========================================================
# Directorio de trabajo
# ==========================================================
WORKDIR /app

# ==========================================================
# Puerto MLflow
# ==========================================================
EXPOSE 5000

# ==========================================================
# Servidor MLflow
# ==========================================================
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://admin:supersecret@db:5432/mlflowdb", \
     "--default-artifact-root", "s3://mlflows3/artifacts", \
     "--serve-artifacts"]