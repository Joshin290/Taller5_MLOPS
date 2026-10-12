
## Taller MLflow

- Cree una instancia de una base de datos (dedicada para metadata de MLflow)

- Cree una instancia de MLflow

- Cree una instancia de MINio (dedicada para MLflow)

- Cree una instancia de JupyterLab

    - Cree un notebook que permita realizar entrenamiento de un modelo, realize multiples ejecuciones a modo de experimentacion (al menos 20, con variaciones de hyperparámetros), todo debe estar registrado en MLflow.

    - Los datos deben existir en una base de datos, los datos procesados deben estar en base de datos. (la base de datos debe ser diferente a la usada para mlflow)

    - Los modelos deben estar registrados en MLflow.

- Cree una API que permita realizar inferencia usando el modelo entrenado, tomando el modelo **mediante** MLflow.


# Desarrollo del ejercicio

# Taller 4 - MLOps

## Desarrollo del ejercicio

En este taller se desarrolla un flujo completo de **Machine Learning Operations (MLOps)** orientado a la clasificación de especies de pingüinos. Para ello, se integran diferentes herramientas que permiten gestionar los datos, realizar el entrenamiento y evaluación de modelos, registrar los experimentos, almacenar los modelos y finalmente exponer el modelo seleccionado mediante una API REST para realizar inferencias.

Para el ejercicio se utiliza el dataset **Palmer Penguins**, cuyos datos son almacenados en una base de datos PostgreSQL independiente de la base de datos utilizada por MLflow. Sobre estos datos se construye un modelo de clasificación utilizando el algoritmo **Random Forest**, evaluando 30 configuraciones diferentes de hiperparámetros.

El proyecto integra:

- Docker y Docker Compose
- PostgreSQL
- JupyterLab
- Scikit-learn
- MLflow
- MinIO
- FastAPI
- Python

El flujo general implementado es:

```text
Dataset
   ↓
PostgreSQL - Datos de Penguins
   ↓
JupyterLab
   ↓
Preprocesamiento
   ↓
Entrenamiento de 30 configuraciones
   ↓
MLflow Tracking
   ↓
Selección del mejor modelo
   ↓
MLflow Model Registry
   ↓
MinIO - Artifacts
   ↓
FastAPI
   ↓
Inferencia
```

---

## Objetivo

El objetivo del taller es implementar una arquitectura básica de MLOps que permita gestionar el ciclo de vida de un modelo de Machine Learning.

Para ello se busca:

1. Almacenar los datos del problema en una base de datos PostgreSQL.
2. Utilizar una segunda base de datos PostgreSQL para los metadatos de MLflow.
3. Utilizar MLflow para registrar experimentos, parámetros, métricas y modelos.
4. Utilizar MinIO como almacenamiento de artifacts compatible con S3.
5. Entrenar múltiples configuraciones de un modelo Random Forest.
6. Comparar las diferentes configuraciones mediante la métrica `Validation Accuracy`.
7. Seleccionar el modelo con mejor desempeño.
8. Registrar el modelo seleccionado en MLflow Model Registry.
9. Crear una API REST que obtenga el modelo directamente desde MLflow.
10. Realizar inferencias mediante la API.

---

# Estructura del proyecto

```text
Taller4_MLOPS/
│
├── api/
│   ├── Dockerfile
│   └── main.py
│
├── Data/
│   └── penguins.csv
│
├── mlflow/
│   └── Dockerfile
│
├── notebooks/
│   └── modelo.ipynb
│
├── scripts/
│   └── load_data.py
│
├── docker-compose.yaml
│
└── README.md
```

### Descripción de los archivos

| Archivo / carpeta | Descripción |
|---|---|
| `api/` | Contiene la API REST utilizada para realizar inferencias |
| `api/Dockerfile` | Define la imagen Docker de la API |
| `api/main.py` | Implementa los endpoints de FastAPI y carga el modelo desde MLflow |
| `Data/` | Contiene el dataset utilizado en el proyecto |
| `Data/penguins.csv` | Dataset Palmer Penguins |
| `mlflow/` | Contiene la configuración de la imagen de MLflow |
| `mlflow/Dockerfile` | Define el servidor MLflow y su conexión con PostgreSQL y MinIO |
| `notebooks/` | Contiene los notebooks utilizados durante el desarrollo |
| `notebooks/modelo.ipynb` | Notebook de preparación, entrenamiento, evaluación y registro del modelo |
| `scripts/` | Contiene scripts auxiliares |
| `scripts/load_data.py` | Carga el dataset desde CSV hacia PostgreSQL |
| `docker-compose.yaml` | Define y orquesta todos los servicios del proyecto |
| `README.md` | Documentación del proyecto |

---

# Dataset Palmer Penguins

El proyecto utiliza el dataset **Palmer Penguins**, almacenado localmente en:

```text
Data/penguins.csv
```

El dataset contiene información relacionada con diferentes características físicas de los pingüinos y variables categóricas.

Las columnas del dataset son:

```text
rowid
species
island
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
sex
year
```

La variable objetivo es:

```text
species
```

Las especies que se buscan clasificar son:

```text
Adelie
Chinstrap
Gentoo
```

Las características utilizadas para el entrenamiento son:

### Variables numéricas

```text
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
```

### Variables categóricas

```text
island
sex
```

---

# Script `load_data.py`

El archivo:

```text
scripts/load_data.py
```

se encarga de cargar automáticamente el dataset `penguins.csv` en PostgreSQL.

El script utiliza:

```text
pandas
SQLAlchemy
psycopg2
```

La conexión utilizada es:

```text
postgresql+psycopg2://admin:supersecret@data_db:5432/penguinsdb
```

El archivo CSV se lee desde:

```text
/Data/penguins.csv
```

y la información se almacena en la tabla:

```text
penguins
```

El script también implementa un mecanismo de espera para PostgreSQL. Antes de cargar los datos, realiza varios intentos de conexión hasta comprobar que la base de datos está disponible.

El servicio `data_loader` se ejecuta automáticamente mediante Docker Compose y depende de que `data_db` se encuentre saludable (healthy).

El flujo de carga es:

```text
penguins.csv
     ↓
data_loader
     ↓
PostgreSQL
penguinsdb
     ↓
tabla penguins
```

---

# PostgreSQL

El proyecto utiliza **dos instancias independientes de PostgreSQL**.

Esta separación permite mantener los datos del problema separados de los metadatos utilizados por MLflow.

## PostgreSQL para MLflow

El servicio:

```text
db
```

utiliza la imagen:

```text
postgres:15-alpine
```

y crea la base de datos:

```text
mlflowdb
```

Esta base de datos almacena los metadatos utilizados por MLflow, entre ellos:

- Experimentos.
- Runs.
- Parámetros.
- Métricas.
- Información de los modelos registrados.

El volumen utilizado es:

```text
postgres_data
```

La conexión interna utilizada por MLflow es:

```text
postgresql://admin:supersecret@db:5432/mlflowdb
```

El puerto publicado en el host es:

```text
8030
```

---

## PostgreSQL para los datos de Penguins

El servicio:

```text
data_db
```

utiliza la misma imagen:

```text
postgres:15-alpine
```

pero utiliza una base de datos independiente:

```text
penguinsdb
```

Esta base de datos almacena los datos del dataset Palmer Penguins.

El volumen utilizado es:

```text
penguins_data
```

La conexión utilizada desde el notebook es:

```text
postgresql+psycopg2://admin:supersecret@data_db:5432/penguinsdb
```

El puerto publicado en el host es:

```text
8031
```

La separación entre las bases de datos queda representada de la siguiente forma:

```text
                 PostgreSQL
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     mlflowdb             penguinsdb
          │                   │
          │                   │
     Metadatos              Datos
     de MLflow             Penguins
```

---

# JupyterLab

JupyterLab se utiliza como entorno de desarrollo para realizar el proceso completo de Machine Learning.

El servicio utiliza la imagen:

```text
jupyter/scipy-notebook:latest
```

El notebook principal es:

```text
notebooks/modelo.ipynb
```

El notebook se conecta a MLflow mediante:

```text
http://mlflow:5000
```

También se conecta a PostgreSQL para obtener los datos y a MinIO para trabajar con el almacenamiento de artifacts.

El puerto utilizado es:

```text
8888
```

Acceso:

```text
http://localhost:8888
```

### Nota: 

Es posible que, al acceder a JupyterLab mediante el puerto configurado, se solicite un token de autenticación. Para obtenerlo, ejecute en la terminal el comando docker logs Jupyter. En los registros, busque una línea similar a:

 http://ec24453d91e8:8888/lab?token=ad583d1b38d72f84babf1b6177e040dfab31a6c6fc59b8b1. 
 
 Copie el valor que aparece después de token= y utilícelo en la página de inicio de JupyterLab. Luego acceda a http://localhost:8888 e introduzca el token para ingresar a JupyterLab.

---

# Notebook `modelo.ipynb`

El archivo:

```text
notebooks/modelo.ipynb
```

contiene el proceso completo de preparación, entrenamiento, evaluación y registro del modelo.

El flujo implementado en el notebook se divide en diferentes etapas.

---

## 1. Instalación de dependencias

El notebook instala las principales dependencias utilizadas durante el desarrollo:

```text
numba >= 0.59
numpy == 1.26.4
scikit-learn == 1.6.1
pandas == 2.0.3
mlflow == 2.22.0
SQLAlchemy == 2.0.40
psycopg2-binary == 2.9.10
boto3
```

Estas dependencias permiten trabajar con el modelo, realizar el procesamiento de datos y establecer comunicación con PostgreSQL, MLflow y MinIO.

---

## 2. Configuración de MLflow

El notebook establece como servidor de seguimiento:

```text
http://mlflow:5000
```

y utiliza el experimento:

```text
Penguins_Classification_RF
```

El autologging de Scikit-learn está deshabilitado:

```python
mlflow.sklearn.autolog(disable=True)
```

Esto permite controlar manualmente qué parámetros y métricas se registran en cada ejecución.

---

## 3. Obtención de los datos

Los datos no se cargan directamente desde el CSV durante el entrenamiento.

El notebook establece una conexión con:

```text
data_db:5432
```

y obtiene la información almacenada en:

```text
penguinsdb
```

mediante la consulta:

```sql
SELECT * FROM penguins
```

Los datos son posteriormente almacenados en un DataFrame de Pandas.

---

# Preprocesamiento

La columna:

```text
species
```

es utilizada como variable objetivo.

Antes del entrenamiento se eliminan los registros que no contienen una especie:

```python
df = df.dropna(
    subset=["species"]
)
```

Las características se dividen en variables numéricas y categóricas.

## Variables numéricas

```text
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
```

Para estas variables se aplica:

1. Imputación de valores faltantes mediante la mediana.
2. Estandarización mediante `StandardScaler`.

## Variables categóricas

```text
island
sex
```

Para estas variables se aplica:

1. Imputación mediante el valor más frecuente.
2. Codificación One-Hot mediante `OneHotEncoder`.

El procesamiento se implementa mediante `ColumnTransformer` y se integra con el clasificador Random Forest mediante un `Pipeline`.

De esta manera, el modelo conserva dentro de sí el proceso necesario para transformar los datos antes de realizar la predicción.

---

# División de los datos

Los datos se dividen en tres conjuntos:

```text
60% → Train
20% → Validation
20% → Test
```

La división utiliza:

```text
random_state = 42
```

y:

```text
stratify = y
```

El conjunto de entrenamiento se utiliza para ajustar los modelos.

El conjunto de validación se utiliza para seleccionar la mejor configuración.

El conjunto de prueba se utiliza únicamente para realizar la evaluación final del modelo seleccionado.

---

# Búsqueda de hiperparámetros

Para evaluar diferentes configuraciones de Random Forest se define el siguiente espacio de búsqueda:

```python
param_grid = {
    "n_estimators": [
        50,
        100,
        150
    ],

    "max_depth": [
        None,
        10
    ],

    "min_samples_split": [
        2,
        5,
        10,
        15,
        20
    ]
}
```

Las combinaciones son generadas mediante:

```python
from itertools import product
```

El total de configuraciones es:

```text
3 × 2 × 5 = 30
```

Por lo tanto, se entrenan:

```text
30 configuraciones diferentes
```

La búsqueda se realiza manualmente utilizando `itertools.product`. No se utiliza `GridSearchCV`.

Esto permite controlar individualmente cada configuración y registrar cada una como un Run independiente en MLflow.

---

# Entrenamiento de los modelos

Para cada combinación se construye un Pipeline compuesto por:

```text
Preprocesamiento
       ↓
Random Forest
```

El clasificador utiliza los siguientes hiperparámetros:

```text
n_estimators
max_depth
min_samples_split
random_state = 42
```

Cada configuración genera un Run independiente en MLflow.

Los Runs se identifican como:

```text
RF_Combination_1
RF_Combination_2
RF_Combination_3
...
RF_Combination_30
```

---

# Registro de experimentos en MLflow

Para cada ejecución se registran los hiperparámetros:

```text
n_estimators
max_depth
min_samples_split
```

También se registra la métrica:

```text
validation_accuracy
```

De esta manera, MLflow permite realizar el seguimiento y comparación de las 30 configuraciones entrenadas.

---

# Selección del mejor modelo

La métrica utilizada para seleccionar el modelo es:

```text
Validation Accuracy
```

Durante el entrenamiento, el notebook compara la Accuracy obtenida por cada configuración.

Cuando se encuentra un resultado superior al mejor resultado encontrado anteriormente, se actualizan:

```text
best_model
best_params
best_accuracy
best_run_id
```

De esta forma, al finalizar las 30 ejecuciones, se dispone de la configuración con mejor desempeño sobre el conjunto de validación.

En caso de que varias configuraciones presenten la misma Accuracy máxima, el código conserva como `best_model` la primera configuración que alcanzó dicho valor, mientras que la tabla de resultados permite identificar todas las configuraciones empatadas.

---

# Evaluación final

Una vez seleccionado el mejor modelo, se realiza la predicción utilizando:

```text
X_test
```

Se calcula:

```text
Test Accuracy
```

y se genera un reporte de clasificación mediante:

```python
classification_report()
```

El reporte permite analizar el desempeño del modelo para las tres especies:

```text
Adelie
Chinstrap
Gentoo
```

---

# Registro del modelo final

Después de seleccionar y evaluar el mejor modelo se crea un Run adicional denominado:

```text
Best_RandomForest
```

En este Run se registran los siguientes hiperparámetros:

```text
n_estimators
max_depth
min_samples_split
```

También se registran las métricas:

```text
validation_accuracy
test_accuracy
```

Además, se registran tags para identificar información relacionada con el proceso:

```text
best_experiment_run_id
model_selection
total_combinations
train_validation_test_split
```

Finalmente, el modelo se registra en MLflow mediante:

```python
mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    registered_model_name="Penguins_RF"
)
```

El modelo registrado recibe el nombre:

```text
Penguins_RF
```

---

# MLflow

MLflow es uno de los componentes principales de la arquitectura.

En este taller se utiliza para:

- Gestionar experimentos.
- Registrar parámetros.
- Registrar métricas.
- Identificar Runs.
- Almacenar artifacts.
- Registrar modelos.
- Administrar versiones mediante Model Registry.

El servidor MLflow se encuentra definido en:

```text
mlflow/Dockerfile
```

Este Dockerfile utiliza:

```text
python:3.9-slim
```

e instala:

```text
mlflow
boto3
psycopg2-binary
```

El servidor MLflow utiliza PostgreSQL como backend de metadatos:

```text
postgresql://admin:supersecret@db:5432/mlflowdb
```

Los artifacts se almacenan mediante MinIO utilizando:

```text
s3://mlflows3/artifacts
```

La interfaz web de MLflow se encuentra disponible en:

```text
http://localhost:5001
```

---

# MinIO

MinIO funciona como **Artifact Store de MLflow**.

El servicio utiliza la imagen:

```text
ghcr.io/coollabsio/minio:RELEASE.2025-04-22T22-12-26Z
```

Las credenciales configuradas son:

```text
Usuario: admin
Contraseña: supersecret
```

El almacenamiento persistente utiliza el volumen:

```text
minio_data
```

El proyecto utiliza el bucket:

```text
mlflows3
```

El bucket es creado automáticamente mediante el servicio:

```text
minio-init
```

Este servicio espera a que MinIO esté disponible y crea el bucket si todavía no existe.

La consola web de MinIO está disponible en:

```text
http://localhost:9001
```

La relación entre MLflow y MinIO es:

```text
MLflow
   │
   ├── Metadatos
   │       ↓
   │   PostgreSQL
   │
   └── Artifacts
           ↓
         MinIO
           ↓
       mlflows3
```

MLflow mantiene la información y referencias del modelo, mientras que MinIO almacena físicamente los artifacts asociados.

---

# Docker Compose

El archivo:

```text
docker-compose.yaml
```

es el encargado de definir y coordinar todos los servicios del proyecto.

Los servicios implementados son:

```text
db
data_db
data_loader
minio
minio-init
mlflow
jupyter
inference_api
```

La arquitectura puede representarse de la siguiente manera:

```text
                         Docker Compose
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
   PostgreSQL             PostgreSQL              MinIO
    mlflowdb              penguinsdb                │
       │                      │                 mlflows3
       │                      │                     │
       ▼                      ▼                     ▼
    MLflow ◄──────────── JupyterLab ─────────► Artifacts
       │
       ▼
MLflow Model Registry
       │
       ▼
    FastAPI
       │
       ▼
   /predict
```

---

# Data Loader

El servicio:

```text
data_loader
```

utiliza la imagen:

```text
python:3.11-slim
```

y monta las siguientes carpetas:

```text
./Data:/Data
./scripts:/scripts
```

Al iniciar instala:

```text
pandas
sqlalchemy
psycopg2-binary
```

y ejecuta:

```text
python /scripts/load_data.py
```

El servicio depende de:

```text
data_db
```

y espera a que PostgreSQL se encuentre saludable antes de ejecutar el script.

Por lo tanto, la carga del dataset en PostgreSQL es automática al iniciar el entorno.

---

# API REST

La API se encuentra en:

```text
api/main.py
```

y utiliza:

```text
FastAPI
MLflow
Pandas
```

La imagen de la API se construye mediante:

```text
api/Dockerfile
```

utilizando:

```text
python:3.11-slim
```

Las principales dependencias de la API son:

```text
fastapi
uvicorn
mlflow==2.22.0
scikit-learn==1.6.1
numpy==1.26.4
pandas==2.0.3
boto3
psycopg2-binary
```

La API se ejecuta mediante Uvicorn en:

```text
0.0.0.0:8010
```

y se publica mediante:

```text
8010:8010
```

---

# Obtención del modelo mediante MLflow

La API no contiene una copia local independiente del modelo.

En lugar de almacenar el modelo dentro de la imagen de Docker de la API, este se obtiene directamente desde **MLflow Model Registry**.

La configuración utilizada es:

```python
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "Penguins_RF"
MODEL_VERSION = "1"
```

La API construye la URI:

```text
models:/Penguins_RF/1
```

y utiliza:

```python
mlflow.pyfunc.load_model()
```

para cargar el modelo.

El flujo de obtención es:

```text
FastAPI
   │
   │ models:/Penguins_RF/1
   ▼
MLflow Model Registry
   │
   ▼
Artifact Store
   │
   ▼
MinIO
```

El modelo se carga durante el inicio de la API.

---

# Endpoints de la API

## GET `/`

Devuelve información básica de la API, del modelo y de MLflow.

Ejemplo:

```json
{
    "message": "Penguins Classification API",
    "model": "Penguins_RF",
    "version": "1",
    "mlflow": "http://mlflow:5000"
}
```

---

## GET `/health`

Permite comprobar si la API está activa y si el modelo fue cargado correctamente.

Ejemplo:

```json
{
    "status": "ok",
    "model_loaded": true,
    "model": "Penguins_RF",
    "version": "1"
}
```

---

## POST `/predict`

Permite realizar una inferencia utilizando el modelo registrado en MLflow.

Los datos esperados son:

```text
island
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
sex
```

### Valores esperados

- `island`: `Biscoe`, `Dream` o `Torgersen`
- `bill_length_mm`: longitud del pico en milímetros
- `bill_depth_mm`: profundidad del pico en milímetros
- `flipper_length_mm`: longitud de la aleta en milímetros
- `body_mass_g`: masa corporal en gramos
- `sex`: `male` o `female`

Ejemplo de entrada:

```json
{
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "male"
}
```

Ejemplo de respuesta:

```json
{
    "prediction": "Adelie",
    "model": "Penguins_RF",
    "version": "1"
}
```

La documentación interactiva de la API está disponible mediante Swagger:

```text
http://localhost:8010/docs
```

Swagger incluye una descripción de los valores que pueden utilizarse para realizar una inferencia.

---

# Preprocesamiento durante la inferencia

La API no realiza manualmente el escalamiento ni la codificación de las variables.

Esto es posible porque el modelo registrado corresponde a un **Pipeline completo de Scikit-learn**, compuesto por:

```text
Datos de entrada
      ↓
ColumnTransformer
      ↓
Imputación
      ↓
Escalamiento / One-Hot Encoding
      ↓
Random Forest
      ↓
Predicción
```

Por esta razón, la API recibe directamente las características originales del pingüino y el Pipeline se encarga de aplicar las transformaciones necesarias antes de generar la predicción.

---

# Lista de puertos

| Servicio | Puerto del contenedor | Puerto del host | Acceso |
|---|---:|---:|---|
| PostgreSQL MLflow | 5432 | 8030 | Base de datos de MLflow |
| PostgreSQL Penguins | 5432 | 8031 | Base de datos del dataset |
| MLflow | 5000 | 5001 | `http://localhost:5001` |
| JupyterLab | 8888 | 8888 | `http://localhost:8888` |
| MinIO API | 9000 | 9000 | API S3 |
| MinIO Console | 9001 | 9001 | `http://localhost:9001` |
| FastAPI | 8010 | 8010 | `http://localhost:8010` |
| Swagger | 8010 | 8010 | `http://localhost:8010/docs` |


## Nota: 
Durante la implementación se presentaron inconvenientes para acceder a los servicios utilizando directamente la dirección IP de la máquina virtual desde el navegador. Debido a esta restricción, las pruebas y validaciones realizadas para este entregable se llevaron a cabo mediante localhost, utilizando los puertos configurados para cada servicio.

---

# Ejecución del proyecto

## 1. Levantar los servicios

Desde la carpeta raíz del proyecto:

```bash
docker compose up -d
```

Para construir las imágenes antes de iniciar:

```bash
docker compose up -d --build
```

---

## 2. Verificar los servicios

```bash
docker compose ps
```

---

## 3. Consultar los logs

Para todos los servicios:

```bash
docker compose logs -f
```

Para MLflow:

```bash
docker compose logs -f mlflow
```

Para la API:

```bash
docker compose logs -f inference_api
```

Para el cargador de datos:

```bash
docker compose logs data_loader
```

---

# Ejecución del entrenamiento

Una vez que los servicios estén funcionando:

1. Acceder a JupyterLab:

```text
http://localhost:8888
```

2. Abrir el notebook:

```text
modelo.ipynb
```

3. Ejecutar las celdas del notebook.

4. El notebook obtiene los datos desde PostgreSQL.

5. Se realiza el preprocesamiento.

6. Se dividen los datos en Train, Validation y Test.

7. Se generan las 30 combinaciones de hiperparámetros.

8. Se entrena cada configuración.

9. Cada entrenamiento genera un Run en MLflow.

10. Se selecciona el modelo con mayor `Validation Accuracy`.

11. El modelo seleccionado se evalúa sobre el conjunto de prueba.

12. Se registra el modelo como:

```text
Penguins_RF
```

13. La API obtiene posteriormente la versión registrada desde MLflow.

---

# Visualización de los experimentos

Después de ejecutar el notebook, los experimentos pueden consultarse en:

```text
http://localhost:5001
```

El experimento utilizado es:

```text
Penguins_Classification_RF
```

Dentro de MLflow se pueden consultar los Runs:

```text
RF_Combination_1
RF_Combination_2
RF_Combination_3
...
RF_Combination_30
```

También se puede consultar el Run:

```text
Best_RandomForest
```

que contiene los parámetros y métricas del modelo seleccionado.

---

# Flujo completo del proyecto

```text
                    penguins.csv
                         │
                         ▼
                  ┌─────────────┐
                  │ DataLoader  │
                  └──────┬──────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ PostgreSQL    │
                 │ penguinsdb    │
                 └───────┬───────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  Jupyter    │
                  │   modelo    │
                  │   .ipynb    │
                  └──────┬──────┘
                         │
                  Preprocesamiento
                         │
                         ▼
                30 configuraciones
                   Random Forest
                         │
                         ▼
                    ┌─────────┐
                    │ MLflow  │
                    └────┬────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        PostgreSQL                 MinIO
        mlflowdb                  mlflows3
        metadatos                 artifacts
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                 Model Registry
                  Penguins_RF
                         │
                         ▼
                    FastAPI
                         │
                         ▼
                    /predict
                         │
                         ▼
                    Predicción
```

---

# Comandos útiles

Ver los contenedores:

```bash
docker compose ps
```

Detener los servicios:

```bash
docker compose down
```

Reiniciar los servicios:

```bash
docker compose restart
```

Reconstruir todas las imágenes:

```bash
docker compose up -d --build
```

Reconstruir únicamente la API:

```bash
docker compose up -d --build inference_api
```

Ver los logs de MLflow:

```bash
docker compose logs -f mlflow
```

Ver los logs de la API:

```bash
docker compose logs -f inference_api
```

Ver los logs del cargador de datos:

```bash
docker compose logs data_loader
```

---

# Volúmenes

Docker Compose utiliza tres volúmenes principales:

```text
minio_data
postgres_data
penguins_data
```

Su función es:

| Volumen | Uso |
|---|---|
| `minio_data` | Almacenamiento persistente de MinIO y artifacts |
| `postgres_data` | Persistencia de la base de datos de MLflow |
| `penguins_data` | Persistencia de la base de datos de Penguins |

Los volúmenes permiten conservar la información aunque los contenedores sean detenidos.

Para detener el proyecto sin eliminar los volúmenes:

```bash
docker compose down
```

Para eliminar también los volúmenes:

```bash
docker compose down -v
```

> **Advertencia:** `docker compose down -v` elimina los volúmenes administrados por Docker Compose. Esto puede eliminar los metadatos de MLflow, los artifacts almacenados en MinIO y los datos almacenados en PostgreSQL.

---

# Arquitectura de almacenamiento

Una de las principales características del proyecto es la separación de los diferentes tipos de información.

## Datos del problema

```text
penguins.csv
      ↓
DataLoader
      ↓
PostgreSQL
penguinsdb
```

## Metadatos de experimentación

```text
MLflow
   ↓
PostgreSQL
mlflowdb
```

## Artifacts de los modelos

```text
MLflow
   ↓
MinIO
mlflows3
```

## Inferencia

```text
FastAPI
   ↓
MLflow Model Registry
   ↓
Penguins_RF v1
   ↓
MinIO
```

Esta separación permite diferenciar claramente los datos del problema, los metadatos de experimentación y los artifacts asociados a los modelos.

---

# Consideraciones de seguridad

Las credenciales utilizadas en este proyecto están definidas directamente en `docker-compose.yaml` con fines académicos y de demostración.

Ejemplo:

```text
Usuario: admin
Contraseña: supersecret
```

Para un entorno de producción no se recomienda almacenar credenciales directamente en el archivo `docker-compose.yaml`.

Se recomienda utilizar variables de entorno mediante un archivo:

```text
.env
```

y agregarlo al archivo `.gitignore`:

```text
.env
```

De esta manera, las credenciales pueden mantenerse fuera del repositorio.

---
