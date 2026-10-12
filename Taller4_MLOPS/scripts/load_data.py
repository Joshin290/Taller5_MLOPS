import time
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://admin:supersecret@data_db:5432/penguinsdb"
CSV_PATH = "/Data/penguins.csv"

print("======================================")
print("CARGADOR DE DATASET PENGUINS")
print("======================================")

print("Esperando a PostgreSQL...")

engine = None

for attempt in range(30):
    try:
        print(f"Intento {attempt + 1}/30...")

        engine = create_engine(
            DATABASE_URL,
            connect_args={"connect_timeout": 5}
        )

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("PostgreSQL está disponible.")
        break

    except Exception as e:
        print(f"ERROR REAL: {type(e).__name__}: {e}")
        time.sleep(2)

else:
    raise RuntimeError("No fue posible conectarse a PostgreSQL.")

print("Leyendo dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Registros encontrados: {len(df)}")
print(f"Columnas: {list(df.columns)}")

print("Cargando dataset en PostgreSQL...")

df.to_sql(
    "penguins",
    engine,
    if_exists="replace",
    index=False
)

print("======================================")
print("Dataset cargado correctamente")
print("======================================")
print("Tabla: penguins")
print(f"Registros: {len(df)}")