# Taller 5 MLOPS

## Grupo 3

### Integrantes
* **Juan David Clavijo Ortiz**
* **Laura Sofía Rodríguez Pérez**
* **Joshua Alexander Valero Lozano**

## Descripción:

Este repositorio contiene el desarrollo del Taller 5 de MLOps. El objetivo principal de esta práctica es relacionar las restricciones de infraestructura (CPU y memoria) con la capacidad observada mediante pruebas de carga, utilizando una API de inferencia basada en un modelo predictivo.

## Estructura de Carpetas del Proyecto

El proyecto está organizado de la siguiente forma:


Taller5_MLOPS/
├── Dockerhub/
│   ├── Dockerhub.png                     # Evidencia de la imagen alojada en la nube
│   └── link (Dockerhub).txt              # Enlace directo al repositorio público
├── Evidencias/
│   ├── Video 1.mp4                       # Grabación 1
│   └── Video 2.mp4                       # Grabación 2
├── locust (maq personal)/
│   └── locustfile.py                     # Script de Python con los escenarios de prueba (implementado en la máquina personal)
├── Resultados/                             
│   ├── reporte_1replica_stats (...).csv  # Reportes de Locust (1 réplica)
│   ├── reporte_3replicas_stats (...).csv # Reportes de Locust (3 réplicas)
│   └── Terminal.txt                      # Historialde resultados en la terminal
├── Taller4_MLOPS/                        # Archivos del taller pasado, necesarios para la creación de la imagen
├── docker-compose.yaml                   # Orquestación de contenedores y límites
├── nginx.conf                            # Configuración del balanceador de carga
└── README.md                             # Documentación principal

## Definiciones clave

Para comprender la arquitectura y las herramientas utilizadas en esta práctica, a continuación se definen los conceptos principales:

**Locust:** Es una herramienta de código abierto basada en Python diseñada para realizar pruebas de carga y rendimiento. Permite simular el comportamiento de miles de usuarios concurrentes atacando un sistema para medir su tiempo de respuesta (latencia), tasa de fallos y capacidad máxima de peticiones por segundo (RPS) antes de saturarse.

**Docker Hub:** Es un servicio de registro en la nube donde los desarrolladores pueden almacenar y distribuir imágenes de contenedores. En este proyecto, funciona como el repositorio público desde el cual la máquina virtual descarga la imagen empaquetada de la API (penguins-api) para ejecutarla.

**Nginx:** Es un servidor web de alto rendimiento que en esta arquitectura actúa como proxy inverso y balanceador de carga. Su función principal es recibir todas las peticiones externas provenientes de Locust y distribuirlas de manera equitativa entre los contenedores disponibles, evitando que uno solo colapse por el tráfico.

**Réplicas:** Son copias idénticas y simultáneas de un mismo contenedor (en este caso, de la API de inferencia) que se ejecutan en paralelo. Aumentar el número de réplicas se conoce como escalabilidad horizontal, lo que permite al sistema procesar una mayor cantidad de tráfico distribuyendo la carga de trabajo entre múltiples recursos (CPUs/RAM).

## Desarrollo del Taller

A continuación, se detalla cómo se abordó cada uno de los puntos solicitados en el taller:

### 1. Cree una imagen Docker que contenga una API FastAPI capaz de realizar inferencia con un modelo previamente entrenado y cargado desde MLflow.

Para el desarrollo de este taller, se reutilizó el flujo de experimentación del Taller 4, seleccionando el mejor modelo de Machine Learning obtenido para la clasificación de la especie de pingüinos: un clasificador Random Forest (Penguins_RF, versión 7).

En el entorno inicial, la API consultaba directamente el servidor de MLflow para realizar cada inferencia. Sin embargo, para garantizar la autonomía, reducir la latencia (evitando tiempos de descarga por red) y permitir el empaquetado del servicio en un contenedor aislado, se implementó una estrategia de extracción e integración local del artefacto.

### Paso 1.1: Descarga del Modelo desde el Registro de MLflow
Se añadió una nueva celda de código en el Jupyter Notebook de entrenamiento (modelo.ipynb) para conectarse al servidor de tracking (http://mlflow:5000), localizar el modelo registrado y descargar sus artefactos directamente al almacenamiento temporal de la máquina virtual.

Fragmento utilizado en Jupyter para extraer el artefacto:

```text
import os
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
destino = "/tmp/modelo_penguins"
os.makedirs(destino, exist_ok=True)

ruta = mlflow.artifacts.download_artifacts(
    artifact_uri="models:/Penguins_RF/7",
    dst_path=destino
)
```

#### Paso 1.2: Integración de los artefactos en el entorno de desarrollo
Como se evidencia en la estructura del proyecto, los archivos descargados desde MLflow fueron trasladados al directorio de código fuente de la API, específicamente en la ruta api/model/. Esta carpeta contiene todos los metadatos y el modelo serializado necesarios para la predicción, alojados junto al script principal (main.py):   

- **model.pkl:** El modelo Random Forest entrenado y serializado.   
- **MLmodel, conda.yaml, python_env.yaml, requirements.txt:** Archivos de metadatos y dependencias generados por MLflow que garantizan la reproducibilidad del entorno predictivo.   

#### Paso 1.3: Refactorización de la API (FastAPI) para Lectura Local
Con los archivos integrados en la carpeta model/, se modificó el código de main.py para cargar el modelo en la memoria RAM del contenedor una única vez durante el arranque (@app.on_event("startup")), utilizando mlflow.sklearn.load_model().

```text
MODEL_NAME = "Penguins_RF"
MODEL_VERSION = "7"
# Apunta a la carpeta local integrada en la imagen Docker
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"CARGANDO MODELO DESDE ARCHIVO LOCAL: {MODEL_PATH}")
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")

```
Gracias a esta estructura, al momento de ejecutar el comando docker build, el Dockerfile copia íntegramente la carpeta api/ al interior de la imagen. Esto da como resultado un contenedor ligero y completamente independiente del servidor de MLflow.

### 2. Publique la imagen de inferencia en Docker Hub.

A continuación, se detalla el flujo de trabajo utilizado en la consola para realizar esta publicación:

#### Paso 2.1: Construcción (Build) de la imagen local
Con los archivos integrados en el paso anterior (modelo, main.py, requirements.txt y Dockerfile), se compiló la imagen de Docker en la máquina virtual. ejecutando el siguiente comando en la raíz del proyecto api/:

```text
docker build -t penguins-api .
```

Este comando instruye a Docker para leer el Dockerfile, instalar las dependencias de Python y empaquetar el código, creando una imagen llamada localmente penguins-api.

#### Paso 2.2: Autenticación en Docker Hub
Antes de poder enviar cualquier imagen, es necesario autenticar la sesión local con los servidores de Docker Hub. Se ejecutó el siguiente comando e ingresaron las credenciales correspondientes a la cuenta:

```text
docker login
```
#### Paso 2.3: Etiquetado (Tag) de la imagen
Por convención, Docker Hub exige que el nombre de la imagen incluya el nombre de usuario del repositorio de destino. Por lo tanto, se procedió a crear una nueva etiqueta (tag) a partir de la imagen local construida en el Paso 2.1, asociándola al respectivo usuario:

```text
docker tag penguins-api:latest joshin290/penguins-api:latest
```

#### Paso 2.4: Subida (Push) al repositorio remoto
Finalmente, una vez etiquetada correctamente, se envió la imagen a los servidores públicos de Docker Hub mediante el comando push:

```text
docker push joshin290/penguins-api:latest
```

### 3. Cree un docker-compose.yaml que permita ejecutar la imagen publicada.

Para desplegar la solución en la máquina virtual, el primer paso fue descargar (hacer pull) la imagen previamente publicada en el registro de Docker Hub. Esto garantiza que el entorno de despliegue cuente con el ejecutable exacto antes de levantar los servicios. Se ejecutó el siguiente comando en la terminal de la máquina virtual:

```text
docker pull joshin290/penguins-api:latest
```

Una vez descargada la imagen localmente en la VM, se estructuró un archivo "docker-compose.yaml" para orquestar su ejecución. Este archivo define la topología de red, los límites de recursos de hardware y la integración con el balanceador de carga.

### 4. Despliegue en una VM la API de inferencia, Nginx como balanceador y los demás servicios que la aplicación necesite, pero no ejecute Locust en esa misma VM. Inicie una sola réplica de la API y limítela a 1 CPU y 1 GB de memoria.

Para ejecutar este despliegue, nos conectamos a la máquina virtual y, utilizando el archivo "docker-compose.yaml" construido en el paso anterior, levantamos la infraestructura en segundo plano con el siguiente comando:

```text
docker compose up -d
```

Es necesario destacar que Locust no se ejecutó en esta máquina virtual. Las pruebas de carga se lanzaron desde un entorno externo (la máquina personal), asegurando que los recursos consumidos al simular a los usuarios concurrentes no compitan con la CPU y la memoria asignadas a la API.

El despliegue en la VM corresponde a los siguientes servicios: 

- **Uso de la imagen remota (image: joshin290/penguins-api:latest):** El servicio hace un llamado explícito a la imagen que se acaba de descargar mediante el comando pull, asegurando que se utilice la versión empaquetada correcta.

- **Escalabilidad e inicio de una sola réplica (replicas: 1):** Se desplegó inicialmente una única instancia de la API. El uso del atributo replicas de Docker Swarm/Compose facilita el cambio rápido a la arquitectura de 3 réplicas solicitada en los puntos posteriores del taller, sin necesidad de reescribir la configuración general. 

- **Límites estrictos de hardware (limits: cpus: '1.0', memory: '1G'):** Para poder medir adecuadamente el punto de quiebre y el rendimiento de la aplicación bajo pruebas de carga, se configuró el bloque deploy.resources.limits. Esto asegura que el contenedor nunca consuma más de 1 CPU y 1 GB de RAM, independientemente de la capacidad total que posea la máquina virtual anfitriona.

- **Servicio Proxy/Balanceador (nginx):**  Se utilizó una imagen ligera de Nginx (alpine) que escucha las peticiones externas en el puerto 8000. Mediante el uso de volumes, se inyectó un archivo local nginx.conf como solo lectura (:ro), el cual contiene las directivas para gestionar y balancear la carga hacia la réplica de inference_api.

- **Restricción de exposición de red (expose: "8010"):** Por seguridad y control de flujo, el puerto de la API (8010) no se publicó hacia el exterior (se omitió la directiva ports). Este puerto solo quedó expuesto dentro de la red virtual interna de Docker, obligando así a que todo el tráfico de la prueba de Locust pase obligatoriamente a través del balanceador Nginx.

### 5. Ejecute Locust desde una segunda VM o desde su máquina personal. Ambas máquinas deben poder comunicarse por red. De esta forma, la CPU y la memoria usadas para simular usuarios no compiten con los recursos de la API.

la generación de carga se realizó desde un entorno externo (la máquina personal local con Windows), enviando las peticiones a través de la red hacia la dirección IP de la máquina virtual ([http://10.43.97.109:8000](http://10.43.97.109:8000)) donde se encontraba desplegada la API detrás de Nginx.

El proceso de ejecución de Locust se dividió en dos fases:

#### Fase 1: Exploración y validación visual (Python local)
Inicialmente, se ejecutó Locust de forma directa utilizando el entorno de Python local de la máquina personal. El objetivo de este paso fue levantar el servidor local de Locust y acceder a su interfaz gráfica a través del explorador web. Esto permitió validar rápidamente la comunicación de red con el entorno de Docker en la máquina virtual y configurar parámetros preliminares de forma interactiva.

```text
py -m locust -f locustfile.py
```
#### Fase 2: Ejecución de pruebas de carga (Docker en terminal)
Posteriormente, para el desarrollo oficial de las pruebas y la recolección estandarizada de métricas, se aplicó la ejecución mediante la imagen oficial de Locust en Docker directamente desde la terminal.

```text
docker run --rm -v "${PWD}:/mnt/locust" locustio/locust `
  -f /mnt/locust/locustfile.py `
  --headless `
  --host http://10.43.97.109:8000 `
  --users [CANTIDAD_USUARIOS] `
  --spawn-rate 10 `
  --run-time 2m `
  --csv /mnt/locust/reporte `
  --only-summary
```
**Análisis de los parámetros utilizados en Docker:** 

**-v "${PWD}:/mnt/locust":** Monta la carpeta actual para que el contenedor de Locust pueda acceder al script locustfile.py y escribir los reportes CSV directamente en la máquina personal.

**--headless:** Desactiva la interfaz web, optimizando el rendimiento de la prueba y permitiendo su ejecución por terminal.

**--host:** Define la dirección IP y el puerto del balanceador Nginx en la máquina virtual destino.

**--run-time 2m:** Establece una ventana de evaluación de 2 minutos.

**--csv:** Instruye a Locust para generar automáticamente los archivos .csv con las tablas de resultados y percentiles al finalizar.

**--only-summary:** Mantiene la terminal limpia imprimiendo únicamente el resumen estadístico final de la prueba, evitando saturar la salida estándar cada dos segundos.

### 6. Aumente gradualmente los usuarios concurrentes y determine la cantidad máxima que puede atender una réplica de 1 CPU y 1 GB sin incumplir el criterio de capacidad.

Para determinar el punto de saturación de la API, se establecióel siguiente criterio de aceptación: la configuración se considera estable si, durante una ventana continua de evaluación de dos minutos, mantiene una tasa de error inferior al 1% y un tiempo de respuesta en su percentil 95 (p95) menor a 500 milisegundos.

Bajo este escenario, y operando con una sola réplica de la API restringida a 1.0 CPU y 1 GB de memoria RAM, se ejecutó una serie de pruebas aumentando de forma escalonada la carga de trabajo. La simulación comenzó con una base de 50 usuarios concurrentes, aplicando incrementos de 50 usuarios adicionales en cada nueva iteración.

### 7. Aumente la API a tres réplicas, cada una con 1 CPU y 1 GB, y repita la misma prueba desde la otra máquina. Mantenga iguales el locustfile.py, la tasa de aparición de usuarios, la duración y el criterio de aceptación.

El aumento de capacidad se configuró directamente en el archivo de orquestación "docker-compose.yaml". Dentro de este archivo, en el servicio correspondiente a la inference_api, se modificó el valor bajo el bloque deploy, pasando el parámetro de replicas: 1 a replicas: 3. Para aplicar este cambio en caliente dentro de la máquina virtual, se ejecutó nuevamente el comando docker compose up -d, permitiendo que Nginx detectara automáticamente los nuevos contenedores para balancear la carga.

### 8. Compare la capacidad máxima, los RPS y los tiempos de respuesta obtenidos con una y con tres réplicas.

Los datos exactos de cada iteración fueron exportados automáticamente por Locust a formato .csv y guardados en la carpeta Resultados/. Por otro lado, el comportamiento del consumo de hardware (CPU/RAM) quedó documentado en el archivo Terminal.txt.   

**Tabla comparativa:** La comparación directa de la capacidad máxima, los RPS (Peticiones Por Segundo) y los tiempos de respuesta entre los escenarios de una y tres réplicas quedó consolidada en la siguiente tabla: 

| CPU/réplica | RAM/réplica | Réplicas | Usuarios | RPS | Peticiones | Fallos % | p95 (ms) | CPU observada | RAM observada | Cumple |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1.0 | 1 GB | 1 | 50 | 23.91 | 2862 | 0.00% | 330 | 74.43% | 204.3 MiB | Sí |
| 1.0 | 1 GB | 1 | 100 | 33.98 | 4071 | 43.13% | 2800 | 101.77% | 205.0 MiB | No |
| 1.0 | 1 GB | 1 | 150 | 40.49 | 4851 | 52.13% | 5200 | 102.87% | 206.3 MiB | No |
| 1.0 | 1 GB | 1 | 200 | 36.54 | 4372 | 38.52% | 7100 | 101.75% | 206.1 MiB | No |
| 1.0 | 1 GB | 3 | 50 | 27.43 | 3283 | 0.21% | 78 | 21.02% | 203.4 MiB | Sí |
| 1.0 | 1 GB | 3 | 100 | 53.62 | 6424 | 0.00% | 100 | 44.53% | 203.4 MiB | Sí |
| 1.0 | 1 GB | 3 | 150 | 77.28 | 9244 | 0.01% | 270 | 58.96% | 203.4 MiB | Sí |
| 1.0 | 1 GB | 3 | 200 | 97.85 | 11728 | 0.00% | 380 | 70.35% | 203.6 MiB | Sí |
| 1.0 | 1 GB | 3 | 220 | 104.36 | 12492 | 0.00% | 450 | 81.90% | 203.7 MiB | Sí |
| 1.0 | 1 GB | 3 | 250 | 99.50 | 11916 | 0.00% | 1200 | 86.38% | 204.6 MiB | No |

## Resolución de preguntas

#### ¿Cuál es la concurrencia máxima sostenible y cuántos RPS procesa una réplica limitada a 1 CPU y 1 GB?

- **Concurrencia máxima sostenible:** 50 usuarios concurrentes.

- **RPS procesadas:** 23.91 Peticiones Por Segundo (RPS).

Con 50 usuarios, el sistema procesa la carga exitosamente manteniendo un 0.00% de fallos y un tiempo de respuesta p95 de 330 ms (dentro del límite exigido de 500 ms).

Sin embargo, al incrementar la carga a 100 usuarios, la única réplica disponible satura completamente su límite de procesamiento (alcanzando un 101.77% de uso de CPU). Esto provoca un cuello de botella severo donde el 43.13% de las peticiones fallan (arrojando errores HTTP 502 Bad Gateway en Nginx) y el tiempo de respuesta p95 se dispara a 2800 ms, incumpliendo totalmente el criterio de capacidad.

#### ¿Cuál es la concurrencia máxima sostenible y cuántos RPS procesan tres réplicas con esos mismos límites por réplica?

- **Concurrencia máxima sostenible:** 220 usuarios concurrentes.

- **RPS procesadas:** 104.36 Peticiones Por Segundo (RPS).

Justificación técnica:
Al escalar horizontalmente a tres réplicas, Nginx logra distribuir la carga efectivamente. Con 220 usuarios concurrentes, la arquitectura procesa exitosamente 104.36 RPS manteniendo un 0.00% de fallos y un tiempo de respuesta p95 de 450 ms (el cual está al límite, pero aún por debajo del umbral de los 500 ms).

Si se incrementa la carga a 250 usuarios, aunque la tasa de fallos se mantiene en 0.00%, el tiempo de respuesta en el percentil 95 (p95) se dispara a 1200 ms, incumpliendo el criterio de capacidad establecido.

#### ¿En qué porcentaje cambió la capacidad? ¿El aumento fue cercano a tres veces o aparecieron otros cuellos de botella?

- **Cambio en la capacidad:** El incremento en usuarios concurrentes sostenibles fue del 340% (pasó de 50 a 220 usuarios). En términos de peticiones procesadas, las RPS aumentaron un 336.5% (pasando de 23.91 a 104.36 RPS).

- **Proporción del aumento:** El aumento medido fue de 4.4 veces, es decir, superó la proporción estrictamente lineal de 3x. No obstante, debido a que los saltos de prueba se hicieron de 50 en 50, la réplica individual fue registrada con un máximo de 50 usuarios porque colapsó al llegar a los 100. Sin embargo, su verdadero límite seguro probablemente se encontraba en un punto intermedio (ej. ~70-75 usuarios). Si se evalúa bajo esa óptica matemática, el escalamiento de Nginx es casi  lineal (aproximadamente 3 veces la capacidad real).

- **Aparición de nuevos cuellos de botella:** Sí, el comportamiento del hardware demuestra que el límite cambió de naturaleza. Con 1 réplica, el cuello de botella fue absoluto y directo: el contenedor llegó al límite duro de su asignación (alcanzando 101.77% de CPU), lo que causó una saturación inmediata y el colapso con errores 502.   Con 3 réplicas, cuando el sistema empieza a fallar en la latencia a los 250 usuarios (el p95 sube a 1200 ms), ninguna de las tres réplicas estaba al 100% de CPU. El consumo oscilaba entre el 82.60% y el 86.38% de CPU por contenedor. 

#### ¿Qué recurso se saturó primero y qué evidencia muestran docker stats, Locust y la máquina generadora de carga?

El recurso que se saturó primero y de forma más crítica fue la CPU asignada al contenedor de la API (en el escenario inicial de una réplica). La memoria RAM, por el contrario, tuvo un consumo marginal y nunca fue un factor limitante.Esta conclusión se fundamenta en las siguientes evidencias cruzadas obtenidas de las herramientas de monitoreo:

- **Evidencia en docker stats (Uso de hardware):**
Los registros de la máquina virtual muestran que al someter la única réplica a una carga de 100 y 150 usuarios, el contenedor alcanzó un uso de CPU de 101.77% y 102.87% respectivamente, chocando directamente contra el límite restrictivo de 1.0 núcleo definido en el archivo YAML. Mientras la CPU colapsaba, la memoria RAM se mantuvo casi inalterable alrededor de los 205 MiB (apenas un ~20% del 1 GB permitido).

- **Evidencia en las métricas de Locust (Tiempos de respuesta):**
Como la CPU estaba al tope de su capacidad, no podía procesar las peticiones a la velocidad que llegaban, generando un efecto de embudo. Locust evidenció esto mediante un disparo exponencial en los tiempos de espera: el percentil 95 (p95) saltó de 78 ms (a 50 usuarios) a unos 2800 ms (a 100 usuarios), subiendo hasta 7100 ms en las pruebas de mayor estrés.   

- **Evidencia en la máquina generadora de carga (Reporte de Errores):**
La terminal local (donde se ejecutaba Locust) comenzó a registrar caídas masivas de la API. El reporte de errores imprimió miles de respuestas con el código HTTP 502: Bad Gateway originadas por Nginx (por ejemplo, 1756 fallos con 100 usuarios, representando el 43.13% del tráfico). Esta es la evidencia definitiva de que Nginx intentaba enrutar las peticiones, pero el contenedor subyacente (FastAPI) estaba tan saturado de CPU que no podía aceptarlas, forzando al balanceador a devolver un error.   

Cuando se aplicó la arquitectura escalada a 3 réplicas, la naturaleza de la saturación cambió. A los 250 usuarios, el sistema incumplió el tiempo de respuesta (p95 de 1200 ms), pero la evidencia en docker stats mostró que el consumo de CPU en cada contenedor estaba entre el 82.60% y el 86.38% (sin llegar al 100%). Esto evidencia que el cuello de botella se desplazó: dejó de ser la CPU del contenedor individual y pasó a ser un límite de concurrencia del servidor web o el agotamiento de la CPU global de la máquina virtual host, que ya no podía suministrar ciclos de procesamiento a las réplicas, Nginx y los demás servicios simultáneamente.

#### ¿Qué servicios adicionales —por ejemplo MLflow, base de datos o balanceador— pudieron limitar el resultado?

Durante la inferencia, el modelo se carga localmente en el contenedor, por lo que la API no hace llamadas a MLflow ni a la base de datos. En docker stats, su consumo de CPU es casi nulo (0.01% - 0.05%).
