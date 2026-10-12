from locust import HttpUser, task, between

class UsuarioDeCarga(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def hacer_inferencia(self):
        # 1. Ajuste del payload a las características de entrada del modelo de pingüinos
        payload = {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE"
        }
        
        # catch_response permite decidir explícitamente si Locust debe registrar
        # la petición como éxito o como fallo.
        with self.client.post(
            "/predict", json=payload, name="/predict", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return

            try:
                body = response.json()
            except ValueError:
                response.failure("La respuesta no contiene JSON válido")
                return

            # 2. Ajuste de la clave de validación. 
            # Reemplaza "prediction" por la clave exacta que retorna tu API de FastAPI 
            # (ej. "resultado", "prediccion", "species").
            if "prediction" not in body:
                response.failure("La respuesta no contiene el campo esperado de predicción")
                return

            response.success()