🚗 Motor de Predicción de Riesgo de Siniestralidad VialProyecto: Desafío Smart Cities Duoc UC 2025Informe Técnico Asociado: Smart City.docx1. Resumen del ProyectoEste repositorio contiene la solución técnica para el Desafío Smart Cities Duoc UC 2025. El núcleo de este proyecto es un "Motor de Riesgo" (IA Tabular), un modelo de Machine Learning diseñado para predecir la severidad de los siniestros de tránsito en la Región Metropolitana.Utilizando un conjunto de datos consolidado de 63,689 puntos de siniestros únicos (2020-2024), se entrenó un clasificador XGBoost. El modelo final, que incorpora ingeniería de características como clustering geoespacial (DBSCAN), es capaz de discriminar entre puntos de "Bajo Riesgo" y "Alto Riesgo" con un AUROC de 0.7191. Este motor sirve como el backend para la API de predicción (/predict).2. 📂 Estructura del RepositorioEl proyecto está organizado para cumplir con los requisitos de la hackathon (/src, /api, /app) y asegurar la reproducibilidad./
├── /api/             # Cód. de la API (FastAPI) y artefactos del modelo
│   ├── main.py
│   ├── modelo_riesgo_vX.joblib
│   └── preprocesador_vX.joblib
│
├── /app/             # Cód. de la App Demo (Streamlit)
│   └── app.py
│
├── /src/             # Scripts de entrenamiento y limpieza
│   ├── Limpieza.py
│   ├── entrenar_modelo.py
│   └── analizar_fairness.py
│
├── /data_raw/        # (Input) CSVs brutos de siniestralidad
│
├── /data_processed/  # (Generado) Dataset maestro limpio
│   └── Siniestros_Maestro_Consolidado_HACKATHON_FINAL.csv
│
├── /kb/              # (Input) Base de conocimiento para el Coach RAG
│
├── requirements.txt  # Dependencias del proyecto
└── README.md         # Este archivo
3. 🛠️ Instalación y DependenciasPara levantar el proyecto localmente, sigue estos pasos:Clonar el repositorio:Bashgit clone [URL_DEL_REPO]
cd [NOMBRE_DEL_REPO]
Crear un entorno virtual (recomendado):Bashpython -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
Instalar dependencias:El archivo requirements.txt contiene todas las librerías necesarias.Bashpip install -r requirements.txt
Librerías clave: pandas, xgboost, scikit-learn, fastapi, uvicorn, joblib, numpy, streamlit.4. 🚀 Flujo de Ejecución (Paso a Paso)Paso 1: Procesamiento de Datos (ETL)Coloca todos los archivos .csv de siniestros brutos dentro de la carpeta /data_raw/. El script Limpieza.py unifica, limpia y agrupa los datos por coordenadas.Bash# Navega a la carpeta de scripts
cd src

# Ejecuta la limpieza
# Input: /data_raw/*.csv
# Output: /data_processed/Siniestros_Maestro_Consolidado_HACKATHON_FINAL.csv
python Limpieza.py
Paso 2: Entrenamiento del ModeloEjecuta el script de entrenamiento principal. Este script realiza la ingeniería de características (DBSCAN), el afinamiento de hiperparámetros (RandomizedSearchCV) y la validación.Bash# Desde la carpeta /src/
python entrenar_modelo.py
Input: /data_processed/Siniestros_Maestro_Consolidado_HACKATHON_FINAL.csvOutput (en /src/): modelo_riesgo_vX.joblib y preprocesador_vX.joblibPaso 3: Mover ArtefactosMueve los dos archivos .joblib generados desde la carpeta /src/ a la carpeta /api/ para que el servidor de FastAPI pueda cargarlos.Paso 4: Ejecutar la API (FastAPI)Esta API expone los endpoints /predict y /coach.Bash# Navega a la carpeta de la API
cd ../api

# Inicia el servidor
uvicorn main:app --reload
La API estará disponible en http://127.0.0.1:8000.Puedes ver la documentación interactiva en http://127.0.0.1:8000/docs.Paso 5: Ejecutar la App Demo (Streamlit)En una terminal separada, lanza la aplicación web interactiva.Bash# Navega a la carpeta de la app
cd ../app

# Inicia Streamlit
streamlit run app.py
La aplicación estará disponible en http://127.0.0.1:8501.5. 📊 Detalles del Modelo y MétricasEl desarrollo del modelo se centró en un clasificador XGBoost.Ingeniería de Características: La variable clave fue Hotspot_Cluster, generada por un algoritmo DBSCAN que identificó 8 clusters de alta densidad de siniestros.Manejo de Desbalance: Se utilizó el hiperparámetro scale_pos_weight (valor: 3.38) para forzar al modelo a priorizar la detección de la clase minoritaria ("Alto Riesgo").Validación: El modelo se evaluó contra un set de prueba (20% de los datos) usando una división estratificada. El análisis de la validación temporal (V5) fue crucial y demostró un colapso del modelo (AUROC 0.437) debido al concept drift post-pandemia. El modelo V4 (split aleatorio) se reporta aquí como el baseline de rendimiento, reconociendo esta limitación.Métricas de Desempeño (Modelo V4)Los resultados del modelo afinado en el set de prueba son:MétricaPuntajeRúbricaAUROC (Principal)0.71917 pts (0.70–0.74)Brier Score0.21031 pt (> 0.18)AUPRC0.4352N/ARecall (Alto Riesgo)0.68N/ANota: El modelo fue optimizado para Recall (sensibilidad), priorizando la minimización de Falsos Negativos (peligros no detectados). El Brier Score bajo (1 pt) fue un trade-off aceptado para maximizar la detección (Recall) mediante scale_pos_weight.
