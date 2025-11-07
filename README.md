# 🚀 Optimizador de Rutas Terrestres (Smart Cities)
### [cite_start]1º Hackathon de Inteligencia Artificial Aplicada Duoc UC 2025 [cite: 46]



**Equipo:** `[Nombre de tu Equipo]`
**Miembros:**
* `[Felipe Alvarez - Rol Líder Backend/ RAG]`
* `[Claudio Gonzalez - Rol Líder IA ]`
* `[Carlos Acuña - Rol Líder Frontend]`
* `[Scarleth Quinzacara - Rol DevOps / Integración]`
* `[Sebastián Altamirano - Rol Líder de Proyecto]`

---

## 1. 🎯 Descripción del Desafío
[cite_start]El objetivo de este proyecto es desarrollar un **Optimizador de rutas terrestres** [cite: 55] [cite_start]que, basándose en datos históricos de siniestros viales, pueda identificar puntos de alto riesgo en la Región Metropolitana y generar planes de acción para reducir la probabilidad de futuros accidentes[cite: 65, 72].

## 2. 💡 Nuestra Solución (Arquitectura Híbrida)
[cite_start]Para resolver este desafío en 27 horas [cite: 56][cite_start], implementamos una arquitectura de **IA Híbrida** [cite: 87] que combina Machine Learning tradicional con LLMs:

1.  **Data Pipeline (ETL):**
    * Consolidamos **+15 archivos CSV** (2020-2024) con más de **70,000 siniestros** en un único dataset maestro (`Siniestros_Maestro_Consolidado_Hackathon.csv`).
    * Agrupamos todos los siniestros por coordenadas (`Latitude`, `Longitude`) para crear **43,679 puntos de riesgo únicos**.
    * Generamos un `Indice_Severidad` y la variable objetivo `Categoria_Ocurrencia` (`Esporadico`, `Comun`, `Muy Frecuente`).

2.  **Motor de Riesgo (ML Tabular) (`/src`):**
    * Entrenamos un modelo `XGBClassifier` (`motor_riesgo.joblib`) para predecir la `Categoria_Ocurrencia` de un punto.
    * Este modelo utiliza features geográficas (`Lat`, `Lon`), contextuales (`Comuna`, `Zona`, `Mes`) y de ubicación (`Ubicacion_Desc`).

3.  **API (FastAPI) (`/api`):**
    * Una API en Python que expone nuestro sistema al mundo.
    * [cite_start]**Endpoint `/predict`:** Recibe coordenadas, las pasa al modelo ML y devuelve el perfil de riesgo[cite: 136].
    * [cite_start]**Endpoint `/coach`:** Recibe el perfil de riesgo y genera un plan de acción[cite: 136].

4.  **Coach (LLM + RAG) (`/kb`):**
    * [cite_start]Un sistema que genera recomendaciones en lenguaje natural[cite: 65].
    * [cite_start]Utiliza un LLM que consulta una base de conocimiento local (`/kb/fichas.md`) para generar planes de acción **basados en evidencia y sin alucinaciones**, citando sus fuentes[cite: 101, 102].

5.  **App Demo (Streamlit) (`/app`):**
    * [cite_start]Una aplicación web interactiva desplegada en Hugging Face Spaces[cite: 142].
    * [cite_start]Permite al usuario seleccionar un punto y ver el análisis de riesgo y el plan de acción del "Coach" en tiempo real[cite: 140].

---

## 3. 🌐 Demo en Vivo

**Puedes probar nuestra aplicación desplegada aquí:**

### `[LINK A TU APP EN HUGGING FACE SPACES]`

---

## 4. ⚙️ Instrucciones de Instalación (Local)

1.  **Clonar el repositorio:**
    ```bash
    git clone [URL-DE-TU-REPO-GIT]
    cd [NOMBRE-DEL-REPO]
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    *Nuestro proyecto cumple con la reproducibilidad*[cite: 121].
    ```bash
    pip install -r requirements.txt
    ```

---

## 5. 🛠️ Instrucciones de Uso (Local)

Para ejecutar la solución completa, necesitas dos terminales:

1.  **Terminal 1: Iniciar la API (Backend):**
    ```bash
    cd api
    uvicorn main:app --reload
    ```
    *La API estará disponible en `http://127.0.0.1:8000`*

2.  **Terminal 2: Iniciar la App (Frontend):**
    ```bash
    cd app
    streamlit run app.py
    ```
    *La aplicación se abrirá en tu navegador en `http://127.0.0.1:8501`*

---

## 6. 📊 Métricas y Justificación (Sección D.3)

Nuestro modelo final (`motor_riesgo.joblib`) fue evaluado contra los datos de prueba, obteniendo los siguientes resultados:

| Métrica | Requisito Rúbrica | Resultado Obtenido | Veredicto |
| :--- | :--- | :--- | :--- |
| **AUROC (weighted)** | [cite_start]`> 0.75` [cite: 160] | **`0.776`** | **✅ Cumplido** |
| **Brier Score** | [cite_start]`< 0.18` [cite: 160] | `0.490` | ⚠️ No Cumplido |
| **Recall (Muy Frecuentes)**| (Métrica de equipo) | `0.44` | **Aceptable** |

### Justificación Estratégica:

Se identificó un fuerte desbalance de clases en los datos reales. Para cumplir con los requisitos, se probó un modelo con Calibración Isotónica (`CalibratedClassifierCV`).

[cite_start]Si bien el modelo final **cumple exitosamente la métrica principal de AUROC (0.776)** [cite: 160][cite_start], el Brier Score (0.49) no mejoró[cite: 160]. Esto indica que el desbalance extremo y el ruido en los datos geográficos limitan la efectividad de la calibración de probabilidades.

[cite_start]El equipo tomó la decisión estratégica de **aceptar este modelo** (con `AUROC > 0.75` y `Recall > 0.44` para clases peligrosas) para priorizar el desarrollo de los entregables críticos de API, RAG y Aplicación Demo, que suman más de 70 puntos de la evaluación[cite: 161, 167]
