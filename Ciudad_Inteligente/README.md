# Smart Route Optimizer (FastAPI + Streamlit)

Proyecto base para el desafío Ciudades Inteligentes.

Estructura:
- api/main.py: API FastAPI con /predict, /coach, /comunas, /geo/points, /stats, /db/health, /admin/ingest
- api/db.py: Conector opcional a Oracle (python-oracledb)
- app/streamlit_app.py: Frontend en Streamlit que consume la API
- kb/fichas.md: Mini base de conocimiento local (RAG)
- requirements.txt: dependencias

## Requisitos
- Python 3.10+
- (Opcional) Oracle DB accesible; la app usa `oracledb` en modo thin por defecto (no requiere Instant Client). Si tu conexión usa wallet/TLS, configura `TNS_ADMIN` hacia `./wallet`.

## Instalación rápida (Windows)
Usa los scripts incluidos:

```bat
scripts\setup.cmd
```
Esto creará `.venv`, activará el entorno, actualizará pip e instalará dependencias de `requirements.txt`.

## Iniciar servicios
- API (puerto 8000):
```bat
scripts\start_api.cmd
```
- App Streamlit (puerto 8501):
```bat
scripts\start_app.cmd
```
La app usa la variable `API_URL` (por defecto `http://127.0.0.1:8000`).

## Variables de Entorno para Oracle (opcional)
Configura si vas a usar BD Oracle:
- ORACLE_USER=usuario
- ORACLE_PASSWORD=clave
- ORACLE_DSN=host:1521/servicio
- (Opcional) ORACLE_WALLET_DIR=ruta\al\wallet
- (Opcional) ORACLE_CLIENT_LIB=ruta\al\instantclient

Ejemplo (cmd):
```bat
set ORACLE_USER=hr
set ORACLE_PASSWORD=hrpwd
set ORACLE_DSN=localhost:1521/XEPDB1
```

Comprueba salud de BD:
- GET http://127.0.0.1:8000/db/health

## Uso manual (sin scripts)
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py api
:: nueva consola
.venv\Scripts\activate
set API_URL=http://127.0.0.1:8000
python main.py app
```

## Datos y Modelo
Coloca en `datos/` si quieres usar datos reales:
- `motor_riesgo.joblib`
- `Siniestros_Maestro_Consolidado_Hackathon.csv`
Si no existen, la API usa un dataset sintético de ejemplo.

Modelos actualizados ya incluidos en `modelo pre entrenado actualizado/` (`preprocesador_v1.joblib` y `modelo_riesgo_v1.joblib`).

## Endpoints clave
- GET `/` → estado API
- POST `/predict` → `{lat, lon}`
- POST `/coach` → `{comuna, ubicacion_desc, causa_comun, indice_severidad}`
- GET `/comunas`
- GET `/geo/points`
- GET `/stats`
- GET `/db/health`

## Troubleshooting
- oracledb: funciona en modo thin por defecto (no requiere Instant Client). Si tu DSN/TNS requiere wallet, usa `ORACLE_WALLET_DIR` o define `TNS_ADMIN` a `wallet/`.
- Si no usarás Oracle, ignora endpoints de BD (responderán 503) y usa `/predict`, `/geo/points`, `/stats` con el CSV o datos sintéticos.
- Si puertos ocupados: cambia `--port` en el comando uvicorn o el puerto de Streamlit.
