@echo off
setlocal
cd /d %~dp0..

if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: No se encontro el entorno virtual. Ejecuta scripts\setup.cmd primero.
  exit /b 1
)
call .venv\Scripts\activate.bat

set "API_URL=http://127.0.0.1:8000"
set "APP_PORT=8501"
if not exist "logs" mkdir "logs"
set "LOG=logs\streamlit_run.log"

echo Iniciando Streamlit en puerto %APP_PORT% apuntando a %API_URL% ...
streamlit run app/streamlit_app.py --server.port %APP_PORT% > "%LOG%" 2>&1

echo Log guardado en %LOG%
endlocal
