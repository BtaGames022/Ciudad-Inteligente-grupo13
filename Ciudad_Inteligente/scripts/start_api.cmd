@echo off
setlocal
cd /d %~dp0..

if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: No se encontro el entorno virtual. Ejecuta scripts\setup.cmd primero.
  exit /b 1
)
call .venv\Scripts\activate.bat

set "API_URL=http://127.0.0.1:8000"
set "HOST=0.0.0.0"
set "PORT=8000"

if not exist "logs" mkdir "logs"
set "LOG=logs\uvicorn_run.log"

echo Iniciando API en %HOST%:%PORT% ...
python -m uvicorn api.main:app --host %HOST% --port %PORT% --reload > "%LOG%" 2>&1

echo Log guardado en %LOG%
endlocal
