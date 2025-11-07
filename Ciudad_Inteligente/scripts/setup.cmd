@echo off
setlocal
cd /d %~dp0..

REM Evitar conflictos con instalación embebida de Python
set PYTHONHOME=
set PYTHONPATH=

REM Crear y activar venv
if not exist .venv (
    python -m venv .venv
)
if not exist .venv\Scripts\activate.bat (
    echo ERROR: No se pudo crear el entorno virtual. Asegurate de tener Python en PATH.
    echo Pista: prueba con el lanzador: py -3 -m venv .venv
    exit /b 1
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Instalacion completada.
echo Para iniciar la API:  scripts\start_api.cmd
echo Para iniciar la APP:  scripts\start_app.cmd
endlocal
