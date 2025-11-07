@echo off
setlocal
cd /d %~dp0..

REM Verificar entorno virtual
if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: No se encontro el entorno virtual .venv. Ejecuta scripts\setup.cmd primero.
  exit /b 1
)
call .venv\Scripts\activate.bat

REM Crear carpeta de logs si no existe
if not exist "logs" mkdir "logs"
set "LOG=logs\uvicorn.log"

REM Iniciar API en segundo plano con logging
start "API" /min cmd /c "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 > %LOG% 2>&1"

REM Esperar a que el puerto responda (max 30s)
set /a MAX=30
set /a COUNT=0
echo Esperando arranque de API...
:waitloop
timeout /t 1 >nul
set /a COUNT+=1
where curl >nul 2>&1
if %errorlevel%==0 (
  curl -s http://127.0.0.1:8000/ | findstr /i "status" >nul 2>&1
  if %errorlevel%==0 goto predict
) else (
  powershell -NoProfile -Command "try { (Invoke-RestMethod -Uri 'http://127.0.0.1:8000/' -UseBasicParsing) | Out-File -FilePath '$env:TEMP\api_root.txt' -Encoding utf8 } catch { exit 1 }" >nul 2>&1
  if exist "%TEMP%\api_root.txt" (
    findstr /i "status" "%TEMP%\api_root.txt" >nul 2>&1 && del "%TEMP%\api_root.txt" && goto predict
  )
)
if %COUNT% GEQ %MAX% (
  echo ERROR: La API no respondio en / tras %MAX% segundos. Revisa %LOG%
  exit /b 1
)
goto waitloop

:predict
echo Probando /predict...
set "TMPJSON=%TEMP%\predict_req.json"
echo { "lat": -33.45, "lon": -70.65 } > "%TMPJSON%"
where curl >nul 2>&1
if %errorlevel%==0 (
  curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @"%TMPJSON%" | findstr /i "riesgo_predicho" >nul 2>&1
  if %errorlevel%==0 goto ok
) else (
  powershell -NoProfile -Command "try { $b=Get-Content -Raw '%TMPJSON%'; $r=Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method POST -Body $b -ContentType 'application/json'; ($r | ConvertTo-Json -Depth 5) | Out-File -FilePath '$env:TEMP\api_predict.txt' -Encoding utf8 } catch { exit 1 }" >nul 2>&1
  if exist "%TEMP%\api_predict.txt" (
    findstr /i "riesgo_predicho" "%TEMP%\api_predict.txt" >nul 2>&1 && del "%TEMP%\api_predict.txt" && goto ok
  )
)

echo ERROR: /predict fallo. Revisa %LOG%
del "%TMPJSON%" 2>nul
exit /b 1

:ok
echo OK - Smoke test completado.
del "%TMPJSON%" 2>nul
endlocal
exit /b 0
