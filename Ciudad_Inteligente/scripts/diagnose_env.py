#!/usr/bin/env python
"""
Diagnostico del entorno Python para el proyecto.

Ejecuta:
  python scripts/diagnose_env.py

Muestra:
- Version de Python, ejecutable y prefijos
- Variables de entorno relevantes (PYTHONHOME, PYTHONPATH, VIRTUAL_ENV)
- Directorios sys.path
- Presencia de site-packages dentro del venv
- Intento de import de dependencias clave
- Recomendaciones si detecta problemas
"""
from __future__ import annotations
import os
import sys
import importlib
from pathlib import Path

CRITICAL = ["fastapi", "uvicorn", "pandas", "numpy", "joblib"]


def check_module(name: str):
    try:
        importlib.import_module(name)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    print("== PYTHON INFO ==")
    print("Executable:", sys.executable)
    print("Version:", sys.version.replace("\n", " "))
    print("Prefix:", sys.prefix)
    print("Base Prefix:", sys.base_prefix)
    print()

    print("== ENV VARS ==")
    for k in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "TNS_ADMIN", "ORACLE_DSN"]:
        if k in os.environ:
            print(f"{k}={os.environ[k]}")
        else:
            print(f"{k}=(no definida)")
    print()

    print("== sys.path (rutas de busqueda) ==")
    for p in sys.path:
        print(" ", p)
    print()

    # Detectar si estamos en un venv
    in_venv = sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))
    print("En entorno virtual:", in_venv)

    if in_venv:
        # Ubicacion esperada de site-packages
        sp_candidates = [Path(sys.prefix) / 'Lib' / 'site-packages', Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages']
        found_sp = [p for p in sp_candidates if p.is_dir()]
        print("Site-packages encontrados:")
        for p in found_sp:
            print("  ", p)
        if not found_sp:
            print("  (No se encontro directorio site-packages: posible corrupcion de venv)")
    else:
        print("ADVERTENCIA: No parece estar activo el entorno virtual .venv")
    print()

    print("== Import de dependencias clave ==")
    missing = []
    for mod in CRITICAL:
        ok, err = check_module(mod)
        if ok:
            print(f"{mod}: OK")
        else:
            print(f"{mod}: FALLO - {err}")
            missing.append(mod)
    print()

    # Heuristicas para error 'Could not find platform independent libraries <prefix>'
    print("== Analisis de errores comunes ==")
    pyhome = os.environ.get("PYTHONHOME")
    if pyhome:
        print("SUGERENCIA: Quita PYTHONHOME antes de activar el venv (set PYTHONHOME=)")
    if not in_venv:
        print("SUGERENCIA: Activa el venv: .venv\\Scripts\\activate")
    if missing:
        print("SUGERENCIA: Reinstala dependencias faltantes: pip install -r requirements.txt")
    print("Diagnostico completado.")

if __name__ == "__main__":
    main()

