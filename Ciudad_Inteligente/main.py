from fastapi import FastAPI
import os
import sys

try:
    from dotenv import load_dotenv  # type: ignore
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    _ENV = os.path.join(_ROOT, '.env')
    if os.path.exists(_ENV):
        load_dotenv(_ENV, override=False)
except Exception:
    pass

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"

    def ensure_venv():
        if sys.prefix == sys.base_prefix and not os.environ.get('VIRTUAL_ENV'):
            print("[ADVERTENCIA] Parece que no estas en el entorno virtual (.venv). Activalo antes: .venv\\Scripts\\activate")
    ensure_venv()

    if mode == "api":
        # Arranca FastAPI con uvicorn
        os.environ.setdefault("UVICORN_HOST", "0.0.0.0")
        os.environ.setdefault("UVICORN_PORT", "8000")
        host = os.environ.get("UVICORN_HOST", "0.0.0.0")
        port = os.environ.get("UVICORN_PORT", "8000")
        print(f"Iniciando API en {host}:{port} ...")
        exit_code = os.system(f"uvicorn api.main:app --host {host} --port {port}")
        if exit_code != 0:
            print("[ERROR] uvicorn terminó con un código distinto de cero.")
    elif mode == "app":
        # Arranca Streamlit
        os.environ.setdefault("API_URL", "http://127.0.0.1:8000")
        port = os.environ.get("STREAMLIT_PORT", "8501")
        print(f"Iniciando Streamlit en puerto {port} apuntando a API_URL={os.environ['API_URL']}")
        exit_code = os.system(f"streamlit run app/streamlit_app.py --server.port {port}")
        if exit_code != 0:
            print("[ERROR] Streamlit terminó con un código distinto de cero.")
    elif mode == "diag":
        # Diagnóstico rápido
        os.system("python scripts/diagnose_env.py")
    else:
        print("Uso: python main.py [api|app|diag]")
