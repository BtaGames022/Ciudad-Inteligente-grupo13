import os
import re
import joblib
import pandas as pd
import numpy as np
try:
    from dotenv import load_dotenv  # type: ignore
    # Cargar .env en la raiz del proyecto si existe (no sobreescribe existentes)
    _ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(_ENV_PATH):
        load_dotenv(_ENV_PATH, override=False)
except Exception:
    pass
from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from .db import OracleDB

print("🚀 Iniciando API del Optimizador de Rutas...")

# --- 1) Modelos de Datos ---
class RutaRequest(BaseModel):
    lat: float
    lon: float

class CoachRequest(BaseModel):
    comuna: str
    ubicacion_desc: str
    causa_comun: str
    indice_severidad: int

class CoachResponse(BaseModel):
    titulo: str
    plan_de_accion: str
    fuente: str

class RutaResponse(BaseModel):
    riesgo_predicho: str
    probabilidad: float
    categoria_0_prob: float
    categoria_1_prob: float
    categoria_2_prob: float
    lat_encontrada: float
    lon_encontrada: float
    comuna: str
    zona: str
    ubicacion_desc: str
    causa_comun: str
    frecuencia_total: int
    indice_severidad: int
    # Campos opcionales cuando el origen es BD Oracle
    id_accidente: Optional[int] = None
    fecha_accidente: Optional[str] = None
    hora_accidente: Optional[str] = None
    tipo_accidente: Optional[str] = None
    condicion_climatica: Optional[str] = None

# --- 2) Carga de recursos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATOS_DIR = os.path.join(ROOT_DIR, "datos")
KB_DIR = os.path.join(ROOT_DIR, "kb")

MODEL_PATH = os.path.join(DATOS_DIR, "motor_riesgo.joblib")
DATA_PATH = os.path.join(DATOS_DIR, "Siniestros_Maestro_Consolidado_Hackathon.csv")
KB_PATH = os.path.join(KB_DIR, "fichas.md")

pipeline = None
df_maestro: Optional[pd.DataFrame] = None
kb_local = None

MODEL_DIR = os.path.join(ROOT_DIR, "modelo pre entrenado actualizado")
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocesador_v1.joblib")
MODEL_V1_PATH = os.path.join(MODEL_DIR, "modelo_riesgo_v1.joblib")

preproc = None
model_v1 = None


def cargar_kb_local(path_kb: str):
    kb = {}
    try:
        with open(path_kb, 'r', encoding='utf-8') as f:
            contenido = f.read()
        fichas = re.split(r'# CAUSA: ', contenido)
        for ficha in fichas:
            if not ficha.strip():
                continue
            partes = ficha.split('\n', 1)
            llave = partes[0].strip().upper()
            contenido_ficha = partes[1].strip() if len(partes) > 1 else ""
            fuente = "Fuente no especificada"
            m = re.search(r"\*Fuente:\*(.*)", contenido_ficha)
            if m:
                fuente = m.group(1).strip()
            kb[llave] = (contenido_ficha, fuente)
        print(f"✅ Base de Conocimiento (RAG) cargada. {len(kb)} fichas.")
        return kb
    except Exception as e:
        print(f"❌ Error al cargar KB: {e}")
        return None

# Cargar modelo y datos si existen
if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Modelo legacy (motor_riesgo.joblib) cargado.")
    except Exception as e:
        print(f"❌ No se pudo cargar el modelo legacy: {e}")
else:
    print(f"⚠️ No se encontró el modelo legacy en {MODEL_PATH}. Se intentará usar modelo actualizado o mock.")

# Carga del modelo actualizado (preprocesador + modelo v1)
if os.path.exists(PREPROC_PATH) and os.path.exists(MODEL_V1_PATH):
    try:
        preproc = joblib.load(PREPROC_PATH)
        model_v1 = joblib.load(MODEL_V1_PATH)
        print("✅ Modelo actualizado v1 y preprocesador cargados.")
    except Exception as e:
        print(f"⚠️ No se pudo cargar modelo actualizado v1: {e}")
else:
    print(f"ℹ️ Archivos de modelo v1 no encontrados en {MODEL_DIR}. Usando pipeline legacy o mock.")

if os.path.exists(DATA_PATH):
    try:
        df_maestro = pd.read_csv(DATA_PATH)
        print(f"✅ CSV maestro cargado: {len(df_maestro)} filas.")
    except Exception as e:
        print(f"❌ No se pudo cargar CSV maestro: {e}")
else:
    print(f"⚠️ No se encontró el CSV maestro en {DATA_PATH}. Se usarán datos sintéticos para demo.")
    # Mini dataset sintético para demo/local
    df_maestro = pd.DataFrame([
        {"Latitude": -33.45, "Longitude": -70.65, "COMUNA": "Santiago", "Zona": "Centro", "Ubicacion_Desc": "Alameda con Ahumada", "Causa__CON": "IMPRUDENCIA DEL CONDUCTOR", "Frecuencia_Total": 120, "Indice_Severidad": 3},
        {"Latitude": -33.48, "Longitude": -70.60, "COMUNA": "Ñuñoa", "Zona": "Oriente", "Ubicacion_Desc": "Irarrazaval con Pedro de Valdivia", "Causa__CON": "DESOBEDIENCIA A SEÑALIZACION", "Frecuencia_Total": 75, "Indice_Severidad": 2},
        {"Latitude": -33.51, "Longitude": -70.58, "COMUNA": "Macul", "Zona": "Suroriente", "Ubicacion_Desc": "Av. Macul con Quilin", "Causa__CON": "IMPRUDENCIA DEL PEATON", "Frecuencia_Total": 90, "Indice_Severidad": 2}
    ])

kb_local = cargar_kb_local(KB_PATH) if os.path.exists(KB_PATH) else None
if kb_local is None:
    print("⚠️ No se encontró KB. /coach funcionará con respuestas por defecto.")

# Inicializar DB si está configurada
oracle = OracleDB.from_env()
if oracle and oracle.is_ready():
    try:
        oracle.ensure_schema()
        print("✅ Esquema Oracle verificado/creado.")
    except Exception as e:
        print(f"⚠️ No se pudo inicializar esquema Oracle: {e}")
else:
    print("ℹ️ Oracle no configurado (defina ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN si desea usarlo).")

# --- 3) App FastAPI ---
app = FastAPI(title="Optimizador de Rutas Terrestres - Hackathon Duoc UC 2025",
              description="API: Predicción de riesgo vial (/predict) y Coach RAG (/coach)")

# CORS para frontends locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---

def _map_oracle_to_legacy_features(row: pd.Series) -> Dict[str, Any]:
    """Mapea columnas del esquema 'accidentes' a las usadas por el modelo legacy.
    No elimina las existentes; solo agrega claves faltantes con nombres legacy.
    """
    out = dict(row)
    if 'Latitude' not in out and 'latitud' in row:
        out['Latitude'] = row.get('latitud')
    if 'Longitude' not in out and 'longitud' in row:
        out['Longitude'] = row.get('longitud')
    if 'COMUNA' not in out:
        out['COMUNA'] = row.get('COMUNA') or row.get('comuna') or row.get('comuna_nombre') or ''
    if 'Zona' not in out:
        out['Zona'] = row.get('Zona') or ''
    if 'Ubicacion_Desc' not in out:
        out['Ubicacion_Desc'] = row.get('Ubicacion_Desc') or row.get('tipo_accidente') or ''
    if 'Causa__CON' not in out:
        out['Causa__CON'] = row.get('Causa__CON') or row.get('caus_principal') or ''
    if 'Frecuencia_Total' not in out:
        # usar num_fallecidos como proxy si no tenemos frecuencia
        out['Frecuencia_Total'] = row.get('Frecuencia_Total') if row.get('Frecuencia_Total') is not None else (row.get('num_fallecidos') or 0)
    if 'Indice_Severidad' not in out:
        out['Indice_Severidad'] = row.get('Indice_Severidad') if row.get('Indice_Severidad') is not None else (row.get('gravedad') or 0)
    return out


def predecir_fila(punto: pd.Series):
    """Devuelve (pred_clase:int, probs:np.ndarray[3]) para una fila del maestro o equivalente.
    Orden de preferencia: pipeline legacy -> (preproc+model_v1) -> mock por reglas.
    """
    # 0) Asegurar features legacy cuando sea posible
    punto_m = pd.Series(_map_oracle_to_legacy_features(punto))

    # 1) Pipeline legacy completo
    if pipeline is not None:
        X_pred = pd.DataFrame([punto_m])
        try:
            pred_clase = int(pipeline.predict(X_pred)[0])
            probs = pipeline.predict_proba(X_pred)[0].astype(float)
            return pred_clase, probs
        except Exception:
            pass
    # 2) Preprocesador + modelo actualizado
    if preproc is not None and model_v1 is not None:
        try:
            X_raw = pd.DataFrame([punto_m])
            X_t = preproc.transform(X_raw)
            if hasattr(model_v1, 'predict_proba'):
                probs = model_v1.predict_proba(X_t)[0]
                import numpy as _np
                probs = _np.array(probs, dtype=float)
                pred_clase = int(probs.argmax())
            else:
                pred_clase = int(model_v1.predict(X_t)[0])
                import numpy as _np
                probs = _np.zeros(3, dtype=float)
                probs[pred_clase] = 1.0
            return pred_clase, probs
        except Exception:
            pass
    # 3) Mock simple (Frecuencia_Total o gravedad/fallecidos)
    ft_val = punto.get('Frecuencia_Total', None)
    if ft_val is None:
        gravedad = punto.get('gravedad', 0)
        falle = punto.get('num_fallecidos', 0)
        try:
            gravedad = float(gravedad) if gravedad is not None else 0.0
        except Exception:
            gravedad = 0.0
        try:
            falle = float(falle) if falle is not None else 0.0
        except Exception:
            falle = 0.0
        ft = gravedad * 40.0 + falle * 60.0
    else:
        try:
            ft = float(ft_val)
        except Exception:
            ft = 50.0
    if ft > 100:
        pred_clase = 2
        probs = np.array([0.1, 0.2, 0.7])
    elif ft > 70:
        pred_clase = 1
        probs = np.array([0.2, 0.6, 0.2])
    else:
        pred_clase = 0
        probs = np.array([0.6, 0.3, 0.1])
    return pred_clase, probs


def _to_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        # Intento convertir strings numerables, si no, default
        try:
            return int(float(val))
        except Exception:
            return default


def _to_float(val, default: Optional[float] = None) -> Optional[float]:
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


# --- 4) Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "api": "Optimizador de Rutas API"}


@app.post("/predict", response_model=RutaResponse)
async def predict_risk(req: RutaRequest, request: Request):
    # Si Oracle está disponible, buscar nearest accidente en BD; sino usar CSV
    punto = None
    accidente_id = None
    if oracle and oracle.is_ready():
        try:
            nearest = oracle.fetch_nearest_accidente(req.lat, req.lon)
            if nearest:
                punto = pd.Series(nearest)
                accidente_id = nearest.get("id_accidente")
        except Exception as e:
            print(f"⚠️ Error fetch_nearest_accidente Oracle: {e}")
    if punto is None:
        if df_maestro is None or df_maestro.empty:
            raise HTTPException(status_code=500, detail="Datos no disponibles")
        distancias = np.sqrt((df_maestro['Latitude'] - req.lat) ** 2 + (df_maestro['Longitude'] - req.lon) ** 2)
        idx = distancias.idxmin()
        punto = df_maestro.loc[idx]
    pred_clase, probs = predecir_fila(punto)
    mapa = {0: 'Esporadicos', 1: 'Comunes', 2: 'Muy Frecuentes'}
    # Construir respuesta
    id_acc = None
    fecha_acc = None
    hora_acc = None
    tipo_acc = None
    clima = None
    if isinstance(punto, pd.Series):
        id_acc = punto.get('id_accidente')
        fecha_acc = punto.get('fecha_accidente')
        hora_acc = punto.get('hora_accidente')
        tipo_acc = punto.get('tipo_accidente', punto.get('Ubicacion_Desc', ''))
        clima = punto.get('condicion_climatica')
    lat_val = punto['Latitude'] if isinstance(punto, pd.Series) and 'Latitude' in punto else punto.get('Latitude')
    lon_val = punto['Longitude'] if isinstance(punto, pd.Series) and 'Longitude' in punto else punto.get('Longitude')
    respuesta = RutaResponse(
        riesgo_predicho=mapa.get(pred_clase, 'Desconocido'),
        probabilidad=float(probs[pred_clase]),
        categoria_0_prob=float(probs[0]),
        categoria_1_prob=float(probs[1]),
        categoria_2_prob=float(probs[2]),
        lat_encontrada=_to_float(lat_val, default=0.0),
        lon_encontrada=_to_float(lon_val, default=0.0),
        comuna=str(punto.get('COMUNA', punto.get('comuna', ''))),
        zona=str(punto.get('Zona', '')),
        ubicacion_desc=str(punto.get('tipo_accidente', punto.get('Ubicacion_Desc', ''))),
        causa_comun=str(punto.get('Causa__CON', '')),
        frecuencia_total=_to_int(punto.get('Frecuencia_Total', punto.get('num_fallecidos', 0)), default=0),
        indice_severidad=_to_int(punto.get('gravedad', punto.get('Indice_Severidad', 0))),
        id_accidente=int(id_acc) if id_acc is not None else None,
        fecha_accidente=str(fecha_acc) if fecha_acc is not None else None,
        hora_accidente=str(hora_acc) if hora_acc is not None else None,
        tipo_accidente=str(tipo_acc) if tipo_acc is not None else None,
        condicion_climatica=str(clima) if clima is not None else None,
    )
    # Log BD
    try:
        if oracle and oracle.is_ready():
            oracle.log_prediction({
                'lat': float(req.lat),
                'lon': float(req.lon),
                'id_accidente': accidente_id,
                'lat_encontrada': respuesta.lat_encontrada,
                'lon_encontrada': respuesta.lon_encontrada,
                'riesgo_predicho': respuesta.riesgo_predicho,
                'probabilidad': respuesta.probabilidad,
                'categoria_0_prob': respuesta.categoria_0_prob,
                'categoria_1_prob': respuesta.categoria_1_prob,
                'categoria_2_prob': respuesta.categoria_2_prob,
                'comuna': respuesta.comuna,
                'ubicacion_desc': respuesta.ubicacion_desc,
                'causa_comun': respuesta.causa_comun,
                'indice_severidad': respuesta.indice_severidad,
                'cliente_ip': request.client.host if request and request.client else None
            })
    except Exception as e:
        print(f"⚠️ Error al guardar predicción: {e}")
    return respuesta


@app.post("/coach", response_model=CoachResponse)
async def coach(req: CoachRequest, save: bool = Query(True)):
    causa = (req.causa_comun or "").upper()
    titulo = f"Plan de Acción para: {req.ubicacion_desc} ({req.comuna})"

    plan = "No se encontró un plan en la KB."
    fuente = "N/A"

    if kb_local and len(kb_local) > 0:
        if causa in kb_local:
            plan, fuente = kb_local[causa]
        else:
            # heurísticas por palabra clave
            if "IMPRUDENCIA" in causa:
                plan, fuente = kb_local.get("IMPRUDENCIA DEL CONDUCTOR", (plan, fuente))
            elif "PEATON" in causa:
                plan, fuente = kb_local.get("IMPRUDENCIA DEL PEATON", (plan, fuente))
            elif "ALCOHOL" in causa:
                plan, fuente = kb_local.get("ALCOHOL EN CONDUCTOR", (plan, fuente))
            elif "DESOBEDIENCIA" in causa:
                plan, fuente = kb_local.get("DESOBEDIENCIA A SEÑALIZACION", (plan, fuente))
            elif "CALZADA" in causa:
                plan, fuente = kb_local.get("CALZADA RESBALADIZA", (plan, fuente))
            else:
                plan, fuente = kb_local.get("OTRO", (plan, fuente))

    response = CoachResponse(titulo=titulo, plan_de_accion=plan, fuente=fuente)
    # Guardar plan en BD si corresponde
    if save and oracle and oracle.is_ready():
        try:
            oracle.save_plan({
                'comuna': req.comuna,
                'ubicacion_desc': req.ubicacion_desc,
                'causa_comun': req.causa_comun,
                'indice_severidad': req.indice_severidad,
                'titulo': titulo,
                'contenido_plan': plan,
                'fuente': fuente
            })
        except Exception as e:
            print(f"⚠️ Error al guardar plan: {e}")
    return response


@app.get("/db/health")
def db_health():
    if oracle and oracle.is_ready():
        try:
            with oracle.connect() as con:
                con.ping()
            return {"oracle": "ok"}
        except Exception as e:
            return {"oracle": "error", "detail": str(e)}
    return {"oracle": "disabled"}


@app.post("/admin/ingest")
def admin_ingest(limit: int = 10000):
    """Ingresa datos del CSV a la BD si ambas están disponibles."""
    if oracle is None or not oracle.is_ready():
        raise HTTPException(status_code=400, detail="Oracle no configurado")
    if df_maestro is None or df_maestro.empty:
        raise HTTPException(status_code=400, detail="CSV no cargado en memoria")
    try:
        inserted = oracle.ingest_df(df_maestro, limit=limit)
        return {"inserted": inserted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Overwrite endpoints to use DB if available for geo/stats/comunas
@app.get("/comunas")
def listar_comunas():
    if oracle and oracle.is_ready():
        try:
            return {"comunas": oracle.list_comunas()}
        except Exception as e:
            print(f"⚠️ Fallback a CSV comunas por error Oracle: {e}")
    # fallback CSV
    if df_maestro is None or df_maestro.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    comunas = (
        df_maestro["COMUNA"].dropna().astype(str).unique().tolist()
    )
    comunas = sorted(comunas)
    return {"comunas": comunas}


@app.get("/geo/points")
def geo_points(
    comuna: Optional[str] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
    include_pred: int = 1,
    limit: int = 2000,
):
    if oracle and oracle.is_ready():
        try:
            bbox = None
            if None not in (min_lat, max_lat, min_lon, max_lon):
                bbox = (float(min_lat), float(max_lat), float(min_lon), float(max_lon))
            rows = oracle.fetch_points(comuna, bbox, limit)
            mapa = {0: 'Esporadicos', 1: 'Comunes', 2: 'Muy Frecuentes'}
            features = []
            for r in rows:
                lat = _to_float(r.get("Latitude"))
                lon = _to_float(r.get("Longitude"))
                if lat is None or lon is None:
                    continue
                props = {
                    "id": r.get("ID"),
                    "comuna": r.get("COMUNA"),
                    "zona": r.get("Zona"),
                    "ubicacion_desc": r.get("Ubicacion_Desc"),
                    "causa_comun": r.get("Causa__CON"),
                    "frecuencia_total": _to_int(r.get("Frecuencia_Total", r.get("num_fallecidos", 0)), 0),
                    "indice_severidad": _to_int(r.get("Indice_Severidad", r.get("gravedad", 0)), 0),
                }
                if include_pred:
                    pred_clase, probs = predecir_fila(pd.Series(r))
                    props.update({
                        "riesgo_predicho": mapa.get(pred_clase, 'Desconocido'),
                        "categoria_0_prob": float(probs[0]),
                        "categoria_1_prob": float(probs[1]),
                        "categoria_2_prob": float(probs[2]),
                    })
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": props,
                })
            return {"type": "FeatureCollection", "features": features}
        except Exception as e:
            print(f"⚠️ Fallback a CSV geo_points por error Oracle: {e}")
    # fallback CSV
    if df_maestro is None or df_maestro.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    df = df_maestro
    if comuna:
        df = df[df["COMUNA"].astype(str).str.upper() == comuna.upper()]
    else:
        if None not in (min_lat, max_lat, min_lon, max_lon):
            df = df[
                (df["Latitude"] >= float(min_lat)) & (df["Latitude"] <= float(max_lat)) &
                (df["Longitude"] >= float(min_lon)) & (df["Longitude"] <= float(max_lon))
            ]

    if df.empty:
        return {"type": "FeatureCollection", "features": []}

    df = df.head(int(limit))

    features: List[Dict] = []
    mapa = {0: 'Esporadicos', 1: 'Comunes', 2: 'Muy Frecuentes'}
    for idx, row in df.iterrows():
        lat = _to_float(row.get("Latitude"))
        lon = _to_float(row.get("Longitude"))
        if lat is None or lon is None:
            continue
        props = {
            "id": int(idx),
            "comuna": str(row.get("COMUNA", "")),
            "zona": str(row.get("Zona", "")),
            "ubicacion_desc": str(row.get("Ubicacion_Desc", "")),
            "causa_comun": str(row.get("Causa__CON", "")),
            "frecuencia_total": _to_int(row.get("Frecuencia_Total", 0), 0),
            "indice_severidad": _to_int(row.get("Indice_Severidad", 0), 0),
        }
        if include_pred:
            pred_clase, probs = predecir_fila(row)
            props.update({
                "riesgo_predicho": mapa.get(pred_clase, 'Desconocido'),
                "categoria_0_prob": float(probs[0]),
                "categoria_1_prob": float(probs[1]),
                "categoria_2_prob": float(probs[2]),
            })
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


@app.get("/stats")
def stats(comuna: Optional[str] = None):
    if oracle and oracle.is_ready():
        try:
            base = oracle.stats(comuna)
            # para por_clase, si se requiere precisión, calcular con predicciones en memoria en esta versión
            # usaremos fallback simple en CSV para por_clase si es posible
        except Exception as e:
            print(f"⚠️ Fallback a CSV stats por error Oracle: {e}")
    # fallback CSV
    if df_maestro is None or df_maestro.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    df = df_maestro
    if comuna:
        df = df[df["COMUNA"].astype(str).str.upper() == comuna.upper()]
    if df.empty:
        return {"total": 0, "por_clase": {}, "avg_severidad": None}

    # Calcular predicciones por fila para estadísticas simples
    conteos = {"Esporadicos": 0, "Comunes": 0, "Muy Frecuentes": 0}
    severidades: List[int] = []
    for _, row in df.iterrows():
        pred, _ = predecir_fila(row)
        etiqueta = "Esporadicos" if pred == 0 else ("Comunes" if pred == 1 else "Muy Frecuentes")
        conteos[etiqueta] += 1
        try:
            severidades.append(int(row.get("Indice_Severidad", 0)))
        except Exception:
            pass

    avg_sev = float(np.mean(severidades)) if severidades else None
    return {"total": int(len(df)), "por_clase": conteos, "avg_severidad": avg_sev}


@app.get("/planes/list")
def planes_list(comuna: Optional[str] = None, causa: Optional[str] = None, limit: int = 100):
    if oracle and oracle.is_ready():
        try:
            return {'planes': oracle.list_planes(comuna, causa, limit)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {'planes': []}


@app.get("/predicciones/list")
def predicciones_list(comuna: Optional[str] = None, limit: int = 100):
    if oracle and oracle.is_ready():
        try:
            return {'predicciones': oracle.list_predicciones(comuna, limit)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {'predicciones': []}


@app.get("/accidentes")
def accidentes(
    comuna: Optional[str] = None,
    tipo: Optional[str] = None,
    gravedad_min: Optional[int] = None,
    gravedad_max: Optional[int] = None,
    limit: int = 200,
    offset: int = 0,
):
    if oracle and oracle.is_ready():
        try:
            data = oracle.list_accidentes(comuna, tipo, gravedad_min, gravedad_max, limit, offset)
            return {"accidentes": data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=503, detail="BD no disponible")


@app.get("/score/accidentes")
def score_accidentes(
    comuna: Optional[str] = None,
    tipo: Optional[str] = None,
    gravedad_min: Optional[int] = None,
    gravedad_max: Optional[int] = None,
    limit: int = 200,
    offset: int = 0,
):
    """Retorna accidentes con predicción del modelo para cada fila."""
    if oracle and oracle.is_ready():
        try:
            rows = oracle.list_accidentes(comuna, tipo, gravedad_min, gravedad_max, limit, offset)
            out = []
            mapa = {0: 'Esporadicos', 1: 'Comunes', 2: 'Muy Frecuentes'}
            for r in rows:
                s = pd.Series(r)
                pred, probs = predecir_fila(s)
                r.update({
                    'riesgo_predicho': mapa.get(pred, 'Desconocido'),
                    'categoria_0_prob': float(probs[0]),
                    'categoria_1_prob': float(probs[1]),
                    'categoria_2_prob': float(probs[2]),
                })
                out.append(r)
            return {'items': out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=503, detail="BD no disponible")


@app.get("/near/accidentes")
def near_accidentes(lat: float, lon: float, limit: int = 50):
    if oracle and oracle.is_ready():
        try:
            data = oracle.list_near_accidentes(lat, lon, limit)
            return {"items": data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=503, detail="BD no disponible")
