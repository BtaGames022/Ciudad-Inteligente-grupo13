import requests
import streamlit as st
import os
import json
import pydeck as pdk
import pandas as pd
from typing import Dict, Any, Optional
import time

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Smart Route Optimizer", page_icon="🚦", layout="wide")

st.title("🚦 Smart Route Optimizer: Optimizador de Rutas Terrestres")

st.info("Prototipo educativo del hackathon. No reemplaza evaluación profesional. Datos de ejemplo.")

# Utils cache
@st.cache_data(ttl=60)
def api_get(url: str, params: Optional[Dict[str, Any]] = None):
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def api_post(url: str, payload: Dict[str, Any]):
    r = requests.post(url, json=payload, timeout=45)
    r.raise_for_status()
    return r.json()

# Sidebar: API and filters
with st.sidebar:
    st.header("Configuración")
    st.write("API URL actual:")
    API_URL = st.text_input("API URL", value=API_URL)
    st.caption("Si cambias la URL, pulsa Enter para aplicar.")

# Tabs
risk_tab, map_tab, coach_tab, history_tab, export_tab = st.tabs([
    "1. Análisis de Riesgo (ML)",
    "2. Mapa y Tráfico (Geo)",
    "3. Coach IA y Plan de Acción (RAG)",
    "4. Historial (BD)",
    "5. Exportar y Compartir"
])

# --- Tab 1: Predict ---
with risk_tab:
    st.subheader("Ingreso de Datos y Evaluación de Riesgo")
    col1, col2 = st.columns([1,1])
    with col1:
        lat = st.number_input("Latitud", value=-33.45, step=0.001, format="%.6f")
        lon = st.number_input("Longitud", value=-70.65, step=0.001, format="%.6f")
        if st.button("Calcular Riesgo Vial", type="primary"):
            try:
                data = api_post(f"{API_URL}/predict", {"lat": lat, "lon": lon})
                st.session_state["predict_result"] = data
            except Exception as e:
                st.error(f"No se pudo conectar a la API: {e}")
    with col2:
        if "predict_result" in st.session_state:
            d = st.session_state["predict_result"]
            st.metric("Riesgo Predicho", d["riesgo_predicho"], help="0: Esporádicos, 1: Comunes, 2: Muy Frecuentes")
            st.progress(min(max(float(d["probabilidad"]), 0.0), 1.0))
            st.write("Probabilidades por clase:")
            st.json({
                "Esporadicos": d["categoria_0_prob"],
                "Comunes": d["categoria_1_prob"],
                "Muy Frecuentes": d["categoria_2_prob"],
            })
            st.write("Contexto del punto (más cercano):")
            st.json({k: d[k] for k in ["comuna", "zona", "ubicacion_desc", "causa_comun", "frecuencia_total", "indice_severidad", "lat_encontrada", "lon_encontrada"]})

            # Mostrar accidentes cercanos en BD
            st.markdown("### Accidentes Cercanos (BD)")
            try:
                near = api_get(f"{API_URL}/near/accidentes", {"lat": d["lat_encontrada"], "lon": d["lon_encontrada"], "limit": 25}).get("items", [])
                st.dataframe(pd.DataFrame(near), use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudieron cargar accidentes cercanos: {e}")

# --- Tab 2: Map ---
with map_tab:
    st.subheader("Mapa de Accidentes, Riesgo Predicho y Densidad (Heat/Hex)")

    # Cargar comunas
    try:
        comunas = api_get(f"{API_URL}/comunas").get("comunas", [])
    except Exception:
        comunas = []
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        comuna_sel = st.selectbox("Filtrar por comuna (opcional)", options=["(todas)"] + comunas)
    with c2:
        include_pred = st.checkbox("Incluir predicción por punto", value=True)
    with c3:
        max_points = st.slider("Máx. puntos", min_value=100, max_value=5000, value=1000, step=100)

    bbox_expander = st.expander("Filtro por Bounding Box (opcional)")
    with bbox_expander:
        min_lat = st.number_input("min_lat", value=-33.6, step=0.01, format="%.4f")
        max_lat = st.number_input("max_lat", value=-33.3, step=0.01, format="%.4f")
        min_lon = st.number_input("min_lon", value=-70.8, step=0.01, format="%.4f")
        max_lon = st.number_input("max_lon", value=-70.5, step=0.01, format="%.4f")
        use_bbox = st.checkbox("Aplicar BBox si no hay comuna seleccionada", value=False)

    params = {"include_pred": 1 if include_pred else 0, "limit": max_points}
    if comuna_sel and comuna_sel != "(todas)":
        params["comuna"] = comuna_sel
    elif use_bbox:
        params.update({"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon})

    try:
        fc = api_get(f"{API_URL}/geo/points", params)
    except Exception as e:
        st.error(f"No se pudo cargar puntos: {e}")
        fc = {"type": "FeatureCollection", "features": []}

    feats = fc.get("features", [])
    rows = []
    for f in feats:
        coords = f.get("geometry", {}).get("coordinates", [None, None])
        prop = f.get("properties", {})
        rows.append({
            "lon": coords[0],
            "lat": coords[1],
            **prop,
        })
    df = pd.DataFrame(rows)

    # Estadísticas
    colA, colB, colC = st.columns(3)
    try:
        ss = api_get(f"{API_URL}/stats", {"comuna": None if comuna_sel == "(todas)" else comuna_sel})
        colA.metric("Total puntos", ss.get("total", 0))
        colB.metric("Severidad promedio", f"{ss.get('avg_severidad', 0):.2f}")
        colC.json(ss.get("por_clase", {}))
    except Exception:
        pass

    # Capas toggles
    st.markdown("### Capas")
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        show_points = st.checkbox("Puntos", value=True)
    with lc2:
        show_heat = st.checkbox("Heatmap", value=True)
    with lc3:
        show_hex = st.checkbox("Hexágonos", value=False)
    with lc4:
        center_on_last = st.checkbox("Centrar en último análisis", value=("predict_result" in st.session_state))

    # Preparar colores por riesgo
    def riesgo_to_color(row):
        label = row.get("riesgo_predicho")
        if label == "Esporadicos":
            return [0, 128, 255]
        if label == "Comunes":
            return [255, 165, 0]
        if label == "Muy Frecuentes":
            return [255, 0, 0]
        # fallback según severidad
        sev = float(row.get("indice_severidad", 0))
        base = min(255, 80 + int(sev * 50))
        return [base, 120, 120]

    if not df.empty:
        df["color"] = df.apply(riesgo_to_color, axis=1)
        # Vista inicial
        if center_on_last and "predict_result" in st.session_state:
            d = st.session_state["predict_result"]
            view_state = pdk.ViewState(latitude=float(d["lat_encontrada"]), longitude=float(d["lon_encontrada"]), zoom=13, pitch=0)
        else:
            view_state = pdk.ViewState(latitude=df["lat"].mean(), longitude=df["lon"].mean(), zoom=12, pitch=0)

        layers = []
        if show_heat:
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=df,
                get_position='["lon", "lat"]',
                aggregation='MEAN',
                get_weight='frecuencia_total',
            ))
        if show_hex:
            layers.append(pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position='["lon", "lat"]',
                radius=80,
                elevation_scale=2,
                elevation_range=[0, 1000],
                extruded=True,
                pickable=True,
                get_elevation_weight='frecuencia_total',
            ))
        if show_points:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='["lon", "lat"]',
                get_radius=50,
                get_fill_color='[color[0], color[1], color[2]]',
                pickable=True,
            ))

        tooltip = {"text": "{ubicacion_desc}\n{comuna}\nRiesgo: {riesgo_predicho}\nSeveridad: {indice_severidad}\nFrecuencia: {frecuencia_total}"}
        rmap = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)
        st.pydeck_chart(rmap, use_container_width=True)

        st.markdown("### Detalle de puntos")
        st.dataframe(df.drop(columns=["lon", "lat", "color"], errors="ignore"), use_container_width=True)

        # Export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar datos filtrados (.csv)", data=csv, file_name="puntos_filtrados.csv", mime="text/csv")
    else:
        st.warning("No hay puntos para mostrar con los filtros actuales.")

# --- Tab 3: Coach ---
with coach_tab:
    st.subheader("Plan de Acción y Sugerencias del Coach IA")
    if "predict_result" not in st.session_state:
        st.warning("Primero calcula el riesgo en la pestaña 'Análisis de Riesgo'.")
    else:
        d = st.session_state["predict_result"]
        # Selector manual de comuna (default la del análisis)
        comuna_plan = st.selectbox("Comuna para el Plan", options=[d["comuna"]] + list({d["comuna"], *(st.session_state.get('extra_comunas', []))}), index=0)
        ubicacion_desc = d["ubicacion_desc"]
        causa_comun = d["causa_comun"]
        indice_severidad = d["indice_severidad"]
        if st.button("Generar Plan (RAG)"):
            payload = {
                "comuna": comuna_plan,
                "ubicacion_desc": ubicacion_desc,
                "causa_comun": causa_comun,
                "indice_severidad": indice_severidad,
            }
            try:
                res = api_post(f"{API_URL}/coach?save=true", payload)
                st.session_state["coach_result"] = res
            except Exception as e:
                st.error(f"No se pudo conectar a la API: {e}")
        if "coach_result" in st.session_state:
            c = st.session_state["coach_result"]
            st.subheader(c.get("titulo", "Plan"))
            st.markdown(c.get("plan_de_accion", ""))
            st.caption(f"Fuente: {c.get('fuente', 'N/A')}")

# --- Tab 4: Historial ---
with history_tab:
    st.subheader("Historial de Planes y Predicciones (BD)")
    filt_col1, filt_col2, filt_col3 = st.columns([1,1,1])
    with filt_col1:
        comuna_hist = st.text_input("Filtrar por comuna (opcional)", value="")
    with filt_col2:
        limite = st.slider("Límite", min_value=10, max_value=500, value=100, step=10)
    with filt_col3:
        refrescar = st.button("Refrescar")

    try:
        planes = api_get(f"{API_URL}/planes/list", {"comuna": comuna_hist or None, "limit": limite, "t": time.time() if refrescar else None}).get("planes", [])
    except Exception as e:
        st.error(f"No se pudo cargar planes: {e}")
        planes = []
    try:
        preds = api_get(f"{API_URL}/predicciones/list", {"comuna": comuna_hist or None, "limit": limite, "t": time.time() if refrescar else None}).get("predicciones", [])
    except Exception as e:
        st.error(f"No se pudo cargar predicciones: {e}")
        preds = []

    cA, cB = st.columns(2)
    with cA:
        st.markdown("### Planes de Acción Guardados")
        st.dataframe(pd.DataFrame(planes), use_container_width=True)
    with cB:
        st.markdown("### Predicciones Recientes")
        st.dataframe(pd.DataFrame(preds), use_container_width=True)

    st.markdown("---")
    st.subheader("Datos de Accidentes en BD y Scoring del Modelo")
    fa1, fa2, fa3, fa4, fa5 = st.columns([1,1,1,1,1])
    with fa1:
        comuna_f = st.text_input("Comuna (opcional)", value="")
    with fa2:
        tipo_f = st.text_input("Tipo de accidente (opcional)", value="")
    with fa3:
        gmin = st.number_input("Gravedad min (opcional)", value=0, step=1)
    with fa4:
        gmax = st.number_input("Gravedad max (opcional)", value=10, step=1)
    with fa5:
        lim = st.slider("Límite filas", min_value=10, max_value=2000, value=200, step=10)

    b1, b2 = st.columns([1,1])
    with b1:
        if st.button("Cargar Accidentes BD"):
            try:
                params = {
                    "comuna": (comuna_f or None),
                    "tipo": (tipo_f or None),
                    "gravedad_min": int(gmin) if gmin is not None else None,
                    "gravedad_max": int(gmax) if gmax is not None else None,
                    "limit": int(lim),
                }
                data = api_get(f"{API_URL}/accidentes", params).get("accidentes", [])
                st.session_state["accidentes_raw"] = pd.DataFrame(data)
            except Exception as e:
                st.error(f"Error al cargar accidentes: {e}")
    with b2:
        if st.button("Calcular Score en Accidentes"):
            try:
                params = {
                    "comuna": (comuna_f or None),
                    "tipo": (tipo_f or None),
                    "gravedad_min": int(gmin) if gmin is not None else None,
                    "gravedad_max": int(gmax) if gmax is not None else None,
                    "limit": int(lim),
                }
                items = api_get(f"{API_URL}/score/accidentes", params).get("items", [])
                st.session_state["accidentes_score"] = pd.DataFrame(items)
            except Exception as e:
                st.error(f"Error al calcular score: {e}")

    if "accidentes_raw" in st.session_state:
        st.markdown("### Accidentes (BD)")
        df_raw = st.session_state["accidentes_raw"]
        st.dataframe(df_raw, use_container_width=True)
        st.download_button("Descargar Accidentes (.csv)", data=df_raw.to_csv(index=False).encode("utf-8"), file_name="accidentes_bd.csv", mime="text/csv")

    if "accidentes_score" in st.session_state:
        st.markdown("### Accidentes con Predicción del Modelo")
        df_sc = st.session_state["accidentes_score"]
        st.dataframe(df_sc, use_container_width=True)
        st.download_button("Descargar Scoring (.csv)", data=df_sc.to_csv(index=False).encode("utf-8"), file_name="accidentes_scoring.csv", mime="text/csv")

# --- Tab 5: Export ---
with export_tab:
    st.subheader("Descargar Plan y Compartir")
    if "coach_result" in st.session_state:
        import io
        from datetime import datetime, timezone
        c = st.session_state["coach_result"]
        contenido = "\n".join([
            f"Plan generado: {c.get('titulo','')}",
            "",
            c.get('plan_de_accion', ''),
            "",
            f"Fuente: {c.get('fuente','N/A')}",
            f"Generado: {datetime.now(timezone.utc).isoformat()}"
        ])
        b = io.BytesIO(contenido.encode("utf-8"))
        st.download_button("Descargar Plan (.txt)", b, file_name="plan_optimizador.txt", mime="text/plain")
    else:
        st.info("Genere el plan primero en la pestaña 3.")
