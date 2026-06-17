# -*- coding: utf-8 -*-
"""Dashboard D3 — Brecha digital y territorio en Chile.
Alineado con la infografía A4 del reporte (mismos 6 gráficos) + análisis
espacial interactivo (Moran's I, clusters LISA, mapa con hover).

Correr local:  streamlit run app.py
Datos livianos precomputados por build_data.py (data/).
"""
import os, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ASSETS = os.path.join(ROOT, "assets")              # figuras del reporte (autocontenido)

st.set_page_config(page_title="Brecha digital y territorio en Chile",
                   page_icon="🗺️", layout="wide")

# Paleta del proyecto
NAVY = "#21456B"
CLUSTER_COLORS = {
    "Alto-Alto": "#b2182b", "Bajo-Bajo": "#2166ac", "Alto-Bajo": "#ef8a62",
    "Bajo-Alto": "#67a9cf", "No significativo": "#e6e2d8",
}
PANEL_FILES = [
    ("01_heatmap_horiz.png", "1 · Servicios de conectividad por región"),
    ("04_dumbbell_tall_legsmall.png", "2 · Acceso urbano vs rural"),
    ("03_bubble_narrow.png", "3 · Acceso por edad y nivel educativo"),
    ("02_mapa_secuencial.png", "4 · Internet fija por comuna"),
    ("06_radar_tight.png", "5 · Perfiles de conectividad regional"),
    ("05_slope_compact.png", "6 · Evolución del acceso (CASEN 2017–2024)"),
]


@st.cache_data
def load():
    df = pd.read_csv(os.path.join(DATA, "comunas.csv"))
    df["CUT"] = df["CUT"].astype(str)
    with open(os.path.join(DATA, "comunas.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        feat["id"] = str(feat["properties"]["CUT"])
    with open(os.path.join(DATA, "moran_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    return df, gj, stats


df, GEO, S = load()

# ---------------------------------------------------------------- Header
st.title("Brecha digital y territorio en Chile")
st.caption("Censo 2024 (INE) · CASEN 2017/2022/2024 (MDSF) · Visualización de Datos — "
           "Universidad de Concepción")
st.markdown(
    "La conectividad digital en Chile no se distribuye al azar: se **agrupa en el "
    "territorio**. Este panel acompaña la infografía del reporte y permite explorar "
    "la dimensión espacial comuna por comuna."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Moran's I (internet fija)", f"{S['moran_I']:.2f}", help="Autocorrelación espacial global")
c2.metric("Significancia", f"p < 0,001", help=f"{S['permutations']} permutaciones · z={S['moran_z']}")
c3.metric("Comunas analizadas", f"{S['n_comunas']}")
c4.metric("Promedio internet fija", f"{S['metric_mean']:.0f}%")

tab_rep, tab_plots, tab_spatial = st.tabs(
    ["📄 Reporte (A4)", "📊 Los 6 gráficos", "🗺️ Análisis espacial interactivo"])

# ---------------------------------------------------------------- Reporte
with tab_rep:
    st.subheader("Infografía del reporte")
    info = os.path.join(ASSETS, "d3_infografia.png")
    if os.path.exists(info):
        st.image(info, use_container_width=True)
        pdf = os.path.join(ASSETS, "d3_infografia.pdf")
        if os.path.exists(pdf):
            with open(pdf, "rb") as f:
                st.download_button("⬇️ Descargar PDF (A4)", f, "d3_infografia.pdf",
                                   "application/pdf")
    else:
        st.warning("Falta figuras/d3_infografia.png (correr el ensamble del reporte).")

# ---------------------------------------------------------------- 6 gráficos
with tab_plots:
    st.subheader("Los seis gráficos del reporte")
    st.caption("Las mismas figuras de la infografía A4 (izq. 1–3 = qué falta y a quién; "
               "der. 4–6 = dónde y cuándo).")
    cols = st.columns(2)
    for i, (fn, title) in enumerate(PANEL_FILES):
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            p = os.path.join(ASSETS, fn)
            if os.path.exists(p):
                st.image(p, use_container_width=True)
            else:
                st.warning(f"Falta {fn}")

# ---------------------------------------------------------------- Espacial
with tab_spatial:
    st.subheader("Autocorrelación espacial — Moran's I")
    st.markdown(
        f"El índice de Moran global es **{S['moran_I']:.2f}** (p < 0,001, "
        f"{S['permutations']} permutaciones): la conectividad **se agrupa "
        "significativamente** en el territorio. Abajo, dónde están esos clusters.")

    left, right = st.columns([3, 2])

    # --- Mapa coroplético interactivo ---
    with left:
        modo = st.radio("Mapa:", ["% internet fija", "Clusters LISA"], horizontal=True)
        if modo == "% internet fija":
            fig = px.choropleth(
                df, geojson=GEO, locations="CUT", color="tiene_internet_fija",
                color_continuous_scale="Blues", range_color=(0, 100),
                hover_name="comuna_nombre",
                hover_data={"region_nombre": True, "tiene_internet_fija": ":.1f",
                            "CUT": False},
                labels={"tiene_internet_fija": "% internet fija"})
        else:
            fig = px.choropleth(
                df, geojson=GEO, locations="CUT", color="lisa_label",
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"lisa_label": list(CLUSTER_COLORS)},
                hover_name="comuna_nombre",
                hover_data={"region_nombre": True, "tiene_internet_fija": ":.1f",
                            "CUT": False},
                labels={"lisa_label": "Cluster"})
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=-0.02))
        st.plotly_chart(fig, use_container_width=True)

    # --- Moran scatterplot ---
    with right:
        st.markdown("**Diagrama de dispersión de Moran**")
        st.caption("Eje X: conectividad (estandarizada). Eje Y: promedio de los vecinos. "
                   "La pendiente es el Moran's I; cada cuadrante es un tipo de cluster.")
        sc = px.scatter(
            df, x="z", y="lag_z", color="lisa_label",
            color_discrete_map=CLUSTER_COLORS,
            category_orders={"lisa_label": list(CLUSTER_COLORS)},
            hover_name="comuna_nombre",
            hover_data={"tiene_internet_fija": ":.1f", "z": False, "lag_z": False},
            labels={"z": "Internet fija (z)", "lag_z": "Vecinos (z)", "lisa_label": "Cluster"})
        # líneas de cuadrante + recta de pendiente = Moran's I
        xs = np.array([df["z"].min(), df["z"].max()])
        sc.add_trace(go.Scatter(x=xs, y=S["moran_I"] * xs, mode="lines",
                                line=dict(color=NAVY, width=2, dash="solid"),
                                name=f"Pendiente = I ({S['moran_I']:.2f})"))
        sc.add_hline(y=0, line=dict(color="#999", width=1))
        sc.add_vline(x=0, line=dict(color="#999", width=1))
        sc.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0),
                         legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        st.plotly_chart(sc, use_container_width=True)

    # --- Clusters: conteo + tabla ---
    st.markdown("##### Clusters significativos (LISA, p < 0,05)")
    cc = st.columns(5)
    order = ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto"]
    labels = {"Alto-Alto": "🔴 Alto-Alto (núcleos)", "Bajo-Bajo": "🔵 Bajo-Bajo (rezago)",
              "Alto-Bajo": "🟠 Alto-Bajo (outlier)", "Bajo-Alto": "🟡 Bajo-Alto (outlier)"}
    for col, k in zip(cc, order):
        col.metric(labels[k], S["clusters"][k])
    cc[4].metric("No significativo", S["no_sig"])

    with st.expander("Ver tabla de comunas"):
        reg = st.multiselect("Filtrar región", sorted(df["region_nombre"].unique()))
        show = df if not reg else df[df["region_nombre"].isin(reg)]
        st.dataframe(
            show[["comuna_nombre", "region_nombre", "tiene_internet_fija",
                  "total_hogares", "lisa_label"]]
            .rename(columns={"comuna_nombre": "Comuna", "region_nombre": "Región",
                             "tiene_internet_fija": "% internet fija",
                             "total_hogares": "Hogares", "lisa_label": "Cluster"})
            .sort_values("% internet fija", ascending=False),
            use_container_width=True, hide_index=True, height=360)

st.divider()
st.caption("Datos: Censo 2024 (INE) y CASEN 2017/2022/2024 (MDSF). "
           "Autocorrelación espacial: Moran's I (contigüidad Queen, 999 permutaciones).")
