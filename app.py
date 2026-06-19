# -*- coding: utf-8 -*-
"""Dashboard D3 — Brecha digital y territorio en Chile.

Versión INTERACTIVA: los 6 gráficos del reporte reconstruidos en Plotly
(mismos datos, colores y orden que la infografía A4) + análisis espacial
interactivo (Moran's I, clusters LISA, mapa coroplético con hover).

Correr local:  streamlit run app.py
Datos livianos precomputados por:
  - build_data.py    -> data/comunas.* + moran_stats.json   (mapa / espacial)
  - build_panels.py  -> data/panel_*.csv                     (gráficos 1,2,3,5,6)
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

# ---------------------------------------------------------------- Paleta (= reporte)
NAVY = "#21456B"
FONDO = "#F4F0E8"
ACENTOS = {"Metropolitana": "#2C5F8A", "O'Higgins": "#C8702D", "La Araucanía": "#3C8C8C"}
C_URBANO, C_RURAL = "#2C5F8A", "#C8702D"
C_PROM, C_RESTO = "#333333", "#B8B2A6"
PALETA_BUBBLE = {"Básica o menos": "#d73027", "Media": "#fc8d59",
                 "Técnica/Sup. inc.": "#91bfdb", "Superior completa": "#4575b4"}
PROM_REGIONAL = 44.6
CLUSTER_COLORS = {
    "Alto-Alto": "#b2182b", "Bajo-Bajo": "#2166ac", "Alto-Bajo": "#ef8a62",
    "Bajo-Alto": "#67a9cf", "No significativo": "#e6e2d8",
}
ORDEN_REGIONES = [
    "Arica y Parinacota", "Tarapacá", "Antofagasta", "Atacama", "Coquimbo",
    "Valparaíso", "Metropolitana", "O'Higgins", "Maule", "Ñuble",
    "Biobío", "La Araucanía", "Los Ríos", "Los Lagos", "Aysén", "Magallanes",
]
ORDEN_SERVICIOS = ["Telefonía móvil", "Internet móvil", "Internet fija", "Computador", "Tablet"]
ABREV_X = {"Telefonía móvil": "Tel. móvil", "Internet móvil": "Int. móvil",
           "Internet fija": "Int. fija", "Computador": "Computador", "Tablet": "Tablet"}
YEARS = ["CASEN 2017", "CASEN 2022", "CASEN 2024"]


def _style(fig, h=430, legend_bottom=True):
    """Estilo común: transparente, márgenes chicos, leyenda horizontal abajo."""
    fig.update_layout(
        height=h, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2a2a2a", size=13))
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                      xanchor="left", x=0))
    return fig


# ---------------------------------------------------------------- Carga de datos
@st.cache_data
def load_spatial():
    df = pd.read_csv(os.path.join(DATA, "comunas.csv"))
    df["CUT"] = df["CUT"].astype(str)
    with open(os.path.join(DATA, "comunas.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        feat["id"] = str(feat["properties"]["CUT"])
    with open(os.path.join(DATA, "moran_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    return df, gj, stats


@st.cache_data
def load_panels():
    reg = pd.read_csv(os.path.join(DATA, "panel_regional.csv"))
    ra = pd.read_csv(os.path.join(DATA, "panel_region_area.csv"))
    bub = pd.read_csv(os.path.join(DATA, "panel_bubble.csv"))
    slo = pd.read_csv(os.path.join(DATA, "panel_slope.csv"))
    return reg, ra, bub, slo


df, GEO, S = load_spatial()
REG, RA, BUB, SLO = load_panels()
MATRIZ = REG.pivot(index="region_nombre", columns="servicio_nombre", values="porcentaje")


# ---------------------------------------------------------------- Constructores de gráficos
def fig_heatmap():
    piv = MATRIZ[ORDEN_SERVICIOS].loc[ORDEN_REGIONES]
    z = piv.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[ABREV_X[c] for c in piv.columns], y=list(piv.index),
        colorscale="Blues", zmin=0, zmax=100,
        hovertemplate="%{y} · %{x}<br>%{z:.0f}%<extra></extra>",
        colorbar=dict(title="%", thickness=12, len=0.9)))
    for i, rn in enumerate(piv.index):
        for j, cn in enumerate(piv.columns):
            v = z[i, j]
            fig.add_annotation(x=ABREV_X[cn], y=rn, text=f"{v:.0f}", showarrow=False,
                               font=dict(color="white" if v > 50 else "black", size=11))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 470, legend_bottom=False)


def fig_dumbbell():
    ra = RA.set_index("region_nombre").loc[ORDEN_REGIONES].reset_index()
    fig = go.Figure()
    for _, r in ra.iterrows():
        fig.add_trace(go.Scatter(x=[r["Rural"], r["Urbano"]], y=[r["region_nombre"]] * 2,
                                 mode="lines", line=dict(color=C_RESTO, width=2),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ra["Urbano"], y=ra["region_nombre"], mode="markers",
                             name="Urbano", marker=dict(color=C_URBANO, size=11),
                             hovertemplate="Urbano: %{x:.1f}%<extra>%{y}</extra>"))
    fig.add_trace(go.Scatter(x=ra["Rural"], y=ra["region_nombre"], mode="markers",
                             name="Rural", marker=dict(color=C_RURAL, size=11),
                             hovertemplate="Rural: %{x:.1f}%<extra>%{y}</extra>"))
    for _, r in ra.iterrows():
        fig.add_annotation(x=r["Urbano"], y=r["region_nombre"], text=f"+{r['brecha']:.0f} pp",
                           xshift=30, showarrow=False, font=dict(size=10, color="#333"))
    fig.add_vline(x=PROM_REGIONAL, line=dict(color=C_PROM, dash="dash", width=1.4))
    fig.add_annotation(x=PROM_REGIONAL, y=ORDEN_REGIONES[-1], yshift=-22,
                       text=f"Promedio regional ({PROM_REGIONAL:.1f}%)".replace(".", ","),
                       showarrow=False, font=dict(size=10, color=C_PROM), xanchor="left")
    fig.update_xaxes(title="Hogares con internet fija (%)", range=[0, 92],
                     showgrid=True, gridcolor="#e6e2d8")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 470)


def fig_bubble():
    fig = px.scatter(BUB, x="edad_quinquenal", y="pct_internet", size="n_hogares",
                     color="nivel", color_discrete_map=PALETA_BUBBLE,
                     category_orders={"nivel": list(PALETA_BUBBLE)}, size_max=34,
                     labels={"edad_quinquenal": "Edad del jefe de hogar (años)",
                             "pct_internet": "Hogares con internet fija (%)",
                             "nivel": "Nivel educativo", "n_hogares": "N° hogares"})
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color="white")))
    fig.update_xaxes(showgrid=True, gridcolor="#e6e2d8")
    fig.update_yaxes(showgrid=True, gridcolor="#e6e2d8")
    return _style(fig, 470)


def fig_choropleth(modo="% internet fija"):
    if modo == "% internet fija":
        fig = px.choropleth(df, geojson=GEO, locations="CUT", color="tiene_internet_fija",
                            color_continuous_scale="Blues", range_color=(0, 100),
                            hover_name="comuna_nombre",
                            hover_data={"region_nombre": True, "tiene_internet_fija": ":.1f",
                                        "CUT": False},
                            labels={"tiene_internet_fija": "% internet fija"})
    else:
        fig = px.choropleth(df, geojson=GEO, locations="CUT", color="lisa_label",
                            color_discrete_map=CLUSTER_COLORS,
                            category_orders={"lisa_label": list(CLUSTER_COLORS)},
                            hover_name="comuna_nombre",
                            hover_data={"region_nombre": True, "tiene_internet_fija": ":.1f",
                                        "CUT": False},
                            labels={"lisa_label": "Cluster"})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.02))
    return fig


def fig_radar():
    cats = ["Internet móvil", "Computador", "Internet fija", "Tablet"]
    fig = go.Figure()
    for region, color in ACENTOS.items():
        vals = [MATRIZ.loc[region, c] for c in cats]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], name=region, fill="toself",
            line=dict(color=color, width=2.5), fillcolor=color, opacity=0.55,
            hovertemplate="%{theta}: %{r:.0f}%<extra>" + region + "</extra>"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 95], tickvals=[20, 40, 60, 80], ticksuffix="%",
                        gridcolor="#d8d2c4"),
        angularaxis=dict(rotation=90, direction="clockwise", gridcolor="#d8d2c4")))
    fig = _style(fig, 470, legend_bottom=False)
    # leyenda ABAJO (si no, choca con la etiqueta "Internet móvil" del eje superior)
    fig.update_layout(margin=dict(l=70, r=70, t=20, b=20),
                      legend=dict(orientation="h", yanchor="top", y=-0.02,
                                  xanchor="center", x=0.5))
    return fig


def fig_slope():
    fig = go.Figure()
    nat = SLO[SLO["region_nombre"] == "Promedio nacional"].iloc[0]
    fig.add_trace(go.Scatter(x=YEARS, y=[nat["pct_2017"], nat["pct_2022"], nat["pct_2024"]],
                             name="Promedio nacional", mode="lines+markers",
                             line=dict(color=C_PROM, dash="dash", width=2.5),
                             hovertemplate="%{x}: %{y:.0f}%<extra>Nacional</extra>"))
    # La Araucanía (24%) y O'Higgins (25%) casi coinciden en 2017: una arriba, otra abajo.
    TXTPOS = {"La Araucanía": "bottom center"}
    for region, color in ACENTOS.items():
        row = SLO[SLO["region_nombre"] == region].iloc[0]
        ys = [row["pct_2017"], row["pct_2022"], row["pct_2024"]]
        fig.add_trace(go.Scatter(x=YEARS, y=ys, name=region, mode="lines+markers+text",
                                 line=dict(color=color, width=2.6),
                                 text=[f"{v:.0f}%" for v in ys],
                                 textposition=TXTPOS.get(region, "top center"),
                                 textfont=dict(color=color, size=11),
                                 hovertemplate="%{x}: %{y:.0f}%<extra>" + region + "</extra>"))
    fig.update_yaxes(title="Hogares con internet fija (%)", range=[20, 82],
                     showgrid=True, gridcolor="#e6e2d8")
    return _style(fig, 470)


# ---------------------------------------------------------------- Header
st.title("Brecha digital y territorio en Chile")
st.caption("Censo 2024 (INE) · CASEN 2017/2022/2024 (MDSF) · Visualización de Datos — "
           "Universidad de Concepción")
st.markdown(
    "La conectividad digital en Chile no se distribuye al azar: se **agrupa en el "
    "territorio**. Este panel reproduce de forma **interactiva** los seis gráficos del "
    "reporte y permite explorar la dimensión espacial comuna por comuna.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Moran's I (internet fija)", f"{S['moran_I']:.2f}", help="Autocorrelación espacial global")
c2.metric("Significancia", "p < 0,001", help=f"{S['permutations']} permutaciones · z={S['moran_z']}")
c3.metric("Comunas analizadas", f"{S['n_comunas']}")
c4.metric("Promedio internet fija", f"{S['metric_mean']:.0f}%")

tab_plots, tab_spatial, tab_rep = st.tabs(
    ["📊 Los 6 gráficos (interactivos)", "🗺️ Análisis espacial interactivo", "📄 Reporte A4"])

# ---------------------------------------------------------------- Los 6 gráficos
with tab_plots:
    st.caption("Las mismas seis figuras del reporte, ahora interactivas (pasa el mouse para "
               "ver valores exactos; usa la leyenda para aislar series). Izq. 1–3 = qué falta "
               "y a quién · der. 4–6 = dónde y cuándo.")

    r1l, r1r = st.columns(2)
    with r1l:
        st.markdown("**1 · Servicios de conectividad por región**")
        st.caption("% de hogares con cada servicio (Censo 2024). La brecha se concentra en "
                   "internet fija y equipamiento; la telefonía móvil es casi universal.")
        st.plotly_chart(fig_heatmap(), width="stretch", key="hm")
    with r1r:
        st.markdown("**4 · Internet fija por comuna**")
        st.caption("Mapa coroplético (343 comunas). El % de hogares con internet fija se "
                   "agrupa en el espacio (Moran's I = 0,48). Detalle y clusters en la pestaña espacial.")
        st.plotly_chart(fig_choropleth(), width="stretch", key="map_main")

    r2l, r2r = st.columns(2)
    with r2l:
        st.markdown("**2 · Acceso urbano vs rural**")
        st.caption("Internet fija por región: hogares urbanos (navy) vs rurales (terracota). "
                   "En las 16 regiones lo urbano supera a lo rural (brecha en pp a la derecha).")
        st.plotly_chart(fig_dumbbell(), width="stretch", key="db")
    with r2r:
        st.markdown("**5 · Perfiles de conectividad regional**")
        st.caption("Radar de 4 servicios para 3 regiones contrastantes. Metropolitana domina "
                   "internet fija y computador; las diferencias se cierran en internet móvil.")
        st.plotly_chart(fig_radar(), width="stretch", key="radar")

    r3l, r3r = st.columns(2)
    with r3l:
        st.markdown("**3 · Acceso por edad y nivel educativo**")
        st.caption("Cada burbuja = un grupo (edad del jefe de hogar × nivel educativo); "
                   "el tamaño es el N° de hogares. El acceso sube con el nivel educativo.")
        st.plotly_chart(fig_bubble(), width="stretch", key="bub")
    with r3r:
        st.markdown("**6 · Evolución del acceso (CASEN 2017–2024)**")
        st.caption("Crecimiento del % con internet fija. Todas las regiones suben, pero las "
                   "que partieron atrás siguen últimas: la brecha se reduce, sin cerrarse.")
        st.plotly_chart(fig_slope(), width="stretch", key="slope")

# ---------------------------------------------------------------- Espacial
with tab_spatial:
    st.subheader("Autocorrelación espacial — Moran's I")
    st.markdown(
        f"El índice de Moran global es **{S['moran_I']:.2f}** (p < 0,001, "
        f"{S['permutations']} permutaciones): la conectividad **se agrupa "
        "significativamente** en el territorio. Abajo, dónde están esos clusters.")

    left, right = st.columns([3, 2])
    with left:
        modo = st.radio("Mapa:", ["% internet fija", "Clusters LISA"], horizontal=True)
        st.plotly_chart(fig_choropleth(modo), width="stretch", key="map_spatial")

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
        xs = np.array([df["z"].min(), df["z"].max()])
        sc.add_trace(go.Scatter(x=xs, y=S["moran_I"] * xs, mode="lines",
                                line=dict(color=NAVY, width=2, dash="solid"),
                                name=f"Pendiente = I ({S['moran_I']:.2f})"))
        sc.add_hline(y=0, line=dict(color="#999", width=1))
        sc.add_vline(x=0, line=dict(color="#999", width=1))
        sc.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0),
                         paper_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        st.plotly_chart(sc, width="stretch", key="moran")

    st.markdown("##### Clusters significativos (LISA, p < 0,05)")
    cc = st.columns(5)
    order = ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto"]
    labels = {"Alto-Alto": "🔴 Alto-Alto (núcleos)", "Bajo-Bajo": "🔵 Bajo-Bajo (rezago)",
              "Alto-Bajo": "🟠 Alto-Bajo (outlier)", "Bajo-Alto": "🟡 Bajo-Alto (outlier)"}
    for col, k in zip(cc, order):
        col.metric(labels[k], S["clusters"][k])
    cc[4].metric("No significativo", S["no_sig"])

    with st.expander("Ver tabla de comunas"):
        reg_sel = st.multiselect("Filtrar región", sorted(df["region_nombre"].unique()))
        show = df if not reg_sel else df[df["region_nombre"].isin(reg_sel)]
        st.dataframe(
            show[["comuna_nombre", "region_nombre", "tiene_internet_fija",
                  "total_hogares", "lisa_label"]]
            .rename(columns={"comuna_nombre": "Comuna", "region_nombre": "Región",
                             "tiene_internet_fija": "% internet fija",
                             "total_hogares": "Hogares", "lisa_label": "Cluster"})
            .sort_values("% internet fija", ascending=False),
            width="stretch", hide_index=True, height=360)

# ---------------------------------------------------------------- Reporte A4
with tab_rep:
    st.subheader("Infografía del reporte (A4)")
    st.caption("La página única entregada en el reporte. El dashboard reproduce estos mismos "
               "seis gráficos de forma interactiva en la primera pestaña.")
    info = os.path.join(ASSETS, "d3_infografia.png")
    if os.path.exists(info):
        st.image(info, width="stretch")
        pdf = os.path.join(ASSETS, "d3_infografia.pdf")
        if os.path.exists(pdf):
            with open(pdf, "rb") as f:
                st.download_button("⬇️ Descargar PDF (A4)", f, "d3_infografia.pdf",
                                   "application/pdf")
    else:
        st.warning("Falta assets/d3_infografia.png (correr el ensamble del reporte).")

st.divider()
st.caption("Datos: Censo 2024 (INE) y CASEN 2017/2022/2024 (MDSF). "
           "Autocorrelación espacial: Moran's I (contigüidad Queen, 999 permutaciones).")
