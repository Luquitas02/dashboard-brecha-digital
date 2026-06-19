# -*- coding: utf-8 -*-
"""Dashboard D3 — Brecha digital y territorio en Chile.

Versión INTERACTIVA: los 6 gráficos del reporte reconstruidos en Plotly
(mismos datos, colores y orden que la infografía A4) + interacción cruzada
(selector de región que resalta varios gráficos) + análisis espacial
interactivo (Moran's I, clusters LISA, mapa con basemap, hover, histograma).

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
C_DESTACA = "#DB2777"   # magenta/rosa fuerte: único hue ausente de la paleta -> resalta en TODOS
                        # los gráficos (incl. radar, donde lo cálido se camufla). No es morado.
PALETA_BUBBLE = {"Básica o menos": "#d73027", "Media": "#fc8d59",
                 "Técnica/Sup. inc.": "#91bfdb", "Superior completa": "#4575b4"}
PROM_REGIONAL = 44.6
CLUSTER_COLORS = {
    "Alto-Alto": "#b2182b", "Bajo-Bajo": "#2166ac", "Alto-Bajo": "#ef8a62",
    "Bajo-Alto": "#67a9cf", "No significativo": "#BDB5A4",   # gris cálido visible sobre el crema
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

# Síntesis = misma leyenda-resumen del reporte A4 (mantiene alineación con el reporte).
SINTESIS = (
    "La telefonía móvil es casi universal (97–99%): la brecha digital se concentra en la "
    "internet fija y el equipamiento del hogar, donde las regiones más se distancian. Es una "
    "brecha territorial que se agrupa en el espacio de forma significativa (Moran's I = 0,48; "
    "p < 0,001): un núcleo central conectado frente a un norte y un extremo sur rezagados. Y es "
    "interna: en las 16 regiones los hogares urbanos superan a los rurales; y social, porque el "
    "acceso acompaña el nivel educativo y la edad del jefe de hogar. Entre 2017 y 2024 el acceso "
    "crece en todas las regiones, pero las que partieron atrás siguen últimas: la brecha se "
    "reduce, sin cerrarse.")


def coma(x, d=2):
    """Formatea con coma decimal (convención en español)."""
    return f"{x:.{d}f}".replace(".", ",")


def _rgba(hex_color, a):
    """Convierte '#RRGGBB' + alpha a 'rgba(r,g,b,a)' (para rellenos translúcidos)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _style(fig, h=430, legend_bottom=True):
    """Estilo común: transparente, márgenes chicos, leyenda horizontal arriba."""
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


@st.cache_data
def region_views(_gj):
    """Centro y zoom aproximados por región para hacer zoom a la región resaltada.
    Usa el centroide de cada comuna y mediana/percentiles (robusto a comunas-isla
    lejanas, ej. Juan Fernández en Valparaíso, que distorsionarían un bbox crudo)."""
    import math
    pts = {}

    def coords(c, xs, ys):
        if isinstance(c[0][0], (int, float)):          # anillo de [lon, lat]
            for lon, lat in c:
                xs.append(lon); ys.append(lat)
        else:
            for sub in c:
                coords(sub, xs, ys)

    for feat in _gj["features"]:
        rn = feat["properties"].get("region_nombre")
        if not rn:
            continue
        xs, ys = [], []
        coords(feat["geometry"]["coordinates"], xs, ys)
        pts.setdefault(rn, []).append((np.mean(xs), np.mean(ys)))  # centroide de la comuna
    views = {}
    for rn, arr in pts.items():
        lon = np.array([p[0] for p in arr]); lat = np.array([p[1] for p in arr])
        span = max(np.percentile(lon, 95) - np.percentile(lon, 5),
                   np.percentile(lat, 95) - np.percentile(lat, 5), 0.4)
        zoom = max(4.0, min(7.0, math.log2(360.0 / span) - 0.9))
        views[rn] = {"center": {"lat": float(np.median(lat)), "lon": float(np.median(lon))},
                     "zoom": zoom}
    return views


df, GEO, S = load_spatial()
REG, RA, BUB, SLO = load_panels()
MATRIZ = REG.pivot(index="region_nombre", columns="servicio_nombre", values="porcentaje")
SLO_REGIONS = set(SLO["region_nombre"])
VIEWS = region_views(GEO)


# ---------------------------------------------------------------- Constructores de gráficos
def fig_heatmap(destacar=None):
    piv = MATRIZ[ORDEN_SERVICIOS].loc[ORDEN_REGIONES]
    z = piv.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[ABREV_X[c] for c in piv.columns], y=list(piv.index),
        colorscale="Blues", zmin=0, zmax=100,
        hovertemplate="%{y} · %{x}<br>%{z:.0f}%<extra></extra>",
        colorbar=dict(title="% hogares", thickness=12, len=0.9)))
    for i, rn in enumerate(piv.index):
        for j, cn in enumerate(piv.columns):
            v = z[i, j]
            fig.add_annotation(x=ABREV_X[cn], y=rn, text=f"{v:.0f}", showarrow=False,
                               font=dict(color="white" if v > 50 else "black", size=11))
    if destacar in list(piv.index):
        pos = list(piv.index).index(destacar)
        fig.add_shape(type="rect", x0=-0.5, x1=len(piv.columns) - 0.5,
                      y0=pos - 0.5, y1=pos + 0.5,
                      line=dict(color=C_DESTACA, width=3))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 460, legend_bottom=False)


def fig_dumbbell(destacar=None):
    ra = RA.set_index("region_nombre").loc[ORDEN_REGIONES].reset_index()
    fig = go.Figure()
    if destacar in ORDEN_REGIONES:
        pos = ORDEN_REGIONES.index(destacar)
        fig.add_shape(type="rect", xref="x", yref="y", x0=0, x1=92,
                      y0=pos - 0.45, y1=pos + 0.45, layer="below",
                      fillcolor=C_DESTACA, opacity=0.12, line_width=0)
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
                       text=f"Promedio regional ({coma(PROM_REGIONAL, 1)}%)",
                       showarrow=False, font=dict(size=10, color=C_PROM), xanchor="left")
    fig.update_xaxes(title="Hogares con internet fija (%)", range=[0, 92],
                     showgrid=True, gridcolor="#D8CFBD")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 460)


def fig_bubble():
    fig = px.scatter(BUB, x="edad_quinquenal", y="pct_internet", size="n_hogares",
                     color="nivel", color_discrete_map=PALETA_BUBBLE,
                     category_orders={"nivel": list(PALETA_BUBBLE)}, size_max=34,
                     labels={"edad_quinquenal": "Edad del jefe de hogar (años)",
                             "pct_internet": "Hogares con internet fija (%)",
                             "nivel": "Nivel educativo", "n_hogares": "N° hogares"})
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color="white")))
    fig.update_xaxes(showgrid=True, gridcolor="#D8CFBD")
    fig.update_yaxes(showgrid=True, gridcolor="#D8CFBD")
    return _style(fig, 460)


def fig_choropleth(modo="% internet fija", destacar=None, height=560):
    view = VIEWS.get(destacar) if destacar else None
    center = view["center"] if view else {"lat": -43.5, "lon": -72.0}
    zoom = view["zoom"] if view else 2.85
    if modo == "% internet fija":
        fig = px.choropleth_map(
            df, geojson=GEO, locations="CUT", color="tiene_internet_fija",
            color_continuous_scale="Blues", range_color=(0, 100),
            map_style="carto-positron", zoom=zoom, center=center,
            opacity=0.8, hover_name="comuna_nombre",
            hover_data={"region_nombre": True, "tiene_internet_fija": ":.1f", "CUT": False},
            labels={"tiene_internet_fija": "% internet fija"})
    else:
        fig = px.choropleth_map(
            df, geojson=GEO, locations="CUT", color="lisa_label",
            color_discrete_map=CLUSTER_COLORS,
            category_orders={"lisa_label": list(CLUSTER_COLORS)},
            map_style="carto-positron", zoom=zoom, center=center,
            opacity=0.8, hover_name="comuna_nombre",
            hover_data={"region_nombre": True, "tiene_internet_fija": ":.1f", "CUT": False},
            labels={"lisa_label": "Cluster"})
    if destacar:   # contorno morado de las comunas de la región resaltada
        sub = df[df["region_nombre"] == destacar]
        fig.add_trace(go.Choroplethmap(
            geojson=GEO, locations=sub["CUT"], z=[1] * len(sub),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
            marker=dict(line=dict(color=C_DESTACA, width=2)), hoverinfo="skip"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.02))
    return fig


def fig_radar(destacar=None, uirev=0):
    cats = ["Internet móvil", "Computador", "Internet fija", "Tablet"]
    fig = go.Figure()
    for region, color in ACENTOS.items():
        vals = [MATRIZ.loc[region, c] for c in cats]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], name=region, fill="toself",
            line=dict(color=color, width=4.5 if region == destacar else 2.5),
            fillcolor=color, opacity=0.55,
            hovertemplate="%{theta}: %{r:.0f}%<extra>" + region + "</extra>"))
    if destacar and destacar not in ACENTOS and destacar in MATRIZ.index:
        vals = [MATRIZ.loc[destacar, c] for c in cats]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], name=destacar, fill="toself",
            line=dict(color=C_DESTACA, width=4),          # línea sólida gruesa = resalta sobre
            fillcolor=_rgba(C_DESTACA, 0.10),             # las capas; relleno casi transparente
            hovertemplate="%{theta}: %{r:.0f}%<extra>" + destacar + "</extra>"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 95], tickvals=[20, 40, 60, 80], ticksuffix="%",
                        gridcolor="#CFC6B2"),
        angularaxis=dict(rotation=90, direction="clockwise", gridcolor="#CFC6B2")))
    fig = _style(fig, 420, legend_bottom=False)
    # Se encoge el área polar (deja una banda abajo) para que la leyenda no la tape.
    # uirevision: cambia solo al pulsar "Restablecer vista" -> Plotly resetea el zoom.
    fig.update_layout(margin=dict(l=70, r=70, t=20, b=8),
                      uirevision=f"radar-{uirev}",
                      polar=dict(domain=dict(y=[0.16, 1.0])),
                      legend=dict(orientation="h", yanchor="top", y=0.11,
                                  xanchor="center", x=0.5))
    return fig


def _reset_radar():
    st.session_state.radar_rev = st.session_state.get("radar_rev", 0) + 1


def fig_slope(destacar=None):
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
                                 line=dict(color=color, width=4.5 if region == destacar else 2.6),
                                 text=[f"{v:.0f}%" for v in ys],
                                 textposition=TXTPOS.get(region, "top center"),
                                 textfont=dict(color=color, size=11),
                                 hovertemplate="%{x}: %{y:.0f}%<extra>" + region + "</extra>"))
    if destacar and destacar not in ACENTOS and destacar in SLO_REGIONS:
        row = SLO[SLO["region_nombre"] == destacar].iloc[0]
        ys = [row["pct_2017"], row["pct_2022"], row["pct_2024"]]
        fig.add_trace(go.Scatter(x=YEARS, y=ys, name=destacar, mode="lines+markers",
                                 line=dict(color=C_DESTACA, width=2.6, dash="dot"),
                                 hovertemplate="%{x}: %{y:.0f}%<extra>" + destacar + "</extra>"))
    fig.update_yaxes(title="Hogares con internet fija (%)", range=[20, 82],
                     showgrid=True, gridcolor="#D8CFBD")
    return _style(fig, 460)


def fig_histograma():
    m = df["tiene_internet_fija"].mean()
    fig = px.histogram(df, x="tiene_internet_fija", nbins=26,
                       color_discrete_sequence=[NAVY],
                       labels={"tiene_internet_fija": "% hogares con internet fija (comuna)"})
    fig.update_traces(marker_line_color="white", marker_line_width=0.5,
                      hovertemplate="%{x}<br>%{y} comunas<extra></extra>")
    fig.add_vline(x=m, line=dict(color=C_RURAL, dash="dash", width=2))
    fig.add_annotation(x=m, xref="x", yref="paper", y=1.07,
                       text=f"Promedio comunal: {coma(m, 0)}%", showarrow=False,
                       font=dict(color=C_RURAL, size=12), xanchor="left", xshift=4)
    fig.update_yaxes(title="N° de comunas", showgrid=True, gridcolor="#D8CFBD")
    fig.update_xaxes(title="% hogares con internet fija (comuna)", showgrid=False)
    fig = _style(fig, 430, legend_bottom=False)
    fig.update_layout(margin=dict(l=10, r=10, t=36, b=10))   # aire arriba para la anotación
    return fig


# ---------------------------------------------------------------- Sidebar (interacción cruzada)
with st.sidebar:
    st.header("🔎 Explorar")
    sel = st.selectbox("Resaltar una región", ["(ninguna)"] + ORDEN_REGIONES, index=0)
    destacar = None if sel == "(ninguna)" else sel
    st.caption("Resalta la región en el **heatmap, dumbbell, radar y evolución** (pestaña 1), "
               "y **centra el mapa + marca sus comunas en el diagrama de Moran** (pestaña 2). "
               "El gráfico 3 (por edad/educación) es nacional y no depende de la región.")
    if destacar:
        fija = MATRIZ.loc[destacar, "Internet fija"]
        rank = int((MATRIZ["Internet fija"] > fija).sum()) + 1
        brecha = RA.set_index("region_nombre").loc[destacar, "brecha"]
        st.markdown(f"#### 🟣 {destacar}")
        mc1, mc2 = st.columns(2)
        mc1.metric("Internet fija", f"{fija:.0f}%", help=f"Puesto {rank} de 16 regiones")
        mc2.metric("Brecha urb-rural", f"+{brecha:.0f} pp")
    st.divider()
    with open(os.path.join(DATA, "comunas.csv"), "rb") as f:
        st.download_button("⬇️ Descargar datos comunales (CSV)", f, "comunas_brecha.csv",
                           "text/csv")

# ---------------------------------------------------------------- Header
st.title("Brecha digital y territorio en Chile")
st.caption("Visualización de Datos · Universidad de Concepción — Entregable 3")
st.markdown(
    "La conectividad digital en Chile no se distribuye al azar: se **agrupa en el "
    "territorio**. Este panel reproduce de forma **interactiva** los seis gráficos del "
    "reporte y permite explorar la dimensión espacial comuna por comuna.")

nat = SLO[SLO["region_nombre"] == "Promedio nacional"].iloc[0]
crecimiento = nat["pct_2024"] - nat["pct_2017"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Moran's I (internet fija)", coma(S["moran_I"], 2),
          help=f"p < 0,001 · {S['permutations']} permutaciones · autocorrelación espacial global")
k2.metric("Acceso nacional 2024", f"{nat['pct_2024']:.0f}%",
          delta=f"+{crecimiento:.0f} pp vs 2017", help="Internet fija, CASEN (ponderado)")
k3.metric("Brecha urbano-rural", f"{RA['brecha'].mean():.0f} pp",
          help="Diferencia promedio urbano − rural en internet fija (16 regiones)")
k4.metric("Promedio comunal", f"{S['metric_mean']:.0f}%",
          help="Media simple de las 343 comunas (base del mapa)")
k5.metric("Comunas analizadas", f"{S['n_comunas']}")
st.caption(
    f"ℹ️ El **promedio comunal** ({S['metric_mean']:.0f}%) pesa cada comuna por igual (es la "
    f"base del mapa); el **acceso nacional** ({nat['pct_2024']:.0f}%) pondera por población. "
    "Difieren porque la mayoría de los hogares está en comunas urbanas, más conectadas.")

tab_plots, tab_spatial, tab_rep = st.tabs(
    ["📊 Los 6 gráficos (interactivos)", "🗺️ Análisis espacial interactivo", "📄 Reporte A4"])

# ---------------------------------------------------------------- Los 6 gráficos
with tab_plots:
    if destacar:
        st.caption(f"🟣 Resaltando **{destacar}** en los gráficos.")
    else:
        st.caption("💡 Pasa el mouse para ver valores · resalta una región desde la barra "
                   "lateral · usa la barra de cada gráfico (arriba a la derecha) o doble-clic "
                   "para zoom; el radar tiene su botón «Restablecer vista». Izquierda 1–3: qué "
                   "falta y a quién · derecha 4–6: dónde y cuándo.")

    r1l, r1r = st.columns(2)
    with r1l.container(border=True, height=580):
        st.markdown("**1 · Servicios de conectividad por región**")
        st.caption("% de hogares con cada servicio por región (Censo 2024).")
        st.plotly_chart(fig_heatmap(destacar), width="stretch", key="hm")
    with r1r.container(border=True, height=580):
        st.markdown("**4 · Internet fija por comuna**")
        st.caption("% de hogares con internet fija por comuna (343). Clusters en la pestaña espacial.")
        st.plotly_chart(fig_choropleth(destacar=destacar, height=460), width="stretch",
                        key="map_main")

    r2l, r2r = st.columns(2)
    with r2l.container(border=True, height=580):
        st.markdown("**2 · Acceso urbano vs rural**")
        st.caption("Internet fija urbano (navy) vs rural (terracota); brecha en pp a la derecha.")
        st.plotly_chart(fig_dumbbell(destacar), width="stretch", key="db")
    with r2r.container(border=True, height=580):
        st.markdown("**5 · Perfiles de conectividad regional**")
        st.caption("Cuatro servicios para tres regiones contrastantes (% de hogares).")
        # Plotly no da botón de reset a los gráficos polares en su modebar, así que va
        # como botón aparte (devuelve el radar a su vista original).
        st.plotly_chart(fig_radar(destacar, st.session_state.get("radar_rev", 0)),
                        width="stretch", key="radar")
        st.button("↩️ Restablecer vista", key="reset_radar", on_click=_reset_radar,
                  width="stretch",
                  help="Devuelve el radar a su vista original (el modebar de Plotly no trae "
                       "reset para gráficos polares). También funciona doble-clic.")

    r3l, r3r = st.columns(2)
    with r3l.container(border=True, height=580):
        st.markdown("**3 · Acceso por edad y nivel educativo**")
        st.caption("% con internet fija por edad y nivel educativo; tamaño = N° de hogares.")
        st.plotly_chart(fig_bubble(), width="stretch", key="bub")
    with r3r.container(border=True, height=580):
        st.markdown("**6 · Evolución del acceso (CASEN 2017–2024)**")
        st.caption("% de hogares con internet fija en 2017, 2022 y 2024.")
        st.plotly_chart(fig_slope(destacar), width="stretch", key="slope")

    with st.container(border=True):
        st.markdown("**📝 Síntesis**")
        st.markdown(SINTESIS)

# ---------------------------------------------------------------- Espacial
with tab_spatial:
    st.subheader("Autocorrelación espacial — Moran's I")
    st.markdown(
        f"El índice de Moran global es **{coma(S['moran_I'], 2)}** (p < 0,001, "
        f"{S['permutations']} permutaciones): la conectividad **se agrupa "
        "significativamente** en el territorio. Abajo, dónde están esos clusters.")

    with st.expander("¿Qué es el Moran's I? (para audiencia no especializada)"):
        st.markdown(
            "- **Idea:** mide si las comunas parecidas en conectividad tienden a estar "
            "**juntas** en el espacio.\n"
            "- **Escala:** va de −1 a +1. Cercano a **0** = patrón al azar; **positivo** = "
            "comunas similares se agrupan (vecinas se parecen); **negativo** = se alternan.\n"
            f"- **Aquí:** {coma(S['moran_I'], 2)} (positivo y alto) → el acceso **se concentra "
            "territorialmente**, no está repartido al azar.\n"
            "- **LISA** identifica *dónde*: núcleos conectados (Alto-Alto) y bolsones de "
            "rezago (Bajo-Bajo).")

    left, right = st.columns([3, 2])
    with left.container(border=True, height=690):
        modo = st.radio("Mapa:", ["% internet fija", "Clusters LISA"], horizontal=True)
        if destacar:
            st.caption(f"🟣 Mapa centrado en **{destacar}**.")
        st.plotly_chart(fig_choropleth(modo, destacar), width="stretch", key="map_spatial")

    with right.container(border=True, height=690):
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
                                name=f"Pendiente = I ({coma(S['moran_I'], 2)})"))
        sc.add_hline(y=0, line=dict(color="#999", width=1))
        sc.add_vline(x=0, line=dict(color="#999", width=1))
        if destacar:
            sub = df[df["region_nombre"] == destacar]
            sc.add_trace(go.Scatter(
                x=sub["z"], y=sub["lag_z"], mode="markers", name=f"{destacar} (resaltada)",
                marker=dict(size=12, color="rgba(0,0,0,0)",
                            line=dict(color=C_DESTACA, width=2.5)),
                text=sub["comuna_nombre"], hoverinfo="text"))
        sc.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                         paper_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=-0.22, font=dict(size=10)))
        st.plotly_chart(sc, width="stretch", key="moran")

    st.markdown("##### Clusters significativos (LISA, p < 0,05)")
    cc = st.columns(5)
    order = ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto"]
    labels = {"Alto-Alto": "🔴 Alto-Alto (núcleos)", "Bajo-Bajo": "🔵 Bajo-Bajo (rezago)",
              "Alto-Bajo": "🟠 Alto-Bajo (outlier)", "Bajo-Alto": "🟡 Bajo-Alto (outlier)"}
    for col, k in zip(cc, order):
        col.metric(labels[k], S["clusters"][k])
    cc[4].metric("No significativo", S["no_sig"])

    st.divider()
    h_col, t_col = st.columns([3, 2])
    with h_col.container(border=True, height=560):
        st.markdown("**Distribución del acceso entre comunas**")
        st.caption("Cuántas comunas hay en cada nivel de internet fija. La cola izquierda "
                   "(comunas muy rezagadas) es la que tira el promedio comunal hacia abajo.")
        st.plotly_chart(fig_histograma(), width="stretch", key="hist")
    with t_col.container(border=True, height=560):
        st.markdown("**Comunas extremas**")
        top = (df.nlargest(5, "tiene_internet_fija")[["comuna_nombre", "tiene_internet_fija"]]
               .rename(columns={"comuna_nombre": "Comuna", "tiene_internet_fija": "%"}))
        bot = (df.nsmallest(5, "tiene_internet_fija")[["comuna_nombre", "tiene_internet_fija"]]
               .rename(columns={"comuna_nombre": "Comuna", "tiene_internet_fija": "%"}))
        st.caption("🔝 5 más conectadas")
        st.dataframe(top.round(1), width="stretch", hide_index=True, height=215)
        st.caption("🔻 5 menos conectadas")
        st.dataframe(bot.round(1), width="stretch", hide_index=True, height=215)

    with st.expander("Ver tabla completa de comunas"):
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
st.caption("Fuentes: Censo 2024 (INE) · CASEN 2017/2022/2024 (MDSF). "
           "Autocorrelación espacial: Moran's I (contigüidad Queen, 999 permutaciones).")
