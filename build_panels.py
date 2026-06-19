# -*- coding: utf-8 -*-
"""Precomputa los datos LIVIANOS de los 6 paneles interactivos del dashboard
(se corre UNA vez; NO en el deploy). Replica EXACTAMENTE la lógica de los
scripts del reporte (notebooks/_gN_*.py) para que el dashboard quede
"completamente alineado con los gráficos del reporte".

Escribe en dashboard/data/:
  - panel_regional.csv    región × servicio (%)         -> heatmap (1) + radar (5)
  - panel_region_area.csv internet fija urbano/rural     -> dumbbell (2)
  - panel_bubble.csv      edad × nivel educativo (% , N)  -> bubble (3)
  - panel_slope.csv       % internet fija 2017/2022/2024  -> slope (6)

El mapa (4) ya se sirve desde data/comunas.* (build_data.py).
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
PROCESADO = os.path.join(ROOT, "..", "..", "datos", "procesado")

# --- constantes copiadas de notebooks/style.py (sin importar matplotlib) ---
NOMBRES_REGIONES = {
    15: "Arica y Parinacota", 1: "Tarapacá", 2: "Antofagasta", 3: "Atacama",
    4: "Coquimbo", 5: "Valparaíso", 13: "Metropolitana", 6: "O'Higgins",
    7: "Maule", 16: "Ñuble", 8: "Biobío", 9: "La Araucanía", 14: "Los Ríos",
    10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
}
ORDEN_REGIONES = [
    "Arica y Parinacota", "Tarapacá", "Antofagasta", "Atacama", "Coquimbo",
    "Valparaíso", "Metropolitana", "O'Higgins", "Maule", "Ñuble",
    "Biobío", "La Araucanía", "Los Ríos", "Los Lagos", "Aysén", "Magallanes",
]
NOMBRES_SERVICIOS = {
    "tel_movil": "Telefonía móvil", "computador": "Computador", "tablet": "Tablet",
    "internet_fija": "Internet fija", "internet_movil": "Internet móvil",
}


def p(fn):
    return os.path.join(PROCESADO, fn)


# ============================================================= 1) REGIONAL
# (heatmap + radar) — sin internet satelital, con nombres legibles.
def regional():
    df = pd.read_parquet(p("brecha_digital_regional.parquet"))
    df = df[df["servicio"] != "internet_satelital"].copy()
    df["region_nombre"] = df["region"].replace(NOMBRES_REGIONES)
    df["servicio_nombre"] = df["servicio"].replace(NOMBRES_SERVICIOS)
    out = df[["region_nombre", "servicio_nombre", "porcentaje"]]
    out.to_csv(os.path.join(DATA, "panel_regional.csv"), index=False, encoding="utf-8")
    print("panel_regional.csv:", out.shape)


# ============================================================= 2) REGION_AREA
# (dumbbell) — internet fija urbano vs rural por región + brecha en pp.
def region_area():
    df = pd.read_parquet(p("brecha_digital_region_area.parquet"))
    fija = df[df["servicio"] == "internet_fija"].copy()
    fija["region_nombre"] = fija["region"].replace(NOMBRES_REGIONES)
    fija["area_label"] = fija["area"].map({1: "Urbano", 2: "Rural"})
    wide = fija.pivot(index="region_nombre", columns="area_label",
                      values="porcentaje").reset_index()
    wide["brecha"] = wide["Urbano"] - wide["Rural"]
    wide = wide.set_index("region_nombre").loc[ORDEN_REGIONES].reset_index()
    wide.to_csv(os.path.join(DATA, "panel_region_area.csv"), index=False, encoding="utf-8")
    print("panel_region_area.csv:", wide.shape, "| promedio brecha=%.1f" % wide["brecha"].mean())


# ============================================================= 3) BUBBLE
# (acceso por edad y nivel educativo) — réplica de _g3_test_narrow.py.
def bubble():
    df = pd.read_parquet(p("brecha_digital_jefe_hogar.parquet"))
    df = df[df["escolaridad"] != -99].copy()

    def nivel(e):
        if e <= 8:
            return "Básica o menos"
        elif e <= 12:
            return "Media"
        elif e <= 16:
            return "Técnica/Sup. inc."
        return "Superior completa"

    df["nivel"] = df["escolaridad"].apply(nivel)
    agg = (df.groupby(["edad_quinquenal", "nivel"])
           .agg(pct_internet=("tiene_internet_fija", "mean"),
                n_hogares=("tiene_internet_fija", "count")).reset_index())
    agg["pct_internet"] = agg["pct_internet"] * 100
    agg = agg[(agg["edad_quinquenal"] >= 15) & (agg["edad_quinquenal"] <= 85)]
    agg.to_csv(os.path.join(DATA, "panel_bubble.csv"), index=False, encoding="utf-8")
    print("panel_bubble.csv:", agg.shape)


# ============================================================= 4) SLOPE
# (evolución 2017-2024) — réplica de _g5_test_compact.py (CASEN, ponderado).
def slope():
    df17 = pd.read_parquet(p("casen2017_internet.parquet"))
    df22 = pd.read_parquet(p("casen2022_internet.parquet"))
    df24 = pd.read_parquet(p("casen2024.parquet"),
                           columns=["r17a", "region", "area", "expr", "pco1", "folio"])

    def por_region(df, integrar=False):
        jh = df[df["pco1"] == 1].copy()
        jh["tiene"] = jh["r17a"].replace({1: 1, 2: 0}).where(jh["r17a"].isin([1, 2]))
        jh = jh.dropna(subset=["tiene"]).copy()
        jh["rc"] = jh["region"].astype(int)
        if integrar:
            jh["rc"] = jh["rc"].replace({16: 8})
        jh["peso"] = jh["expr"].astype(float)
        jh["tp"] = jh["tiene"] * jh["peso"]
        agg = (jh.groupby("rc").agg(st=("tp", "sum"), sp=("peso", "sum")).reset_index()
               .rename(columns={"rc": "region"}))
        agg["porcentaje"] = agg["st"] / agg["sp"] * 100
        return agg[["region", "porcentaje"]]

    def nacional(df):
        jh = df[df["pco1"] == 1].copy()
        jh["tiene"] = jh["r17a"].replace({1: 1, 2: 0}).where(jh["r17a"].isin([1, 2]))
        jh = jh.dropna(subset=["tiene"]).copy()
        w = jh["expr"].astype(float)
        return (jh["tiene"] * w).sum() / w.sum() * 100

    g17, g22, g24 = por_region(df17, False), por_region(df22, True), por_region(df24, True)
    N15 = {k: v for k, v in NOMBRES_REGIONES.items() if k != 16}
    O15 = [r for r in ORDEN_REGIONES if r != "Ñuble"]
    comp = (g17.rename(columns={"porcentaje": "pct_2017"})
            .merge(g22.rename(columns={"porcentaje": "pct_2022"}), on="region")
            .merge(g24.rename(columns={"porcentaje": "pct_2024"}), on="region"))
    comp["region_nombre"] = comp["region"].replace(N15)
    comp = comp.set_index("region_nombre").loc[O15].reset_index()
    # fila de promedio nacional (ponderado)
    nac = pd.DataFrame([{"region_nombre": "Promedio nacional", "region": 0,
                         "pct_2017": nacional(df17), "pct_2022": nacional(df22),
                         "pct_2024": nacional(df24)}])
    out = pd.concat([comp, nac], ignore_index=True)[
        ["region_nombre", "pct_2017", "pct_2022", "pct_2024"]]
    out.to_csv(os.path.join(DATA, "panel_slope.csv"), index=False, encoding="utf-8")
    print("panel_slope.csv:", out.shape)


if __name__ == "__main__":
    regional()
    region_area()
    bubble()
    slope()
    print("OK | 4 CSV de paneles escritos en", DATA)
