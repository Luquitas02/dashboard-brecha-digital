# -*- coding: utf-8 -*-
"""Prepara los datos LIVIANOS del dashboard (se corre UNA vez, no en el deploy).
Calcula Moran's I global, LISA, lag espacial y z; simplifica la geometría y
guarda en dashboard/data/:
  - comunas.geojson  (geometría simplificada + propiedades, para el mapa Plotly)
  - comunas.csv      (tabla: métricas + LISA + z/lag, para scatter y tabla)
  - moran_stats.json (I, p, z, conteos de clusters)

Así el app de Streamlit NO necesita geopandas/libpysal ni el GDB de 300 MB.
"""
import warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, KNN, attach_islands, lag_spatial
from esda.moran import Moran, Moran_Local

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
PROCESADO = os.path.join(ROOT, "..", "..", "datos", "procesado")
GDB = os.path.join(ROOT, "..", "..", "datos", "crudo", "censo2024",
                   "Cartografia_censo2024_Pais.gdb")
METRIC = "tiene_internet_fija"
EXCLUIR_CUT = [5201, 5202, 5104, 12202]   # islas oceánicas + Antártica
LISA_LABEL = {1: "Alto-Alto", 2: "Bajo-Alto", 3: "Bajo-Bajo", 4: "Alto-Bajo"}

NOMBRES_REGIONES = {
    15: "Arica y Parinacota", 1: "Tarapacá", 2: "Antofagasta", 3: "Atacama",
    4: "Coquimbo", 5: "Valparaíso", 13: "Metropolitana", 6: "O'Higgins",
    7: "Maule", 16: "Ñuble", 8: "Biobío", 9: "La Araucanía", 14: "Los Ríos",
    10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
}


def main():
    gdf = gpd.read_file(GDB, layer="Comunal_CPV24")
    df = pd.read_parquet(os.path.join(PROCESADO, "brecha_digital_comunal.parquet"))
    g = gdf.merge(df, left_on="CUT", right_on="comuna", how="inner")
    g = g[~g["CUT"].isin(EXCLUIR_CUT)].copy()
    g = g.dropna(subset=[METRIC]).reset_index(drop=True)
    g["region_nombre"] = g["region"].map(NOMBRES_REGIONES)
    g["comuna_nombre"] = g["COMUNA"].str.title()

    y = g[METRIC].values

    # --- Pesos Queen (+ islas a su vecino más cercano) ---
    w = Queen.from_dataframe(g, use_index=False)
    if w.islands:
        w = attach_islands(w, KNN.from_dataframe(g, k=1))
    w.transform = "r"

    # --- Moran global ---
    mi = Moran(y, w, permutations=999)
    # --- Moran local / LISA ---
    lm = Moran_Local(y, w, permutations=999, seed=42)
    g["lisa_q"] = lm.q
    g["lisa_sig"] = (lm.p_sim < 0.05)
    g["lisa_label"] = np.where(g["lisa_sig"], g["lisa_q"].map(LISA_LABEL), "No significativo")
    # --- z estandarizado y lag espacial (para el Moran scatterplot) ---
    z = (y - y.mean()) / y.std()
    g["z"] = z
    g["lag_z"] = lag_spatial(w, z)

    # --- columnas finales ---
    cols = ["CUT", "comuna_nombre", "region", "region_nombre", METRIC,
            "total_hogares", "lisa_q", "lisa_label", "lisa_sig", "z", "lag_z"]
    tab = g[cols].copy()
    tab.to_csv(os.path.join(DATA, "comunas.csv"), index=False, encoding="utf-8")

    # --- geojson liviano (geometría simplificada) ---
    gj = g[cols + ["geometry"]].copy()
    gj["geometry"] = gj["geometry"].simplify(0.008, preserve_topology=True)
    gj = gj.to_crs(4674)
    gj.to_file(os.path.join(DATA, "comunas.geojson"), driver="GeoJSON")

    # --- estadísticos ---
    stats = {
        "moran_I": round(float(mi.I), 4),
        "moran_p": float(mi.p_sim),
        "moran_z": round(float(mi.z_sim), 2),
        "permutations": 999,
        "n_comunas": int(len(g)),
        "metric_mean": round(float(y.mean()), 1),
        "clusters": {LISA_LABEL[k]: int(((g["lisa_q"] == k) & g["lisa_sig"]).sum())
                     for k in [1, 3, 4, 2]},
        "no_sig": int((~g["lisa_sig"]).sum()),
    }
    with open(os.path.join(DATA, "moran_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    sz = os.path.getsize(os.path.join(DATA, "comunas.geojson")) / 1e6
    print(f"OK | {len(g)} comunas | Moran I={mi.I:.3f} p={mi.p_sim:.4f} z={mi.z_sim:.1f}")
    print(f"   geojson={sz:.1f} MB | csv + moran_stats.json escritos en {DATA}")
    print("   clusters:", stats["clusters"])


if __name__ == "__main__":
    main()
