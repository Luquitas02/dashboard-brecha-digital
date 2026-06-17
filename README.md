# Dashboard D3 — Brecha digital y territorio en Chile

Dashboard interactivo (Streamlit) que acompaña la infografía A4 del Deliverable 3.
Muestra los **mismos 6 gráficos** del reporte + un **análisis espacial interactivo**
(Moran's I, clusters LISA, mapa coroplético con hover, diagrama de dispersión de Moran).

## Estructura

```
dashboard/
├── app.py                ← la aplicación Streamlit
├── build_data.py         ← precomputa los datos livianos (correr UNA vez)
├── requirements.txt      ← dependencias para el deploy
├── .streamlit/config.toml← tema (paleta del proyecto)
└── data/                 ← generado por build_data.py
    ├── comunas.geojson   (geometría simplificada + props)
    ├── comunas.csv       (métricas + LISA + z/lag)
    └── moran_stats.json  (I, p, z, conteos de clusters)
```

`build_data.py` usa geopandas/libpysal/esda y el GDB del Censo (pesado); el **app NO**
los necesita (solo pandas + plotly + streamlit) → deploy liviano y rápido.

## Correr local

```bash
# 1) generar los datos (una vez; necesita geopandas, libpysal, esda)
python build_data.py
# 2) levantar el dashboard
streamlit run app.py
```

## Desplegar (Streamlit Community Cloud → link público)

1. Subir la carpeta `dashboard/` a un repo de GitHub (incluyendo `data/`, que es liviano).
2. En https://share.streamlit.io → New app → elegir el repo y `app.py`.
3. Streamlit instala `requirements.txt` y entrega el link público.

> El reporte (`../figuras/d3_infografia.png` y `_pruebas/*.png`) debe estar en el repo
> para que las pestañas "Reporte" y "Los 6 gráficos" muestren las figuras.

## Fuentes
Censo 2024 (INE) · CASEN 2017/2022/2024 (MDSF). Autocorrelación espacial: Moran's I
con contigüidad Queen (999 permutaciones), métrica = % de hogares con internet fija.
