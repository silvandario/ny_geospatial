import geopandas as gpd
import pandas as pd

# Load the green/relax POI data
gdf = gpd.read_file('data/processed/green_relax_pois.gpkg')

print("=== GREEN/RELAX POI BREAKDOWN ===")
print(f"Total POIs: {len(gdf)}\n")

# Check which columns exist
print("Available columns:", gdf.columns.tolist()[:20])
print()

# Analyze by leisure type
if 'leisure' in gdf.columns:
    print("By LEISURE type:")
    print(gdf['leisure'].value_counts().head(15))
    print()

# Analyze by landuse type
if 'landuse' in gdf.columns:
    print("By LANDUSE type:")
    print(gdf['landuse'].value_counts().head(15))
    print()

# Analyze by natural type
if 'natural' in gdf.columns:
    print("By NATURAL type:")
    print(gdf['natural'].value_counts().head(15))
    print()
