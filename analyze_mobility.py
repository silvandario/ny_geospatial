import geopandas as gpd
import pandas as pd

# Load the mobility POI data
gdf = gpd.read_file('data/processed/mobility_pois.gpkg')

print("=== MOBILITY/TRANSPORT POI BREAKDOWN ===")
print(f"Total POIs: {len(gdf)}\n")

# Analyze by railway type
if 'railway' in gdf.columns:
    print("By RAILWAY type:")
    print(gdf['railway'].value_counts())
    print()

# Analyze by public_transport type
if 'public_transport' in gdf.columns:
    print("By PUBLIC_TRANSPORT type:")
    print(gdf['public_transport'].value_counts())
    print()

# Analyze by highway type
if 'highway' in gdf.columns:
    print("By HIGHWAY type:")
    print(gdf['highway'].value_counts())
    print()

# Analyze by amenity type
if 'amenity' in gdf.columns:
    print("By AMENITY type:")
    print(gdf['amenity'].value_counts())
    print()
