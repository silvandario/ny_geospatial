import osmnx as ox
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import os

def generate_hex_grid(resolution=8):
    """
    Generate H3 hexagon grid for NYC.
    """
    print("Loading NYC boundary from OSM...")
    try:
        nyc_gdf = ox.geocode_to_gdf("New York City, USA")
    except Exception as e:
        print(f"Error loading NYC boundary: {e}")
        return None

    nyc_gdf = nyc_gdf.to_crs(epsg=4326)
    print(f"Loaded NYC boundary. Geometry type: {nyc_gdf.geometry.iloc[0].geom_type}")

    print(f"Generating H3 hexagons (Resolution {resolution})...")
    hex_ids = set()

    for idx, geom in enumerate(nyc_gdf.geometry):
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = geom.geoms
        else:
            continue

        for i, poly in enumerate(polys):
            coords = list(poly.exterior.coords)
            latlon_coords = [[p[1], p[0]] for p in coords]
            
            holes_latlon = []
            for interior in poly.interiors:
                holes_latlon.append([[p[1], p[0]] for p in interior.coords])
            
            try:
                if holes_latlon:
                    h3_poly = h3.LatLngPoly(latlon_coords, *holes_latlon)
                else:
                    h3_poly = h3.LatLngPoly(latlon_coords)
                    
                cells = h3.polygon_to_cells(h3_poly, res=resolution)
                hex_ids.update(cells)
            except Exception as e:
                print(f"Error processing polygon {i}: {e}")

    print(f"Generated {len(hex_ids)} unique hexagons.")

    # Convert H3 IDs to Shapely Polygons
    hex_polys = []
    valid_hex_ids = []

    for hid in hex_ids:
        try:
            boundary = h3.cell_to_boundary(hid)
            boundary_lonlat = [(p[1], p[0]) for p in boundary]
            hex_polys.append(Polygon(boundary_lonlat))
            valid_hex_ids.append(hid)
        except Exception as e:
            print(f"Error converting hex {hid}: {e}")

    hex_gdf = gpd.GeoDataFrame(
        {'h3_id': valid_hex_ids, 'geometry': hex_polys}, 
        crs="EPSG:4326"
    )

    return hex_gdf

def load_shooting_data(filepath):
    """
    Load NYPD shooting incidents CSV and convert to GeoDataFrame.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None

    print(f"Loading NYPD shooting data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        
        # Check column names
        print(f"Columns in file: {list(df.columns)}")
        
        # Try to identify lat/lon columns (case insensitive)
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            if col.lower() in ['latitude', 'lat']:
                lat_col = col
            elif col.lower() in ['longitude', 'lon', 'long']:
                lon_col = col
        
        if lat_col is None or lon_col is None:
            print(f"ERROR: Could not find Latitude/Longitude columns.")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print(f"Using columns: {lat_col}, {lon_col}")
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=[lat_col, lon_col])
        print(f"Loaded {len(df)} shooting incidents with valid coordinates.")
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"
        )
        
        return gdf
        
    except Exception as e:
        print(f"Error loading shooting data: {e}")
        import traceback
        traceback.print_exc()
        return None

def spatial_join_and_aggregate(hex_gdf, shooting_gdf):
    """
    Perform spatial join and count shooting incidents per hexagon.
    """
    print("Performing spatial join...")
    
    # Ensure both GDFs are in the same CRS
    hex_gdf = hex_gdf.to_crs(epsg=4326)
    shooting_gdf = shooting_gdf.to_crs(epsg=4326)
    
    # Spatial join: join shootings to hexagons
    joined = gpd.sjoin(shooting_gdf, hex_gdf, how='inner', predicate='within')
    
    print(f"Matched {len(joined)} shooting incidents to hexagons.")
    
    # Count incidents per hexagon
    shooting_counts = joined.groupby('h3_id').size().reset_index(name='shooting_count')
    
    print(f"Found shooting incidents in {len(shooting_counts)} hexagons.")
    
    # Merge counts back to hex grid
    hex_gdf = hex_gdf.merge(shooting_counts, on='h3_id', how='left')
    
    # Fill NaN with 0
    hex_gdf['shooting_count'] = hex_gdf['shooting_count'].fillna(0)
    
    return hex_gdf

def visualize_results(hex_gdf):
    """
    Create choropleth map and print top hexagons.
    """
    # Top 3 most dangerous hexagons
    top3 = hex_gdf.nlargest(3, 'shooting_count')[['h3_id', 'shooting_count']]
    print("\n=== TOP 3 MOST DANGEROUS HEXAGONS (Most Shootings) ===")
    for idx, row in top3.iterrows():
        print(f"  H3 ID: {row['h3_id']} - Shootings: {int(row['shooting_count'])}")
    
    # Statistics
    print(f"\n=== SHOOTING INCIDENT STATISTICS ===")
    print(f"Total shootings aggregated: {int(hex_gdf['shooting_count'].sum())}")
    print(f"Hexagons with shootings: {int((hex_gdf['shooting_count'] > 0).sum())}")
    print(f"Max shootings in one hexagon: {int(hex_gdf['shooting_count'].max())}")
    if (hex_gdf['shooting_count'] > 0).sum() > 0:
        print(f"Average shootings per hexagon (with incidents): {hex_gdf[hex_gdf['shooting_count'] > 0]['shooting_count'].mean():.2f}")
    
    # Plot
    print("\nCreating choropleth map...")
    fig, ax = plt.subplots(figsize=(14, 14))
    
    hex_gdf.plot(
        column='shooting_count', 
        ax=ax, 
        legend=True,
        cmap='OrRd',  # Orange to Red gradient for crime
        edgecolor='black',
        linewidth=0.1,
        legend_kwds={'label': 'Shooting Incidents per Hexagon'}
    )
    
    plt.title("NYC NYPD Shooting Incidents by Hexagon (Resolution 8)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    print("Displaying plot window...")
    plt.show()

def main():
    """
    Main execution flow.
    """
    # Step 1: Generate hex grid
    hex_gdf = generate_hex_grid(resolution=8)
    if hex_gdf is None:
        return
    
    # Step 2: Load shooting data
    shooting_filepath = "data/raw/nypd_shootings_clean_geo.csv"
    shooting_gdf = load_shooting_data(shooting_filepath)
    if shooting_gdf is None:
        return
    
    # Step 3: Spatial join and aggregate
    hex_gdf = spatial_join_and_aggregate(hex_gdf, shooting_gdf)
    
    # Step 4: Visualize
    visualize_results(hex_gdf)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
