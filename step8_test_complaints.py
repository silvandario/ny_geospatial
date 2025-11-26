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

def load_complaint_data(filepath):
    """
    Load NYPD complaint CSV and convert to GeoDataFrame.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None

    print(f"Loading NYPD complaint data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        
        # Check column names
        print(f"Columns in file: {list(df.columns)[:10]}...")  # Show first 10 columns
        
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
        print(f"Loaded {len(df)} complaints with valid coordinates.")
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"
        )
        
        return gdf
        
    except Exception as e:
        print(f"Error loading complaint data: {e}")
        import traceback
        traceback.print_exc()
        return None

def spatial_join_and_aggregate(hex_gdf, complaint_gdf):
    """
    Perform spatial join and count complaints per hexagon.
    """
    print("Performing spatial join...")
    
    # Ensure both GDFs are in the same CRS
    hex_gdf = hex_gdf.to_crs(epsg=4326)
    complaint_gdf = complaint_gdf.to_crs(epsg=4326)
    
    # Spatial join: join complaints to hexagons
    joined = gpd.sjoin(complaint_gdf, hex_gdf, how='inner', predicate='within')
    
    print(f"Matched {len(joined)} complaints to hexagons.")
    
    # Count complaints per hexagon
    complaint_counts = joined.groupby('h3_id').size().reset_index(name='complaint_count')
    
    print(f"Found complaints in {len(complaint_counts)} hexagons.")
    
    # Merge counts back to hex grid
    hex_gdf = hex_gdf.merge(complaint_counts, on='h3_id', how='left')
    
    # Fill NaN with 0
    hex_gdf['complaint_count'] = hex_gdf['complaint_count'].fillna(0)
    
    return hex_gdf

def visualize_results(hex_gdf):
    """
    Create choropleth map and print top hexagons.
    """
    # Top 3 hexagons by complaint count
    top3 = hex_gdf.nlargest(3, 'complaint_count')[['h3_id', 'complaint_count']]
    print("\n=== TOP 3 HEXAGONS BY COMPLAINT COUNT ===")
    for idx, row in top3.iterrows():
        print(f"  H3 ID: {row['h3_id']} - Complaints: {int(row['complaint_count'])}")
    
    # Statistics
    print(f"\n=== NYPD COMPLAINT STATISTICS (2020 onwards) ===")
    print(f"Total complaints aggregated: {int(hex_gdf['complaint_count'].sum())}")
    print(f"Hexagons with complaints: {int((hex_gdf['complaint_count'] > 0).sum())}")
    print(f"Max complaints in one hexagon: {int(hex_gdf['complaint_count'].max())}")
    if (hex_gdf['complaint_count'] > 0).sum() > 0:
        print(f"Average complaints per hexagon (with complaints): {hex_gdf[hex_gdf['complaint_count'] > 0]['complaint_count'].mean():.2f}")
    
    # Plot
    print("\nCreating choropleth map...")
    fig, ax = plt.subplots(figsize=(14, 14))
    
    hex_gdf.plot(
        column='complaint_count', 
        ax=ax, 
        legend=True,
        cmap='Reds',  # Red gradient for complaints
        edgecolor='black',
        linewidth=0.1,
        legend_kwds={'label': 'NYPD Complaints per Hexagon (2020+)'}
    )
    
    plt.title("NYC NYPD Complaints by Hexagon (Resolution 8, 2020 onwards)")
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
    
    # Step 2: Load complaint data
    complaint_filepath = "data/raw/NYPD_Complaint_Data_Historic_2020onwards.csv"
    complaint_gdf = load_complaint_data(complaint_filepath)
    if complaint_gdf is None:
        return
    
    # Step 3: Spatial join and aggregate
    hex_gdf = spatial_join_and_aggregate(hex_gdf, complaint_gdf)
    
    # Step 4: Visualize
    visualize_results(hex_gdf)
    
    print("\nNYPD Complaint test completed successfully!")

if __name__ == "__main__":
    main()
