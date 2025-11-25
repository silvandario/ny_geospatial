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

def load_airbnb_data(filepath):
    """
    Load Airbnb listings CSV and convert to GeoDataFrame.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None

    print(f"Loading Airbnb data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        
        # Check column names
        print(f"Columns in file: {list(df.columns)}")
        
        # Try to identify lat/lon columns (case insensitive)
        lat_col = None
        lon_col = None
        price_col = None
        
        for col in df.columns:
            if col.lower() in ['latitude', 'lat']:
                lat_col = col
            elif col.lower() in ['longitude', 'lon', 'long']:
                lon_col = col
            elif col.lower() in ['price']:
                price_col = col
        
        if lat_col is None or lon_col is None:
            print(f"ERROR: Could not find Latitude/Longitude columns.")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        if price_col is None:
            print(f"ERROR: Could not find Price column.")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print(f"Using columns: {lat_col}, {lon_col}, {price_col}")
        
        # Remove rows with missing coordinates or price
        df = df.dropna(subset=[lat_col, lon_col, price_col])
        
        # Convert price to numeric (remove $ and commas if present)
        if df[price_col].dtype == 'object':
            df[price_col] = df[price_col].str.replace('$', '').str.replace(',', '')
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            df = df.dropna(subset=[price_col])
        
        # Filter out unrealistic prices (optional, adjust as needed)
        df = df[(df[price_col] > 0) & (df[price_col] < 10000)]
        
        print(f"Loaded {len(df)} Airbnb listings with valid coordinates and prices.")
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"
        )
        
        # Store price column name for later use
        gdf['price_value'] = df[price_col]
        
        return gdf
        
    except Exception as e:
        print(f"Error loading Airbnb data: {e}")
        import traceback
        traceback.print_exc()
        return None

def spatial_join_and_aggregate(hex_gdf, airbnb_gdf):
    """
    Perform spatial join and calculate average price per hexagon.
    """
    print("Performing spatial join...")
    
    # Ensure both GDFs are in the same CRS
    hex_gdf = hex_gdf.to_crs(epsg=4326)
    airbnb_gdf = airbnb_gdf.to_crs(epsg=4326)
    
    # Spatial join: join listings to hexagons
    joined = gpd.sjoin(airbnb_gdf, hex_gdf, how='inner', predicate='within')
    
    print(f"Matched {len(joined)} Airbnb listings to hexagons.")
    
    # Calculate average price per hexagon
    price_stats = joined.groupby('h3_id').agg({
        'price_value': ['mean', 'median', 'count']
    }).reset_index()
    
    # Flatten column names
    price_stats.columns = ['h3_id', 'avg_price', 'median_price', 'listing_count']
    
    print(f"Found listings in {len(price_stats)} hexagons.")
    
    # Merge stats back to hex grid
    hex_gdf = hex_gdf.merge(price_stats, on='h3_id', how='left')
    
    # NaN values remain for hexagons without listings (water, parks, etc.)
    print(f"Hexagons with NaN prices (water/uninhabitable): {hex_gdf['avg_price'].isna().sum()}")
    
    return hex_gdf

def visualize_results(hex_gdf):
    """
    Create choropleth map and print top hexagons.
    """
    # Filter out NaN for statistics
    hex_with_prices = hex_gdf.dropna(subset=['avg_price'])
    
    # Top 3 most expensive hexagons
    top3_expensive = hex_with_prices.nlargest(3, 'avg_price')[['h3_id', 'avg_price', 'listing_count']]
    print("\n=== TOP 3 MOST EXPENSIVE HEXAGONS ===")
    for idx, row in top3_expensive.iterrows():
        print(f"  H3 ID: {row['h3_id']} - Avg Price: ${row['avg_price']:.2f} ({int(row['listing_count'])} listings)")
    
    # Top 3 cheapest hexagons
    top3_cheap = hex_with_prices.nsmallest(3, 'avg_price')[['h3_id', 'avg_price', 'listing_count']]
    print("\n=== TOP 3 CHEAPEST HEXAGONS ===")
    for idx, row in top3_cheap.iterrows():
        print(f"  H3 ID: {row['h3_id']} - Avg Price: ${row['avg_price']:.2f} ({int(row['listing_count'])} listings)")
    
    # Statistics
    print(f"\n=== PRICE STATISTICS ===")
    print(f"Total listings aggregated: {int(hex_with_prices['listing_count'].sum())}")
    print(f"Hexagons with listings: {len(hex_with_prices)}")
    print(f"Hexagons without listings (water/parks): {hex_gdf['avg_price'].isna().sum()}")
    print(f"Overall avg price: ${hex_with_prices['avg_price'].mean():.2f}")
    print(f"Overall median price: ${hex_with_prices['median_price'].median():.2f}")
    print(f"Price range: ${hex_with_prices['avg_price'].min():.2f} - ${hex_with_prices['avg_price'].max():.2f}")
    
    # Plot
    print("\nCreating choropleth map...")
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Plot only hexagons with prices (water hexagons will be invisible)
    hex_with_prices.plot(
        column='avg_price', 
        ax=ax, 
        legend=True,
        cmap='RdYlGn_r',  # Red = expensive, Green = cheap
        edgecolor='black',
        linewidth=0.1,
        legend_kwds={'label': 'Average Airbnb Price ($) per Hexagon'}
    )
    
    plt.title("NYC Airbnb Prices by Hexagon (Resolution 8)")
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
    
    # Step 2: Load Airbnb data
    airbnb_filepath = "data/raw/nyc_airbnb_listings_2024_clean_geo.csv"
    airbnb_gdf = load_airbnb_data(airbnb_filepath)
    if airbnb_gdf is None:
        return
    
    # Step 3: Spatial join and aggregate
    hex_gdf = spatial_join_and_aggregate(hex_gdf, airbnb_gdf)
    
    # Step 4: Visualize
    visualize_results(hex_gdf)
    
    print("\n✅ Test completed successfully!")
    print("Note: Hexagons with NaN prices will be filtered in the final ETL pipeline.")

if __name__ == "__main__":
    main()
