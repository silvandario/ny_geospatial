"""
NYC Vibe & Value Finder - Complete ETL Pipeline
================================================
This script loads all data sources (CSV + OSM), performs spatial joins,
calculates normalized scores, and creates the final dataset for the Streamlit dashboard.

Author: Senior Data Engineer
Date: 2025-11-27
"""

import os
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')

# ========================================
# STEP 1: GENERATE OR LOAD HEXAGON GRID
# ========================================

def generate_hex_grid(resolution=8):
    """
    Generate H3 hexagon grid for NYC.
    """
    import osmnx as ox
    
    print("\n" + "="*60)
    print("STEP 1: GENERATING NYC HEXAGON GRID")
    print("="*60)
    
    print("Loading NYC boundary from OSM...")
    nyc_gdf = ox.geocode_to_gdf("New York City, USA")
    nyc_gdf = nyc_gdf.to_crs(epsg=4326)
    
    print(f"Generating H3 hexagons (Resolution {resolution})...")
    hex_ids = set()
    
    for idx, geom in enumerate(nyc_gdf.geometry):
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = geom.geoms
        else:
            continue
            
        for poly in polys:
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
                print(f"Warning: Error processing polygon: {e}")
    
    print(f"✓ Generated {len(hex_ids)} unique hexagons")
    
    # Convert to GeoDataFrame
    hex_polys = []
    valid_hex_ids = []
    
    for hid in hex_ids:
        try:
            boundary = h3.cell_to_boundary(hid)
            boundary_lonlat = [(p[1], p[0]) for p in boundary]
            hex_polys.append(Polygon(boundary_lonlat))
            valid_hex_ids.append(hid)
        except Exception as e:
            print(f"Warning: Error converting hex {hid}: {e}")
    
    hex_gdf = gpd.GeoDataFrame(
        {'h3_id': valid_hex_ids, 'geometry': hex_polys},
        crs="EPSG:4326"
    )
    
    return hex_gdf

# ========================================
# STEP 2: LOAD CSV DATA
# ========================================

def load_csv_data(filepath, lat_col='Latitude', lon_col='Longitude'):
    """
    Load CSV file and convert to GeoDataFrame.
    """
    if not os.path.exists(filepath):
        print(f"  Warning: File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Try to find lat/lon columns (case insensitive)
    for col in df.columns:
        if col.lower() in ['latitude', 'lat']:
            lat_col = col
        elif col.lower() in ['longitude', 'lon', 'long']:
            lon_col = col
    
    # Remove rows with missing coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    
    return gdf

def load_all_csv_data():
    """
    Load all CSV datasets.
    """
    print("\n" + "="*60)
    print("STEP 2: LOADING CSV DATA")
    print("="*60)
    
    csv_data = {}
    
    # Airbnb
    print("Loading Airbnb data...")
    csv_data['airbnb'] = load_csv_data("data/raw/nyc_airbnb_listings_2024_clean_geo.csv")
    if csv_data['airbnb'] is not None:
        print(f"  ✓ Loaded {len(csv_data['airbnb']):,} Airbnb listings")
    
    # Rats
    print("Loading rat sighting data...")
    csv_data['rats'] = load_csv_data("data/raw/nyc_rat_sightings_clean_geo.csv")
    if csv_data['rats'] is not None:
        print(f"  ✓ Loaded {len(csv_data['rats']):,} rat sightings")
    
    # Noise
    print("Loading noise complaint data...")
    csv_data['noise'] = load_csv_data("data/raw/nyc_party_calls_clean_geo.csv")
    if csv_data['noise'] is not None:
        print(f"  ✓ Loaded {len(csv_data['noise']):,} noise complaints")
    
    # Shootings
    print("Loading NYPD shooting data...")
    csv_data['shootings'] = load_csv_data("data/raw/nypd_shootings_clean_geo.csv")
    if csv_data['shootings'] is not None:
        print(f"  ✓ Loaded {len(csv_data['shootings']):,} shooting incidents")
    
    # Collisions
    print("Loading vehicle collision data...")
    csv_data['collisions'] = load_csv_data("data/raw/vehicle_collisions_clean_geo.csv")
    if csv_data['collisions'] is not None:
        print(f"  ✓ Loaded {len(csv_data['collisions']):,} collisions")
    
    # Complaints
    print("Loading NYPD complaint data (2020+)...")
    csv_data['complaints'] = load_csv_data("data/raw/NYPD_Complaint_Data_Historic_2020onwards.csv")
    if csv_data['complaints'] is not None:
        print(f"  ✓ Loaded {len(csv_data['complaints']):,} complaints")
    
    return csv_data

# ========================================
# STEP 3: LOAD OSM POI DATA
# ========================================

def load_osm_poi_hexagons():
    """
    Load pre-aggregated OSM POI hexagon data.
    """
    print("\n" + "="*60)
    print("STEP 3: LOADING OSM POI DATA")
    print("="*60)
    
    osm_data = {}
    
    files = {
        'nightlife': 'data/processed/nightlife_hexagons.gpkg',
        'culture': 'data/processed/sightseeing_culture_hexagons.gpkg',
        'restaurants': 'data/processed/restaurant_cafe_hexagons.gpkg',
        'green': 'data/processed/green_relax_hexagons.gpkg',
        'mobility': 'data/processed/mobility_hexagons.gpkg',
        'shopping': 'data/processed/shopping_hexagons.gpkg'
    }
    
    for key, filepath in files.items():
        if os.path.exists(filepath):
            gdf = gpd.read_file(filepath)
            # Extract only h3_id and count column
            count_col = [c for c in gdf.columns if 'count' in c][0]
            osm_data[key] = gdf[['h3_id', count_col]].copy()
            osm_data[key] = osm_data[key].rename(columns={count_col: f'{key}_count'})
            print(f"  ✓ Loaded {key}: {len(osm_data[key])} hexagons with POIs")
        else:
            print(f"  Warning: {filepath} not found")
            osm_data[key] = None
    
    return osm_data

# ========================================
# STEP 4: SPATIAL JOINS (CSV -> HEXAGONS)
# ========================================

def spatial_join_count(points_gdf, hex_gdf, column_name):
    """
    Perform spatial join and count points per hexagon.
    """
    if points_gdf is None or len(points_gdf) == 0:
        hex_gdf[column_name] = 0
        return hex_gdf
    
    # Spatial join
    joined = gpd.sjoin(points_gdf, hex_gdf[['h3_id', 'geometry']], how='inner', predicate='within')
    
    # Count per hexagon
    counts = joined.groupby('h3_id').size().reset_index(name=column_name)
    
    # Merge back
    hex_gdf = hex_gdf.merge(counts, on='h3_id', how='left')
    hex_gdf[column_name] = hex_gdf[column_name].fillna(0)
    
    return hex_gdf

def spatial_join_mean(points_gdf, hex_gdf, value_column, output_column):
    """
    Perform spatial join and calculate mean value per hexagon.
    """
    if points_gdf is None or len(points_gdf) == 0:
        hex_gdf[output_column] = None
        return hex_gdf
    
    # Ensure value column is numeric
    if value_column in points_gdf.columns:
        points_gdf[value_column] = pd.to_numeric(points_gdf[value_column], errors='coerce')
        points_gdf = points_gdf.dropna(subset=[value_column])
    else:
        hex_gdf[output_column] = None
        return hex_gdf
    
    # Spatial join
    joined = gpd.sjoin(points_gdf, hex_gdf[['h3_id', 'geometry']], how='inner', predicate='within')
    
    # Calculate mean per hexagon
    means = joined.groupby('h3_id')[value_column].mean().reset_index(name=output_column)
    
    # Merge back
    hex_gdf = hex_gdf.merge(means, on='h3_id', how='left')
    
    return hex_gdf

def aggregate_csv_data(hex_gdf, csv_data):
    """
    Aggregate all CSV data into hexagons.
    """
    print("\n" + "="*60)
    print("STEP 4: AGGREGATING CSV DATA INTO HEXAGONS")
    print("="*60)
    
    # Airbnb (mean price)
    print("Aggregating Airbnb prices...")
    if csv_data['airbnb'] is not None:
        # Try to find price column
        price_col = None
        for col in csv_data['airbnb'].columns:
            if col.lower() == 'price':
                price_col = col
                break
        
        if price_col:
            hex_gdf = spatial_join_mean(csv_data['airbnb'], hex_gdf, price_col, 'price_avg')
            print(f"  ✓ Aggregated Airbnb prices")
        else:
            hex_gdf['price_avg'] = None
            print(f"  Warning: Price column not found in Airbnb data")
    else:
        hex_gdf['price_avg'] = None
    
    # Rats (count)
    print("Aggregating rat sightings...")
    hex_gdf = spatial_join_count(csv_data['rats'], hex_gdf, 'rat_count')
    print(f"  ✓ Aggregated rat sightings")
    
    # Noise (count)
    print("Aggregating noise complaints...")
    hex_gdf = spatial_join_count(csv_data['noise'], hex_gdf, 'noise_count')
    print(f"  ✓ Aggregated noise complaints")
    
    # Shootings (count)
    print("Aggregating shooting incidents...")
    hex_gdf = spatial_join_count(csv_data['shootings'], hex_gdf, 'shooting_count')
    print(f"  ✓ Aggregated shooting incidents")
    
    # Collisions (count)
    print("Aggregating vehicle collisions...")
    hex_gdf = spatial_join_count(csv_data['collisions'], hex_gdf, 'collision_count')
    print(f"  ✓ Aggregated vehicle collisions")
    
    # Complaints (count)
    print("Aggregating NYPD complaints...")
    hex_gdf = spatial_join_count(csv_data['complaints'], hex_gdf, 'complaint_count')
    print(f"  ✓ Aggregated NYPD complaints")
    
    return hex_gdf

# ========================================
# STEP 5: MERGE OSM POI DATA
# ========================================

def merge_osm_data(hex_gdf, osm_data):
    """
    Merge OSM POI counts into hexagon grid.
    """
    print("\n" + "="*60)
    print("STEP 5: MERGING OSM POI DATA")
    print("="*60)
    
    for key, data in osm_data.items():
        if data is not None:
            hex_gdf = hex_gdf.merge(data, on='h3_id', how='left')
            count_col = f'{key}_count'
            hex_gdf[count_col] = hex_gdf[count_col].fillna(0)
            print(f"  ✓ Merged {key} POI counts")
        else:
            hex_gdf[f'{key}_count'] = 0
            print(f"  Warning: No data for {key}")
    
    return hex_gdf

# ========================================
# STEP 6: CALCULATE NORMALIZED SCORES
# ========================================

def min_max_normalize(series):
    """
    Min-Max normalization (0 to 1).
    """
    if series.max() == 0:
        return series * 0  # All zeros
    return series / series.max()

def calculate_scores(hex_gdf):
    """
    Calculate normalized scores (0-1 scale) for all criteria.
    1.0 = good, 0.0 = bad
    """
    print("\n" + "="*60)
    print("STEP 6: CALCULATING NORMALIZED SCORES")
    print("="*60)
    
    # POSITIVE CRITERIA (More = Better)
    print("Calculating positive scores (more = better)...")
    hex_gdf['score_nightlife'] = min_max_normalize(hex_gdf['nightlife_count'])
    hex_gdf['score_culture'] = min_max_normalize(hex_gdf['culture_count'])
    hex_gdf['score_restaurants'] = min_max_normalize(hex_gdf['restaurants_count'])
    hex_gdf['score_green'] = min_max_normalize(hex_gdf['green_count'])
    hex_gdf['score_mobility'] = min_max_normalize(hex_gdf['mobility_count'])
    hex_gdf['score_shopping'] = min_max_normalize(hex_gdf['shopping_count'])
    print("  ✓ Calculated lifestyle scores")
    
    # NEGATIVE CRITERIA (Less = Better) - INVERTED
    print("Calculating negative scores (less = better, inverted)...")
    
    # Clean: Inverted rats
    hex_gdf['score_clean'] = 1 - min_max_normalize(hex_gdf['rat_count'])
    print("  ✓ Calculated cleanliness score")
    
    # Quiet: Inverted noise
    hex_gdf['score_quiet'] = 1 - min_max_normalize(hex_gdf['noise_count'])
    print("  ✓ Calculated quietness score")
    
    # Safety: Combined and inverted
    print("Calculating safety score (combination of 3 factors)...")
    norm_shootings = min_max_normalize(hex_gdf['shooting_count'])
    norm_collisions = min_max_normalize(hex_gdf['collision_count'])
    norm_complaints = min_max_normalize(hex_gdf['complaint_count'])
    
    # Average of normalized values
    avg_danger = (norm_shootings + norm_collisions + norm_complaints) / 3
    
    # Invert: 1.0 = safe, 0.0 = dangerous
    hex_gdf['score_safety'] = 1 - avg_danger
    print("  ✓ Calculated safety score")
    
    return hex_gdf

# ========================================
# STEP 7: WATER FILTER
# ========================================

def filter_water_hexagons(hex_gdf):
    """
    Remove hexagons without Airbnb data (water/uninhabitable areas).
    """
    print("\n" + "="*60)
    print("STEP 7: FILTERING WATER HEXAGONS")
    print("="*60)
    
    before_count = len(hex_gdf)
    hex_gdf = hex_gdf[hex_gdf['price_avg'].notna()].copy()
    after_count = len(hex_gdf)
    
    removed = before_count - after_count
    print(f"  ✓ Removed {removed} hexagons without Airbnb data (water/parks)")
    print(f"  ✓ Remaining hexagons: {after_count}")
    
    return hex_gdf

# ========================================
# STEP 8: SAVE FINAL DATA
# ========================================

def save_final_data(hex_gdf, output_path):
    """
    Save the final dataset to GeoPackage.
    """
    print("\n" + "="*60)
    print("STEP 8: SAVING FINAL DATASET")
    print("="*60)
    
    # Select columns to save
    columns_to_save = [
        'h3_id', 'geometry',
        # Scores (normalized 0-1)
        'score_nightlife', 'score_culture', 'score_restaurants',
        'score_green', 'score_mobility', 'score_shopping',
        'score_safety', 'score_clean', 'score_quiet',
        # Price (absolute value)
        'price_avg',
        # Raw counts (optional, for debugging)
        'nightlife_count', 'culture_count', 'restaurants_count',
        'green_count', 'mobility_count', 'shopping_count',
        'rat_count', 'noise_count', 'shooting_count',
        'collision_count', 'complaint_count'
    ]
    
    # Filter to existing columns
    columns_to_save = [c for c in columns_to_save if c in hex_gdf.columns]
    
    final_gdf = hex_gdf[columns_to_save].copy()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to GeoPackage
    final_gdf.to_file(output_path, driver="GPKG")
    
    print(f"  ✓ Saved {len(final_gdf)} hexagons to {output_path}")
    print(f"  ✓ Columns saved: {len(columns_to_save)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total hexagons: {len(final_gdf)}")
    print(f"Average Airbnb price: ${final_gdf['price_avg'].mean():.2f}")
    print(f"\nScore ranges (0=bad, 1=good):")
    for col in final_gdf.columns:
        if col.startswith('score_'):
            print(f"  {col:25s}: {final_gdf[col].min():.3f} - {final_gdf[col].max():.3f} (mean: {final_gdf[col].mean():.3f})")
    
    return final_gdf

# ========================================
# MAIN PIPELINE
# ========================================

def main():
    """
    Execute the complete ETL pipeline.
    """
    print("\n" + "="*60)
    print("NYC VIBE & VALUE FINDER - ETL PIPELINE")
    print("="*60)
    print("This pipeline will:")
    print("  1. Generate NYC hexagon grid (H3 Resolution 8)")
    print("  2. Load all CSV data (Airbnb, Rats, Noise, Crime, etc.)")
    print("  3. Load all OSM POI data (Nightlife, Culture, etc.)")
    print("  4. Perform spatial joins (aggregate data into hexagons)")
    print("  5. Calculate normalized scores (0-1 scale)")
    print("  6. Filter water/uninhabitable hexagons")
    print("  7. Save final dataset to nyc_final_data.gpkg")
    print("="*60)
    
    # Step 1: Generate hex grid
    hex_gdf = generate_hex_grid(resolution=8)
    
    # Step 2: Load CSV data
    csv_data = load_all_csv_data()
    
    # Step 3: Load OSM POI data
    osm_data = load_osm_poi_hexagons()
    
    # Step 4: Aggregate CSV data
    hex_gdf = aggregate_csv_data(hex_gdf, csv_data)
    
    # Step 5: Merge OSM data
    hex_gdf = merge_osm_data(hex_gdf, osm_data)
    
    # Step 6: Calculate scores
    hex_gdf = calculate_scores(hex_gdf)
    
    # Step 7: Filter water hexagons
    hex_gdf = filter_water_hexagons(hex_gdf)
    
    # Step 8: Save final data
    output_path = "data/processed/nyc_final_data.gpkg"
    final_gdf = save_final_data(hex_gdf, output_path)
    
    print("\n" + "="*60)
    print("✅ ETL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final dataset saved to: {output_path}")
    print(f"Ready for Streamlit dashboard!")
    print("="*60)
    
    return final_gdf

if __name__ == "__main__":
    main()
