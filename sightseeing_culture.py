import osmnx as ox
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def generate_hex_grid(resolution=8):
    """
    Generate H3 hexagon grid for NYC.
    """
    print("Loading NYC boundary from OSM...")
    try:
        nyc_gdf = ox.geocode_to_gdf("New York City, USA")
    except Exception as e:
        print(f"Error loading NYC boundary: {e}")
        return None, None

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

    return hex_gdf, nyc_gdf

def fetch_sightseeing_culture_pois(nyc_boundary):
    """
    Download sightseeing & culture POIs from OpenStreetMap for NYC.
    """
    print("\nFetching sightseeing & culture POIs from OpenStreetMap...")
    
    # Define all culture/sightseeing tags
    tags = {
        # Top sightseeing
        'tourism': ['attraction', 'viewpoint', 'museum', 'gallery', 'artwork', 
                    'theme_park', 'zoo', 'aquarium'],
        
        # Historic sites
        'historic': ['castle', 'ruins', 'memorial', 'monument', 'archaeological_site'],
        
        # Cultural infrastructure
        'amenity': ['place_of_worship', 'theatre', 'cinema', 'arts_centre', 
                    'exhibition_centre', 'community_centre', 'library', 'public_bookcase',
                    'fountain', 'marketplace'],
        
        # Buildings (religious, cultural, landmarks)
        'building': ['church', 'cathedral', 'mosque', 'temple', 'synagogue', 'monastery',
                     'townhall', 'castle', 'museum', 'tower', 'stadium']
    }
    
    try:
        # Get the polygon to query
        nyc_polygon = nyc_boundary.geometry.iloc[0]
        
        # Fetch POIs using osmnx
        print("  Downloading from OpenStreetMap (this may take a while)...")
        pois = ox.features_from_polygon(nyc_polygon, tags=tags)
        
        print(f"Downloaded {len(pois)} sightseeing/culture POIs from OSM.")
        
        # Ensure CRS is EPSG:4326
        pois = pois.to_crs(epsg=4326)
        
        return pois
        
    except Exception as e:
        print(f"Error fetching POIs: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_to_points(pois_gdf):
    """
    Convert all geometries to points (polygon centroids -> points).
    """
    print("\nConverting all geometries to points...")
    
    point_geometries = []
    
    for idx, row in pois_gdf.iterrows():
        geom = row.geometry
        
        if geom.geom_type == 'Point':
            point_geometries.append(geom)
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            # Convert polygon to centroid
            point_geometries.append(geom.centroid)
        else:
            # For other types, try centroid
            try:
                point_geometries.append(geom.centroid)
            except:
                print(f"  Warning: Could not convert geometry type {geom.geom_type}")
                point_geometries.append(None)
    
    # Create new GeoDataFrame with point geometries
    pois_points = pois_gdf.copy()
    pois_points['geometry'] = point_geometries
    
    # Remove rows with None geometries
    pois_points = pois_points[pois_points['geometry'].notna()]
    
    print(f"Converted to {len(pois_points)} point geometries.")
    
    return pois_points

def spatial_join_and_aggregate(hex_gdf, pois_gdf):
    """
    Perform spatial join and count sightseeing/culture POIs per hexagon.
    """
    print("\nPerforming spatial join...")
    
    # Ensure both GDFs are in the same CRS
    hex_gdf = hex_gdf.to_crs(epsg=4326)
    pois_gdf = pois_gdf.to_crs(epsg=4326)
    
    # Spatial join: join POIs to hexagons
    joined = gpd.sjoin(pois_gdf, hex_gdf, how='inner', predicate='within')
    
    print(f"Matched {len(joined)} POIs to hexagons.")
    
    # Count POIs per hexagon
    culture_counts = joined.groupby('h3_id').size().reset_index(name='culture_count')
    
    print(f"Found culture/sightseeing in {len(culture_counts)} hexagons.")
    
    # Merge counts back to hex grid
    hex_gdf = hex_gdf.merge(culture_counts, on='h3_id', how='left')
    
    # Fill NaN with 0
    hex_gdf['culture_count'] = hex_gdf['culture_count'].fillna(0)
    
    return hex_gdf

def visualize_results(hex_gdf):
    """
    Create choropleth map and print top hexagons.
    """
    # Top 5 culture hexagons
    top5 = hex_gdf.nlargest(5, 'culture_count')[['h3_id', 'culture_count']]
    print("\n=== TOP 5 CULTURE/SIGHTSEEING HEXAGONS ===")
    for idx, row in top5.iterrows():
        print(f"  H3 ID: {row['h3_id']} - Culture POIs: {int(row['culture_count'])}")
    
    # Statistics
    print(f"\n=== CULTURE/SIGHTSEEING STATISTICS ===")
    print(f"Total culture POIs: {int(hex_gdf['culture_count'].sum())}")
    print(f"Hexagons with culture: {int((hex_gdf['culture_count'] > 0).sum())}")
    print(f"Max POIs in one hexagon: {int(hex_gdf['culture_count'].max())}")
    if (hex_gdf['culture_count'] > 0).sum() > 0:
        print(f"Average POIs per hexagon (with culture): {hex_gdf[hex_gdf['culture_count'] > 0]['culture_count'].mean():.2f}")
    
    # Plot
    print("\nCreating choropleth map...")
    fig, ax = plt.subplots(figsize=(14, 14))
    
    hex_gdf.plot(
        column='culture_count', 
        ax=ax, 
        legend=True,
        cmap='YlGnBu',  # Yellow-Green-Blue for culture
        edgecolor='black',
        linewidth=0.1,
        legend_kwds={'label': 'Culture/Sightseeing POIs per Hexagon'}
    )
    
    plt.title("NYC Culture & Sightseeing Zones by Hexagon (OSM Data, Resolution 8)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    print("Displaying plot window...")
    plt.show()

def main():
    """
    Main execution flow.
    """
    import os
    
    # Step 1: Generate hex grid
    print("=== STEP 1: Generate Hexagon Grid ===")
    hex_gdf, nyc_boundary = generate_hex_grid(resolution=8)
    if hex_gdf is None:
        return
    
    # Step 2: Fetch OSM culture/sightseeing POIs
    print("\n=== STEP 2: Fetch Culture/Sightseeing POIs from OpenStreetMap ===")
    pois_gdf = fetch_sightseeing_culture_pois(nyc_boundary)
    if pois_gdf is None or len(pois_gdf) == 0:
        print("No POIs found. Exiting.")
        return
    
    # Step 3: Convert to points
    print("\n=== STEP 3: Convert Geometries to Points ===")
    pois_points = convert_to_points(pois_gdf)
    
    # Save raw POIs for future use
    os.makedirs("data/processed", exist_ok=True)
    pois_output = "data/processed/sightseeing_culture_pois.gpkg"
    print(f"\nSaving raw POI data to {pois_output}...")
    pois_points.to_file(pois_output, driver="GPKG")
    print(f"✓ Saved {len(pois_points)} POIs")
    
    # Step 4: Spatial join and aggregate
    print("\n=== STEP 4: Aggregate POIs into Hexagons ===")
    hex_gdf = spatial_join_and_aggregate(hex_gdf, pois_points)
    
    # Save aggregated data
    hex_output = "data/processed/sightseeing_culture_hexagons.gpkg"
    print(f"\nSaving aggregated hexagon data to {hex_output}...")
    hex_gdf.to_file(hex_output, driver="GPKG")
    print(f"✓ Saved {len(hex_gdf)} hexagons with culture_count column")
    
    # Step 5: Visualize
    print("\n=== STEP 5: Visualize Results ===")
    visualize_results(hex_gdf)
    
    print("\n✅ Culture/Sightseeing aggregation completed successfully!")
    print("Expected hotspots: Manhattan (Midtown, Financial District), Brooklyn (Museums)")
    print("\n📁 Saved files:")
    print(f"  - {pois_output} (raw POI points)")
    print(f"  - {hex_output} (aggregated hexagons)")

if __name__ == "__main__":
    main()
