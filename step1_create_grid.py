import osmnx as ox
import h3
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt

def create_grid():
    """
    Loads NYC boundary, generates H3 hexagon grid, and visualizes the result.
    """
    # 1. Load NYC Boundary
    print("Loading NYC boundary from OSM...")
    try:
        nyc_gdf = ox.geocode_to_gdf("New York City, USA")
    except Exception as e:
        print(f"Error loading NYC boundary: {e}")
        return

    # Ensure CRS is EPSG:4326 (Lat/Lon)
    nyc_gdf = nyc_gdf.to_crs(epsg=4326)
    print(f"Loaded NYC boundary. Geometry type: {nyc_gdf.geometry.iloc[0].geom_type}")

    # 2. Generate Hexagons
    resolution = 8
    print(f"H3 version: {h3.__version__}")
    print(f"Generating H3 hexagons (Resolution {resolution})...")
    hex_ids = set()

    # Iterate over geometries (handling MultiPolygon)
    for idx, geom in enumerate(nyc_gdf.geometry):
        print(f"Processing geometry {idx+1}/{len(nyc_gdf)}: {geom.geom_type}")
        
        # Standardize to list of polygons
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = geom.geoms
        else:
            continue

        for i, poly in enumerate(polys):
            # Debug: Test centroid
            centroid = poly.centroid
            print(f"  Poly {i} Centroid: {centroid.y}, {centroid.x} (Lat, Lon)")
            try:
                test_cell = h3.latlng_to_cell(centroid.y, centroid.x, resolution)
                print(f"  Test H3 cell at centroid: {test_cell}")
            except Exception as e:
                print(f"  Test H3 cell failed: {e}")

            # Convert shapely polygon to GeoJSON-like dict for h3
            # mapping(poly) returns {'type': 'Polygon', 'coordinates': ...}
            
            try:
                # Method 1: GeoJSON with LISTS (not tuples)
                # h3.polygon_to_cells expects lists in GeoJSON
                geo_json = mapping(poly)
                import json
                geo_json_str = json.dumps(geo_json)
                geo_json_clean = json.loads(geo_json_str) # Now it has lists
                
                # Method: Use h3.LatLngPoly (Found via inspection)
                coords = list(poly.exterior.coords)
                # Shapely is (Lon, Lat), H3 wants (Lat, Lon)
                # Use lists [lat, lon]
                latlon_coords = [[p[1], p[0]] for p in coords]
                
                holes_latlon = []
                for interior in poly.interiors:
                    holes_latlon.append([[p[1], p[0]] for p in interior.coords])
                
                try:
                    # Create h3.LatLngPoly
                    # Assuming signature is LatLngPoly(outer, *holes) or similar
                    # If holes exist, we pass them as separate arguments or a list?
                    # Let's try passing outer loop first, then holes as varargs if supported
                    
                    if holes_latlon:
                        h3_poly = h3.LatLngPoly(latlon_coords, *holes_latlon)
                    else:
                        h3_poly = h3.LatLngPoly(latlon_coords)
                        
                    cells = h3.polygon_to_cells(h3_poly, res=resolution)
                    
                    if cells:
                        print(f"  Success: Generated {len(cells)} cells for poly {i} using h3.LatLngPoly")
                        hex_ids.update(cells)
                    else:
                        print(f"  Warning: No cells for poly {i} using h3.LatLngPoly")

                except Exception as e_poly:
                    print(f"  h3.LatLngPoly attempt failed for poly {i}: {e_poly}")
                    # Fallback: Try GeoJSON again but maybe via geo_to_h3shape?
                    try:
                        geo_json = mapping(poly)
                        import json
                        geo_json_clean = json.loads(json.dumps(geo_json))
                        shape = h3.geo_to_h3shape(geo_json_clean)
                        cells = h3.polygon_to_cells(shape, res=resolution)
                        if cells:
                             hex_ids.update(cells)
                             print(f"  Success: Generated {len(cells)} cells via geo_to_h3shape")
                    except Exception as e_shape:
                        print(f"  geo_to_h3shape failed: {e_shape}")

            except Exception as e:
                print(f"Error processing polygon {i}: {e}")

    print(f"Generated {len(hex_ids)} unique hexagons.")

    if len(hex_ids) == 0:
        print("ERROR: No hexagons generated. Check input geometry or H3 version.")
        return

    # 3. Convert H3 IDs back to Shapely Polygons
    print("Converting H3 IDs to Shapely Polygons...")
    hex_polys = []
    valid_hex_ids = []

    for hid in hex_ids:
        try:
            # h3.cell_to_boundary(h) returns tuple of (lat, lon)
            boundary = h3.cell_to_boundary(hid)
            
            # Shapely expects (lon, lat), so we swap
            boundary_lonlat = [(p[1], p[0]) for p in boundary]
            
            hex_polys.append(Polygon(boundary_lonlat))
            valid_hex_ids.append(hid)
        except Exception as e:
            print(f"Error converting hex {hid}: {e}")

    # Create GeoDataFrame
    hex_gdf = gpd.GeoDataFrame(
        {'h3_id': valid_hex_ids, 'geometry': hex_polys}, 
        crs="EPSG:4326"
    )
    
    print(f"Created GeoDataFrame with {len(hex_gdf)} rows.")

    # 4. Save Grid to File
    import os
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nyc_hex_grid.gpkg")
    
    print(f"Saving grid to {output_path}...")
    hex_gdf.to_file(output_path, driver="GPKG")
    print(f"✓ Grid saved successfully!")

    # 5. Verification Plot
    print("Plotting result...")
    try:
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot NYC boundary (Red outline)
        nyc_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2, zorder=2, label='NYC Boundary')

        # Plot Hexagons (Blue, transparent)
        hex_gdf.plot(ax=ax, color='blue', alpha=0.3, edgecolor='blue', linewidth=0.2, zorder=1, label='Hexagons')

        plt.title(f"NYC Hexagon Grid (Res {resolution}) - {len(hex_gdf)} Hexagons")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        
        print("Displaying plot window...")
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")
    
    return hex_gdf

if __name__ == "__main__":
    create_grid()
