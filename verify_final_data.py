import geopandas as gpd
import pandas as pd

# Load the final dataset
print("Loading final dataset...")
gdf = gpd.read_file('data/processed/nyc_final_data.gpkg')

print("\n" + "="*60)
print("FINAL DATASET VERIFICATION")
print("="*60)

print(f"\nTotal Hexagons: {len(gdf):,}")
print(f"Total Columns: {len(gdf.columns)}")

# ========================================
# CHECK 1: CRS VERIFICATION
# ========================================
print("\n" + "="*60)
print("CHECK 1: CRS (Coordinate Reference System)")
print("="*60)
if gdf.crs is None:
    print("ALARM: No CRS defined!")
elif gdf.crs.to_string() == "EPSG:4326":
    print("OK: CRS is EPSG:4326 (WGS 84 - correct for web maps)")
else:
    print(f"WARNING: CRS is {gdf.crs.to_string()} (expected EPSG:4326)")

# ========================================
# CHECK 2: NaN CHECK
# ========================================
print("\n" + "="*60)
print("CHECK 2: NaN (Missing Values)")
print("="*60)

nan_counts = gdf.isna().sum()
nan_counts = nan_counts[nan_counts > 0]

if len(nan_counts) == 0:
    print("OK: No NaN values found in any column")
else:
    print("WARNING: Found NaN values:")
    for col, count in nan_counts.items():
        pct = (count / len(gdf)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

# ========================================
# CHECK 3: NULL/ZERO CHECK FOR SCORES
# ========================================
print("\n" + "="*60)
print("CHECK 3: Zero-Value Check (Empty Criteria)")
print("="*60)

score_columns = [col for col in gdf.columns if col.startswith('score_')]

all_ok = True
for col in score_columns:
    max_val = gdf[col].max()
    if max_val == 0:
        print(f"  ALARM: {col} - Max is 0! (Spatial join failed or no data)")
        all_ok = False
    else:
        print(f"  OK: {col} - Max is {max_val:.3f}")

if all_ok:
    print("\nOK: All score criteria have valid data")

# ========================================
# CHECK 4: SCORE RANGE VERIFICATION
# ========================================
print("\n" + "="*60)
print("CHECK 4: Score Range Verification (0-1)")
print("="*60)

all_valid = True
for col in score_columns:
    min_val = gdf[col].min()
    max_val = gdf[col].max()
    
    if min_val < 0 or max_val > 1:
        print(f"  ERROR: {col} - Out of range! ({min_val:.3f} - {max_val:.3f})")
        all_valid = False
    else:
        print(f"  OK: {col} - Range: {min_val:.3f} - {max_val:.3f}")

if all_valid:
    print("\nOK: All scores are within valid range [0, 1]")

# ========================================
# SUMMARY STATISTICS
# ========================================
print("\n" + "="*60)
print("SCORE STATISTICS (0=bad, 1=good)")
print("="*60)

for col in score_columns:
    name = col.replace('score_', '').replace('_', ' ').title()
    mean = gdf[col].mean()
    median = gdf[col].median()
    print(f"\n{name}:")
    print(f"  Mean:   {mean:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Range:  {gdf[col].min():.3f} - {gdf[col].max():.3f}")

# ========================================
# AIRBNB PRICE STATISTICS
# ========================================
print("\n" + "="*60)
print("AIRBNB PRICE STATISTICS")
print("="*60)
print(f"Min:    ${gdf['price_avg'].min():.2f}")
print(f"Max:    ${gdf['price_avg'].max():.2f}")
print(f"Mean:   ${gdf['price_avg'].mean():.2f}")
print(f"Median: ${gdf['price_avg'].median():.2f}")

# Check for price anomalies
cheap = gdf[gdf['price_avg'] < 50]
expensive = gdf[gdf['price_avg'] > 500]
print(f"\nHexagons with price < $50:  {len(cheap)}")
print(f"Hexagons with price > $500: {len(expensive)}")

# ========================================
# TOP HEXAGONS
# ========================================
print("\n" + "="*60)
print("TOP 5 HEXAGONS BY SAFETY SCORE")
print("="*60)
top_safe = gdf.nlargest(5, 'score_safety')[['h3_id', 'score_safety', 'price_avg']]
for idx, row in top_safe.iterrows():
    print(f"  {row['h3_id']}: Safety={row['score_safety']:.3f}, Price=${row['price_avg']:.2f}")

print("\n" + "="*60)
print("TOP 5 CHEAPEST HEXAGONS")
print("="*60)
top_cheap = gdf.nsmallest(5, 'price_avg')[['h3_id', 'price_avg', 'score_safety']]
for idx, row in top_cheap.iterrows():
    print(f"  {row['h3_id']}: Price=${row['price_avg']:.2f}, Safety={row['score_safety']:.3f}")

# ========================================
# FINAL VERDICT
# ========================================
print("\n" + "="*60)
print("DATASET VERIFICATION COMPLETE")
print("="*60)

# Count issues
issues = 0
if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
    issues += 1
if len(nan_counts) > 0:
    issues += 1
if not all_ok:
    issues += 1
if not all_valid:
    issues += 1

if issues == 0:
    print("SUCCESS: All checks passed!")
    print("Dataset is ready for Streamlit dashboard.")
else:
    print(f"WARNING: Found {issues} issue(s)")
    print("Review the checks above before proceeding.")

print("="*60)
