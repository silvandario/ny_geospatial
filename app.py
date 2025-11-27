"""
NYC Vibe & Value Finder - Streamlit Dashboard
==============================================
Interactive dashboard for finding the perfect NYC neighborhood based on
lifestyle preferences, budget, and quality of life factors.

Author: Senior Data Engineer
Date: 2025-11-27
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="NYC Vibe & Value Finder",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# LOAD DATA
# ========================================

@st.cache_data
def load_data():
    """Load the final processed dataset."""
    gdf = gpd.read_file("data/processed/nyc_final_data.gpkg")
    return gdf

# Load data
gdf = load_data()

# ========================================
# HEADER
# ========================================

st.title("🗽 NYC Vibe & Value Finder")
st.markdown("### Find your perfect neighborhood based on your lifestyle – not just the price.")
st.markdown("---")

# ========================================
# SIDEBAR - CONTROLS
# ========================================

st.sidebar.header("⚙️ Your Preferences")

# GROUP A: HARD FACTS (Budget Filter)
st.sidebar.subheader("💰 Budget")
max_price = st.sidebar.slider(
    "Maximum Price per Night ($)",
    min_value=0,
    max_value=int(gdf['price_avg'].max()),
    value=200,
    step=10,
    help="Only hexagons with average Airbnb prices below this value will be displayed."
)

st.sidebar.markdown("---")

# GROUP B: LIFESTYLE (Positive Criteria)
st.sidebar.subheader("✨ Your Vibe (Lifestyle)")
st.sidebar.markdown("*The higher the value, the more important this criterion is to you.*")

w_nightlife = st.sidebar.slider("🍸 Nightlife", 0, 10, 5, help="Bars, Clubs, Music Venues")
w_culture = st.sidebar.slider("🎭 Culture", 0, 10, 7, help="Museums, Theaters, Galleries, Attractions")
w_restaurants = st.sidebar.slider("☕ Dining", 0, 10, 6, help="Restaurants, Cafés, Food Scene")
w_green = st.sidebar.slider("🌳 Parks & Nature", 0, 10, 5, help="Green Spaces, Parks, Recreation")
w_shopping = st.sidebar.slider("🛍️ Shopping", 0, 10, 4, help="Shopping Options")

st.sidebar.markdown("---")

# GROUP C: REALITY (Quality of Life)
st.sidebar.subheader("🎯 Reality Check (Quality)")
st.sidebar.markdown("*10 = Best Quality (very safe, quiet, clean)*")

w_safety = st.sidebar.slider("👮 Safety", 0, 10, 9, help="Low Crime Rate")
w_quiet = st.sidebar.slider("🤫 Quiet", 0, 10, 6, help="Low Noise Pollution")
w_clean = st.sidebar.slider("✨ Cleanliness", 0, 10, 7, help="Few Rat Sightings")
w_mobility = st.sidebar.slider("🚇 Public Transit", 0, 10, 8, help="Subway, Bus, Transport Access")

# ========================================
# CALCULATE FINAL SCORE
# ========================================

# Filter by budget
df_filtered = gdf[gdf['price_avg'] <= max_price].copy()

if len(df_filtered) == 0:
    st.error("❌ No hexagons found in this price range. Increase your budget!")
    st.stop()

# Calculate weighted final score
df_filtered['final_score'] = (
    (df_filtered['score_nightlife'] * w_nightlife) +
    (df_filtered['score_culture'] * w_culture) +
    (df_filtered['score_restaurants'] * w_restaurants) +
    (df_filtered['score_green'] * w_green) +
    (df_filtered['score_shopping'] * w_shopping) +
    (df_filtered['score_safety'] * w_safety) +
    (df_filtered['score_quiet'] * w_quiet) +
    (df_filtered['score_clean'] * w_clean) +
    (df_filtered['score_mobility'] * w_mobility)
)

# Normalize to 0-10 scale
max_possible_score = (w_nightlife + w_culture + w_restaurants + w_green + w_shopping + 
                      w_safety + w_quiet + w_clean + w_mobility)

if max_possible_score > 0:
    df_filtered['final_score_normalized'] = (df_filtered['final_score'] / max_possible_score) * 10
else:
    df_filtered['final_score_normalized'] = 0

# Get winner hexagon
winner = df_filtered.nlargest(1, 'final_score_normalized').iloc[0]

# ========================================
# MAIN AREA - WINNER METRICS
# ========================================

st.subheader("🏆 Your Perfect Neighborhood")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Top Hexagon",
        value=f"{winner['h3_id'][:10]}...",
        delta=None
    )

with col2:
    st.metric(
        label="Match Score",
        value=f"{winner['final_score_normalized']:.1f} / 10",
        delta=None,
        help="How well does this neighborhood match your preferences?"
    )

with col3:
    st.metric(
        label="Price per Night",
        value=f"${winner['price_avg']:.0f}",
        delta=None
    )

with col4:
    safety_label = "High" if winner['score_safety'] > 0.7 else "Medium" if winner['score_safety'] > 0.4 else "Low"
    st.metric(
        label="Safety",
        value=safety_label,
        delta=None,
        help=f"Safety Score: {winner['score_safety']:.2f}"
    )

st.markdown("---")

# ========================================
# MAIN AREA - INTERACTIVE MAP
# ========================================

st.subheader("🗺️ Interactive Map")

# Create Folium map
m = folium.Map(
    location=[40.7128, -74.0060],  # NYC coordinates
    zoom_start=11,
    tiles='CartoDB positron'  # Clean, light basemap
)

# Create choropleth with smooth color gradient based on final_score_normalized
# Color scale: Red (0-3) -> Orange/Yellow (4-7) -> Green (8-10)
def get_color(score):
    """
    Get smooth gradient color based on score (0-10).
    Uses RGB interpolation for smooth transitions.
    """
    # Normalize score to 0-1 range
    normalized = score / 10.0
    
    if score <= 3:  # Red zone (0-3)
        # Deep red to orange-red
        r = 231
        g = int(76 + (normalized / 0.3) * 80)  # 76 -> 156
        b = int(60 + (normalized / 0.3) * 20)  # 60 -> 80
    elif score <= 7:  # Yellow/Orange zone (4-7)
        # Orange to yellow
        progress = (score - 3) / 4  # 0 to 1 within this range
        r = int(243 - progress * 3)  # 243 -> 240
        g = int(156 + progress * 100)  # 156 -> 256
        b = int(18 + progress * 60)  # 18 -> 78
    else:  # Green zone (8-10)
        # Yellow-green to pure green
        progress = (score - 7) / 3  # 0 to 1 within this range
        r = int(241 - progress * 195)  # 241 -> 46
        g = int(196 + progress * 48)  # 196 -> 244
        b = int(15 + progress * 51)  # 15 -> 66
    
    return f'#{r:02x}{g:02x}{b:02x}'

# Add hexagons to map
for idx, row in df_filtered.iterrows():
    # Create tooltip content
    tooltip_html = f"""
    <div style="font-family: Arial; font-size: 12px;">
        <b>Score:</b> {row['final_score_normalized']:.1f}/10<br>
        <b>Price:</b> ${row['price_avg']:.0f}/night<br>
        <b>Safety:</b> {row['score_safety']*10:.1f}/10<br>
        <b>Nightlife:</b> {row['score_nightlife']*10:.1f}/10<br>
        <b>Culture:</b> {row['score_culture']*10:.1f}/10<br>
        <b>Parks:</b> {row['score_green']*10:.1f}/10
    </div>
    """
    
    # Get color based on score
    fill_color = get_color(row['final_score_normalized'])
    
    # Add hexagon
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=fill_color: {
            'fillColor': color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.6
        },
        tooltip=folium.Tooltip(tooltip_html)
    ).add_to(m)

# Highlight winner hexagon
folium.GeoJson(
    winner['geometry'],
    style_function=lambda x: {
        'fillColor': '#9b59b6',  # Purple (Violett)
        'color': 'black',
        'weight': 3,
        'fillOpacity': 0.8
    },
    tooltip=folium.Tooltip("<b>🏆 Top Match!</b>")
).add_to(m)

# Display map
st_folium(m, width=1400, height=600)

# ========================================
# FOOTER - STATISTICS
# ========================================

st.markdown("---")
st.subheader("📊 Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Hexagons Displayed", len(df_filtered))

with col2:
    st.metric("Average Price", f"${df_filtered['price_avg'].mean():.0f}")

with col3:
    st.metric("Avg. Match Score", f"{df_filtered['final_score_normalized'].mean():.1f}/10")

# ========================================
# SIDEBAR - LEGEND
# ========================================

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Map Legend")
st.sidebar.markdown("""
**Colors:**
- 🟢 Green: High Score (8-10)
- 🟡 Yellow/Orange: Medium Score (4-7)
- 🔴 Red: Low Score (0-3)
- 🟣 Purple: Top Match (Winner)

**Note:** Colors blend smoothly for better differentiation.

**Controls:**
- Hover over hexagons for details
- Zoom and pan the map
""")

st.sidebar.markdown("---")
