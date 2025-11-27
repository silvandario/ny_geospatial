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
from geopy.geocoders import Nominatim
from functools import lru_cache
import openai
import json
# 1. Text Control (OpenAI)
# ========================================
# LOAD SECRETS
# ========================================

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("❌ Missing API Keys in .streamlit/secrets.toml")
    st.stop()

openai.api_key = OPENAI_API_KEY

# ========================================
# HELPER FUNCTIONS
# ========================================

@lru_cache(maxsize=100)
def get_neighborhood_name(lat, lon):
    """
    Get neighborhood name from coordinates using reverse geocoding.
    Cached to avoid repeated API calls for the same location.
    """
    try:
        geolocator = Nominatim(user_agent="nyc_vibe_finder")
        location = geolocator.reverse(f"{lat}, {lon}", language='en', timeout=10)
        
        if location and location.raw.get('address'):
            address = location.raw['address']
            
            # Try to get neighborhood, suburb, or borough
            neighborhood = (
                address.get('neighbourhood') or 
                address.get('suburb') or 
                address.get('city_district') or
                address.get('borough') or
                address.get('town') or
                'NYC'
            )
            
            return neighborhood
    except Exception as e:
        # Fallback to just "NYC" if geocoding fails
        return "NYC"
    
    return "NYC"

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
# VALIDATE REQUIRED COLUMNS
# ========================================

# List of required columns
required_columns = [
    'h3_id', 'geometry', 'price_avg',
    'score_nightlife', 'score_culture', 'score_restaurants',
    'score_green', 'score_mobility', 'score_shopping',
    'score_safety', 'score_clean', 'score_quiet'
]

# Check for missing columns
missing_columns = [col for col in required_columns if col not in gdf.columns]

if missing_columns:
    st.error(f"❌ ERROR: Missing required columns in dataset: {', '.join(missing_columns)}")
    st.info("Available columns: " + ", ".join(gdf.columns.tolist()))
    st.stop()

# ========================================
# HEADER
# ========================================

st.title("🗽 NYC Vibe & Value Finder")
st.markdown("### Find your perfect neighborhood based on your lifestyle – not just the price.")
st.markdown("---")

# ========================================
# INITIALIZE SESSION STATE
# ========================================

defaults = {
    "max_price": int(gdf['price_avg'].max()),
    "w_nightlife": 0,
    "w_culture": 0,
    "w_restaurants": 0,
    "w_green": 0,
    "w_shopping": 0,
    "w_safety": 0,
    "w_quiet": 0,
    "w_clean": 0,
    "w_mobility": 0
}

for key, default_val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# ========================================
# SIDEBAR - CONTROLS
# ========================================

st.sidebar.header("⚙️ Your Preferences")

# ========================================
# AI CONTROL
# ========================================
st.sidebar.subheader("🤖 AI Assistant")
st.sidebar.markdown("**Text Control**")
text_input = st.sidebar.text_area("Describe your ideal neighborhood:", height=70)

if st.sidebar.button("✨ Update Filters"):
    if text_input:
        try:
            with st.spinner("AI is thinking..."):
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """
                        You are an AI assistant controlling a Streamlit dashboard.
                        Extract user preferences and return a JSON object with the following keys (only if relevant):
                        - max_price (int)
                        - w_nightlife (0-10)
                        - w_culture (0-10)
                        - w_restaurants (0-10)
                        - w_green (0-10)
                        - w_shopping (0-10)
                        - w_safety (0-10)
                        - w_quiet (0-10)
                        - w_clean (0-10)
                        - w_mobility (0-10)
                        
                        LOGIC RULES:
                        1. If the user likes/wants something, set its weight to 8-10.
                        2. If the user explicitly says they do NOT like or do NOT care about something (e.g. "I don't like it clean", "safety doesn't matter"), set that weight to 0.
                        3. CRITICAL: If the user expresses a negative preference (e.g. "not clean"), you MUST set the weights of other "neutral" positive traits to 5 (baseline). 
                           - Example: "I don't like it clean" -> Set w_clean=0, and set w_safety=5, w_quiet=5, w_mobility=5, etc. (unless specified otherwise).
                           - This ensures that "not clean" actually means "cleanliness is less important than other factors".
                           - If all weights are 0, the ranking falls back to price only, which ignores the user's specific dislike.
                        
                        Return ONLY the JSON object.
                        """},
                        {"role": "user", "content": text_input}
                    ]
                )
                
                content = response.choices[0].message.content
                # Parse JSON (handle potential markdown code blocks)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                updates = json.loads(content)
                
                # Update session state
                updated_count = 0
                for key, value in updates.items():
                    if key in st.session_state:
                        st.session_state[key] = value
                        updated_count += 1
                
                if updated_count > 0:
                    st.success(f"Updated {updated_count} preferences!")
                    st.rerun()
                else:
                    st.warning("No relevant preferences found.")
                    
        except Exception as e:
            st.error(f"AI Error: {str(e)}")

st.sidebar.markdown("---")

# GROUP A: HARD FACTS (Budget Filter)
st.sidebar.subheader("💰 Budget")
st.sidebar.slider(
    "Maximum Price per Night ($)",
    min_value=0,
    max_value=int(gdf['price_avg'].max()),
    step=10,
    key="max_price",
    help="Only hexagons with average Airbnb prices below this value will be displayed."
)

st.sidebar.markdown("---")

# GROUP B: LIFESTYLE (Positive Criteria)
st.sidebar.subheader("✨ Your Vibe (Lifestyle)")
st.sidebar.markdown("*The higher the value, the more important this criterion is to you.*")

st.sidebar.slider("🍸 Nightlife", 0, 10, key="w_nightlife", help="Bars, Clubs, Music Venues")
st.sidebar.slider("🎭 Culture", 0, 10, key="w_culture", help="Museums, Theaters, Galleries, Attractions")
st.sidebar.slider("☕ Dining", 0, 10, key="w_restaurants", help="Restaurants, Cafés, Food Scene")
st.sidebar.slider("🌳 Parks & Nature", 0, 10, key="w_green", help="Green Spaces, Parks, Recreation")
st.sidebar.slider("🛍️ Shopping", 0, 10, key="w_shopping", help="Shopping Options")

st.sidebar.markdown("---")

# GROUP C: REALITY (Quality of Life)
st.sidebar.subheader("🎯 Reality Check (Quality)")
st.sidebar.markdown("*10 = Best Quality (very safe, quiet, clean)*")

st.sidebar.slider("👮 Safety", 0, 10, key="w_safety", help="Low Crime Rate")
st.sidebar.slider("🤫 Quiet", 0, 10, key="w_quiet", help="Low Noise Pollution")
st.sidebar.slider("✨ Cleanliness", 0, 10, key="w_clean", help="Few Rat Sightings")
st.sidebar.slider("🚇 Public Transit", 0, 10, key="w_mobility", help="Subway, Bus, Transport Access")

# ========================================
# CALCULATE FINAL SCORE
# ========================================

# Filter by budget
df_filtered = gdf[gdf['price_avg'] <= st.session_state.max_price].copy()

if len(df_filtered) == 0:
    st.error("❌ No hexagons found in this price range. Increase your budget!")
    st.stop()

# Calculate weighted final score
df_filtered['final_score'] = (
    (df_filtered['score_nightlife'] * st.session_state.w_nightlife) +
    (df_filtered['score_culture'] * st.session_state.w_culture) +
    (df_filtered['score_restaurants'] * st.session_state.w_restaurants) +
    (df_filtered['score_green'] * st.session_state.w_green) +
    (df_filtered['score_shopping'] * st.session_state.w_shopping) +
    (df_filtered['score_safety'] * st.session_state.w_safety) +
    (df_filtered['score_quiet'] * st.session_state.w_quiet) +
    (df_filtered['score_clean'] * st.session_state.w_clean) +
    (df_filtered['score_mobility'] * st.session_state.w_mobility)
)

# Normalize to 0-10 scale
max_possible_score = (st.session_state.w_nightlife + st.session_state.w_culture + st.session_state.w_restaurants + st.session_state.w_green + st.session_state.w_shopping + 
                      st.session_state.w_safety + st.session_state.w_quiet + st.session_state.w_clean + st.session_state.w_mobility)

# EDGE CASE: All weights are 0 (user only cares about price)
if max_possible_score == 0:
    # Rank by price: cheapest = 10, most expensive = 0
    # Invert price so cheaper is better
    min_price = df_filtered['price_avg'].min()
    max_price_in_range = df_filtered['price_avg'].max()
    
    if max_price_in_range > min_price:
        # Inverted normalization: cheaper = higher score
        df_filtered['final_score_normalized'] = 10 * (1 - (df_filtered['price_avg'] - min_price) / (max_price_in_range - min_price))
    else:
        # All prices are the same
        df_filtered['final_score_normalized'] = 5.0
    
    # Show info message
    st.info("ℹ️ All criteria weights are 0. Showing results ranked by price (cheapest = best).")
else:
    # Normal case: calculate weighted score
    df_filtered['final_score_normalized'] = (df_filtered['final_score'] / max_possible_score) * 10

# Get winner hexagon
winner = df_filtered.nlargest(1, 'final_score_normalized').iloc[0]

# Get neighborhood name from winner hexagon centroid
winner_centroid = winner['geometry'].centroid
neighborhood_name = get_neighborhood_name(winner_centroid.y, winner_centroid.x)

# ========================================
# MAIN AREA - WINNER METRICS
# ========================================

st.subheader("🏆 Your Perfect Neighborhood")

# Helper function for traffic light indicator
def get_traffic_light(score):
    """Return traffic light emoji based on score (0-1 scale)."""
    if score >= 0.8:
        return "🟢"  # Green: Good
    elif score >= 0.3:
        return "🟡"  # Yellow: Medium
    else:
        return "🔴"  # Red: Poor

# ROW 1: Main summary (3 columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Top Neighborhood",
        value=neighborhood_name,
        delta=None,
        help=f"H3 ID: {winner['h3_id']}"
    )

with col2:
    st.metric(
        label="Match Score",
        value=f"{winner['final_score_normalized']:.1f} / 10",
        delta=None
    )

with col3:
    st.metric(
        label="Price per Night",
        value=f"${winner['price_avg']:.0f}",
        delta=None
    )

st.markdown("---")

# ROW 2: Detailed criteria with traffic lights
st.markdown("**📊 Neighborhood Details**")

# Create 3 columns for 9 criteria (3 per column)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"{get_traffic_light(winner['score_safety'])} **Safety**: {winner['score_safety']*10:.1f}/10")
    st.markdown(f"{get_traffic_light(winner['score_quiet'])} **Quiet**: {winner['score_quiet']*10:.1f}/10")
    st.markdown(f"{get_traffic_light(winner['score_clean'])} **Cleanliness**: {winner['score_clean']*10:.1f}/10")

with col2:
    st.markdown(f"{get_traffic_light(winner['score_mobility'])} **Public Transit**: {winner['score_mobility']*10:.1f}/10")
    st.markdown(f"{get_traffic_light(winner['score_nightlife'])} **Nightlife**: {winner['score_nightlife']*10:.1f}/10")
    st.markdown(f"{get_traffic_light(winner['score_culture'])} **Culture**: {winner['score_culture']*10:.1f}/10")

with col3:
    st.markdown(f"{get_traffic_light(winner['score_restaurants'])} **Dining**: {winner['score_restaurants']*10:.1f}/10")
    st.markdown(f"{get_traffic_light(winner['score_green'])} **Parks & Nature**: {winner['score_green']*10:.1f}/10")
    st.markdown(f"{get_traffic_light(winner['score_shopping'])} **Shopping**: {winner['score_shopping']*10:.1f}/10")

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
