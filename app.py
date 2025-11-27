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
st.markdown("### Finde dein perfektes Viertel basierend auf deinem Lifestyle – nicht nur dem Preis.")
st.markdown("---")

# ========================================
# SIDEBAR - CONTROLS
# ========================================

st.sidebar.header("⚙️ Deine Präferenzen")

# GROUP A: HARD FACTS (Budget Filter)
st.sidebar.subheader("💰 Budget")
max_price = st.sidebar.slider(
    "Maximaler Preis pro Nacht ($)",
    min_value=0,
    max_value=int(gdf['price_avg'].max()),
    value=200,
    step=10,
    help="Nur Hexagone mit durchschnittlichen Airbnb-Preisen unter diesem Wert werden angezeigt."
)

st.sidebar.markdown("---")

# GROUP B: LIFESTYLE (Positive Criteria)
st.sidebar.subheader("✨ Dein Vibe (Lifestyle)")
st.sidebar.markdown("*Je höher der Wert, desto wichtiger ist dir dieses Kriterium.*")

w_nightlife = st.sidebar.slider("🍸 Nachtleben", 0, 10, 5, help="Bars, Clubs, Musik-Venues")
w_culture = st.sidebar.slider("🎭 Kultur", 0, 10, 7, help="Museen, Theater, Galerien, Sehenswürdigkeiten")
w_restaurants = st.sidebar.slider("☕ Gastronomie", 0, 10, 6, help="Restaurants, Cafés, Food-Szene")
w_green = st.sidebar.slider("🌳 Parks & Natur", 0, 10, 5, help="Grünflächen, Parks, Erholung")
w_shopping = st.sidebar.slider("🛍️ Shopping", 0, 10, 4, help="Einkaufsmöglichkeiten")

st.sidebar.markdown("---")

# GROUP C: REALITY (Quality of Life)
st.sidebar.subheader("🎯 Die Realität (Qualität)")
st.sidebar.markdown("*10 = Beste Qualität (sehr sicher, ruhig, sauber)*")

w_safety = st.sidebar.slider("👮 Sicherheit", 0, 10, 9, help="Niedrige Kriminalitätsrate")
w_quiet = st.sidebar.slider("🤫 Ruhe", 0, 10, 6, help="Wenig Lärmbelästigung")
w_clean = st.sidebar.slider("✨ Sauberkeit", 0, 10, 7, help="Wenig Ratten-Sichtungen")
w_mobility = st.sidebar.slider("🚇 ÖPNV-Nähe", 0, 10, 8, help="U-Bahn, Bus, Transport")

# ========================================
# CALCULATE FINAL SCORE
# ========================================

# Filter by budget
df_filtered = gdf[gdf['price_avg'] <= max_price].copy()

if len(df_filtered) == 0:
    st.error("❌ Keine Hexagone in diesem Preisbereich gefunden. Erhöhe dein Budget!")
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

st.subheader("🏆 Dein perfektes Viertel")

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
        help="Wie gut passt dieses Viertel zu deinen Präferenzen?"
    )

with col3:
    st.metric(
        label="Preis pro Nacht",
        value=f"${winner['price_avg']:.0f}",
        delta=None
    )

with col4:
    safety_label = "Hoch" if winner['score_safety'] > 0.7 else "Mittel" if winner['score_safety'] > 0.4 else "Niedrig"
    st.metric(
        label="Sicherheit",
        value=safety_label,
        delta=None,
        help=f"Safety Score: {winner['score_safety']:.2f}"
    )

st.markdown("---")

# ========================================
# MAIN AREA - INTERACTIVE MAP
# ========================================

st.subheader("🗺️ Interaktive Karte")

# Create Folium map
m = folium.Map(
    location=[40.7128, -74.0060],  # NYC coordinates
    zoom_start=11,
    tiles='CartoDB positron'  # Clean, light basemap
)

# Create choropleth with color based on final_score_normalized
# Color scale: Red (0) -> Yellow (5) -> Green (10)
def get_color(score):
    """Get color based on score (0-10)."""
    if score >= 8:
        return '#2ecc71'  # Green
    elif score >= 6:
        return '#27ae60'  # Dark green
    elif score >= 4:
        return '#f39c12'  # Orange
    elif score >= 2:
        return '#e67e22'  # Dark orange
    else:
        return '#e74c3c'  # Red

# Add hexagons to map
for idx, row in df_filtered.iterrows():
    # Create tooltip content
    tooltip_html = f"""
    <div style="font-family: Arial; font-size: 12px;">
        <b>Score:</b> {row['final_score_normalized']:.1f}/10<br>
        <b>Preis:</b> ${row['price_avg']:.0f}/Nacht<br>
        <b>Sicherheit:</b> {row['score_safety']*10:.1f}/10<br>
        <b>Nachtleben:</b> {row['score_nightlife']*10:.1f}/10<br>
        <b>Kultur:</b> {row['score_culture']*10:.1f}/10<br>
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
        'fillColor': '#FFD700',  # Gold
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
st.subheader("📊 Statistiken")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Hexagone angezeigt", len(df_filtered))

with col2:
    st.metric("Durchschnittspreis", f"${df_filtered['price_avg'].mean():.0f}")

with col3:
    st.metric("Durchschn. Match-Score", f"{df_filtered['final_score_normalized'].mean():.1f}/10")

# ========================================
# SIDEBAR - LEGEND
# ========================================

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Karten-Legende")
st.sidebar.markdown("""
**Farben:**
- 🟢 Grün: Hoher Score (8-10)
- 🟡 Gelb/Orange: Mittlerer Score (4-8)
- 🔴 Rot: Niedriger Score (0-4)
- 🟡 Gold: Top Match (Gewinner)

**Bedienung:**
- Hovere über Hexagone für Details
- Zoome und verschiebe die Karte
""")

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit & H3")
