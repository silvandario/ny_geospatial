"""
NYC Vibe & Value Finder - Streamlit Dashboard with Voice Agent
===============================================================
Interactive dashboard with VAPI voice agent for finding perfect NYC neighborhoods.

Author: Feisal Abassy & Silvan Ladner
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
    """Get neighborhood name from coordinates using reverse geocoding."""
    try:
        geolocator = Nominatim(user_agent="nyc_vibe_finder")
        location = geolocator.reverse(f"{lat}, {lon}", language='en', timeout=10)
        
        if location and location.raw.get('address'):
            address = location.raw['address']
            
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

required_columns = [
    'h3_id', 'geometry', 'price_avg',
    'score_nightlife', 'score_culture', 'score_restaurants',
    'score_green', 'score_mobility', 'score_shopping',
    'score_safety', 'score_clean', 'score_quiet'
]

missing_columns = [col for col in required_columns if col not in gdf.columns]

if missing_columns:
    st.error(f"❌ ERROR: Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# ========================================
# SYSTEM PROMPT DEFINITION
# ========================================
AI_SYSTEM_PROMPT = """
You are a configuration assistant for a NYC travel dashboard. 
Your goal is to translate natural language user requests into a JSON configuration for sliders.

Output Format: JSON only.
Keys: 
- max_price (integer, default 500 if not mentioned)
- w_nightlife, w_culture, w_restaurants, w_green, w_shopping (Lifestyle Weights 0-10)
- w_safety, w_quiet, w_clean, w_mobility (Quality Weights 0-10)

SEMANTIC MAPPING (How our data works):
- **w_green**: Measures **Parks, Gyms, Fitness, Sports**. If user mentions "gym", "workout", "bodybuilding", "athlete", or "nature" -> set this high.
- **w_clean**: Measures **Rat Sightings**. If user mentions "rats", "mice", "vermin", or "dirty streets" -> set this high.
- **w_quiet**: Measures **Noise Complaints** (Parties, Construction). If user wants "no parties", "sleep well", "quiet" -> set this high.
- **w_safety**: Measures **Shootings & Traffic Accidents**. If user mentions "traffic safety", "guns", "crime", "safe streets" -> set this high.
- **w_mobility**: Measures **Subway Entrances**. If user needs "train", "commute", "easy travel" -> set this high.

LOGIC RULES FOR WEIGHTS (0-10):

1. **Strong Desire ("Love", "Must have", "Very important", "Focus on"):**
   -> Set weight to **10**.
   (e.g., "I hate rats" -> w_clean: 10)

2. **Moderate Desire ("Like", "Prefer", "Good", "Should have"):**
   -> Set weight to **6-7**.

3. **Slight Desire ("Maybe", "A bit", "If possible"):**
   -> Set weight to **3-4**.

4. **Explicit Dislike / Avoidance:**
   -> Set weight to **0** for Lifestyle categories (e.g., "No party" -> w_nightlife: 0).
   -> Set weight to **10** for Quality categories to AVOID the negative (e.g., "I hate noise" -> w_quiet: 10).

5. **Not Mentioned / Irrelevant:**
   -> Set weight to **0**. 
   (Do not include criteria that are not mentioned).

6. **Budget:**
   -> Extract number if mentioned.
   -> Keywords: "Cheap"/"Budget" -> 150. "Luxury" -> 800. Default -> 500.
"""
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
# VOICE CONTROL
# ========================================
st.sidebar.subheader("🎙️ Voice Search")
st.sidebar.markdown("**Quick Audio Search**")
st.sidebar.markdown("Record a short voice note describing what you want.")

# Audio recording
audio_component = st.sidebar.audio_input("Record your voice")

if audio_component:
    if st.sidebar.button("🔍 Analyze Voice Input"):
        with st.spinner("Transcribing and analyzing..."):
            try:
                # Save audio temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_component.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Transcribe
                with open(tmp_file_path, "rb") as audio_file:
                    transcript = openai.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=audio_file
                    )
                
                transcribed_text = transcript.text
                st.sidebar.info(f"📝 You said: {transcribed_text}")
                
                # Use existing OpenAI analysis logic
                response = openai.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[
                        {"role": "system", "content": AI_SYSTEM_PROMPT},
                        {"role": "user", "content": transcribed_text}
                    ]
                )
                
                content = response.choices[0].message.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                updates = json.loads(content)
                
                # Store original voice input for AI Recommendations
                st.session_state['user_specific_request'] = transcribed_text
                st.session_state['audio_transcript'] = transcribed_text

                # Update filters
                updated_count = 0
                for key, value in updates.items():
                    if key in st.session_state:
                        st.session_state[key] = value if value is not None else 0
                        updated_count += 1
                
                if updated_count > 0:
                    st.success(f"✓ Updated {updated_count} preferences from voice!")
                    st.rerun()
                else:
                    st.warning("No relevant preferences found in voice input.")
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

st.sidebar.markdown("---")

# ========================================
# AI TEXT CONTROL
# ========================================
st.sidebar.subheader("🤖 AI Text Assistant")
st.sidebar.markdown("**Describe your ideal neighborhood**")
text_input = st.sidebar.text_area("", height=70, placeholder="e.g., Safe area with parks, under $200, close to restaurants")

if st.sidebar.button("✨ Update Filters from Text"):
    if text_input:
        try:
            with st.spinner("AI is analyzing..."):
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": AI_SYSTEM_PROMPT},
                        {"role": "user", "content": text_input}
                    ]
                )
                
                content = response.choices[0].message.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                updates = json.loads(content)

                # Store original text input for AI Recommendations  
                st.session_state['user_specific_request'] = text_input
                
                updated_count = 0
                for key, value in updates.items():
                    if key in st.session_state:
                        st.session_state[key] = value if value is not None else 0
                        updated_count += 1
                
                if updated_count > 0:
                    st.success(f"✓ Updated {updated_count} preferences!")
                    st.rerun()
                else:
                    st.warning("No relevant preferences found.")
                    
        except Exception as e:
            st.error(f"AI Error: {str(e)}")

st.sidebar.markdown("---")

# ========================================
# MANUAL CONTROLS
# ========================================

# Budget
st.sidebar.subheader("💰 Budget")
st.sidebar.slider(
    "Maximum Price per Night ($)",
    min_value=0,
    max_value=int(gdf['price_avg'].max()),
    step=10,
    key="max_price"
)

st.sidebar.markdown("---")

# Lifestyle
st.sidebar.subheader("✨ Your Vibe (Lifestyle)")
st.sidebar.slider("🍸 Nightlife", 0, 10, key="w_nightlife")
st.sidebar.slider("🎭 Culture", 0, 10, key="w_culture")
st.sidebar.slider("☕ Dining", 0, 10, key="w_restaurants")
st.sidebar.slider("🌳 Parks & Nature", 0, 10, key="w_green")
st.sidebar.slider("🛍️ Shopping", 0, 10, key="w_shopping")

st.sidebar.markdown("---")

# Quality of Life
st.sidebar.subheader("🎯 Reality Check (Quality)")
st.sidebar.slider("👮 Safety", 0, 10, key="w_safety")
st.sidebar.slider("🤫 Quiet", 0, 10, key="w_quiet")
st.sidebar.slider("✨ Cleanliness", 0, 10, key="w_clean")
st.sidebar.slider("🚇 Public Transit", 0, 10, key="w_mobility")

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

# Normalize
max_possible_score = sum([
    st.session_state.w_nightlife, st.session_state.w_culture, 
    st.session_state.w_restaurants, st.session_state.w_green, 
    st.session_state.w_shopping, st.session_state.w_safety, 
    st.session_state.w_quiet, st.session_state.w_clean, 
    st.session_state.w_mobility
])

if max_possible_score == 0:
    # Rank by price
    min_price = df_filtered['price_avg'].min()
    max_price_in_range = df_filtered['price_avg'].max()
    
    if max_price_in_range > min_price:
        df_filtered['final_score_normalized'] = 10 * (1 - (df_filtered['price_avg'] - min_price) / (max_price_in_range - min_price))
    else:
        df_filtered['final_score_normalized'] = 5.0
    
    st.info("ℹ️ All criteria weights are 0. Showing results ranked by price (cheapest = best).")
else:
    df_filtered['final_score_normalized'] = (df_filtered['final_score'] / max_possible_score) * 10

# Get winner
winner = df_filtered.nlargest(1, 'final_score_normalized').iloc[0]
winner_centroid = winner['geometry'].centroid
neighborhood_name = get_neighborhood_name(winner_centroid.y, winner_centroid.x)

# ========================================
# MAIN AREA - WINNER METRICS
# ========================================

st.subheader("🏆 Your Perfect Neighborhood")

def get_traffic_light(score):
    if score >= 0.8:
        return "🟢"
    elif score >= 0.3:
        return "🟡"
    else:
        return "🔴"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Top Neighborhood", neighborhood_name)

with col2:
    st.metric("Match Score", f"{winner['final_score_normalized']:.1f} / 10")

with col3:
    st.metric("Price per Night", f"${winner['price_avg']:.0f}")

st.markdown("---")

st.markdown("**📊 Neighborhood Details**")

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
# AI RECOMMENDATIONS
# ========================================

st.markdown("### 🤖 AI Recommendations")

# Function to generate AI recommendations
def generate_ai_recommendations(neighborhood_name, user_criteria, winner_data, text_input=None, audio_input=None):
    # 1. User Priorities
    high_priority_criteria = []
    for key, value in user_criteria.items():
        if value >= 7:
            criterion_name = key.replace('w_', '').replace('_', ' ').title()
            high_priority_criteria.append(f"{criterion_name} ({value}/10)")
    
    criteria_text = ", ".join(high_priority_criteria) if high_priority_criteria else "General recommendation"
    
    # 2. Data Reality Check
    safety_status = "Very Safe" if winner_data['score_safety'] > 0.7 else "Moderate Safety" if winner_data['score_safety'] > 0.4 else "High Crime Rate"
    noise_status = "Quiet Area" if winner_data['score_quiet'] > 0.7 else "Moderate Noise" if winner_data['score_quiet'] > 0.4 else "Noisy / Party Area"
    price_val = int(winner_data['price_avg'])
    
    # 3. Construct Prompt
    prompt = f"""
    You are a NYC local expert. A visitor is interested in **{neighborhood_name}**.
    
    **DATA REALITY CHECK (Based on our analysis):**
    - Safety: {safety_status} (Score: {winner_data['score_safety']:.2f})
    - Noise Level: {noise_status}
    - Avg Price: ${price_val}/night
    
    **USER PRIORITIES:** {criteria_text}
    """
    
    if text_input:
        prompt += f"\n\n**USER SPECIFIC REQUEST:** \"{text_input}\""
    if audio_input:
        prompt += f"\n\n**USER VOICE INPUT:** \"{audio_input}\""
    
    prompt += """
    
    **TASK:**
    Provide 3 specific, actionable recommendations for this neighborhood.
    
    **RULES:**
    1. Tailor recommendations to user request (e.g. if they asked for Gyms, find Gyms).
    2. If Data Reality Check shows negative flags, add polite warning.
    3. Format as bullet points starting with •
    4. Keep concise.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful NYC neighborhood guide."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"• Unable to generate recommendations: {str(e)}"

# Get user criteria
user_criteria = {}
for criterion in ['w_nightlife', 'w_culture', 'w_restaurants', 'w_green', 'w_shopping', 'w_safety', 'w_quiet', 'w_clean', 'w_mobility']:
    user_criteria[criterion] = st.session_state.get(criterion, 0)

# Check for optional inputs
text_input_for_ai = st.session_state.get('user_specific_request', None)

# Generate & cache recommendations
@st.cache_data(show_spinner=False)
def cached_recommendations(hex_id, neighborhood, criteria_str, winner_dict, text_input):
    winner_data = pd.Series(winner_dict)  # Nimmt dict statt JSON
    return generate_ai_recommendations(neighborhood, json.loads(criteria_str), winner_data, text_input=text_input)

# Prepare winner data without geometry (causes recursion error)
winner_dict = {
    'h3_id': winner['h3_id'],
    'price_avg': winner['price_avg'],
    'score_safety': winner['score_safety'],
    'score_quiet': winner['score_quiet'],
    'score_clean': winner['score_clean'],
    'score_nightlife': winner['score_nightlife'],
    'score_culture': winner['score_culture'],
    'score_restaurants': winner['score_restaurants'],
    'score_green': winner['score_green'],
    'score_mobility': winner['score_mobility'],
    'score_shopping': winner['score_shopping']
}

with st.spinner("🤖 Generating personalized recommendations..."):
    recommendations = cached_recommendations(
        winner['h3_id'], 
        neighborhood_name, 
        json.dumps(user_criteria), 
        winner_dict,  
        text_input_for_ai
    )

st.markdown(recommendations)
st.markdown("---")

# ========================================
# MAP
# ========================================

st.subheader("🗺️ Interactive Map")

m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11,
    tiles='CartoDB positron'
)

def get_color(score):
    normalized = score / 10.0
    
    if score <= 3:
        r = 231
        g = int(76 + (normalized / 0.3) * 80)
        b = int(60 + (normalized / 0.3) * 20)
    elif score <= 7:
        progress = (score - 3) / 4
        r = int(243 - progress * 3)
        g = int(156 + progress * 100)
        b = int(18 + progress * 60)
    else:
        progress = (score - 7) / 3
        r = int(241 - progress * 195)
        g = int(196 + progress * 48)
        b = int(15 + progress * 51)
    
    return f'#{r:02x}{g:02x}{b:02x}'

for idx, row in df_filtered.iterrows():
    tooltip_html = f"""
    <div style="font-family: Arial; font-size: 12px;">
        <b>Score:</b> {row['final_score_normalized']:.1f}/10<br>
        <b>Price:</b> ${row['price_avg']:.0f}/night<br>
        <b>Safety:</b> {row['score_safety']*10:.1f}/10<br>
        <b>Nightlife:</b> {row['score_nightlife']*10:.1f}/10
    </div>
    """
    
    fill_color = get_color(row['final_score_normalized'])
    
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

# Highlight winner
folium.GeoJson(
    winner['geometry'],
    style_function=lambda x: {
        'fillColor': '#9b59b6',
        'color': 'black',
        'weight': 3,
        'fillOpacity': 0.8
    },
    tooltip=folium.Tooltip("<b>🏆 Top Match!</b>")
).add_to(m)

st_folium(m, width=2200, height=1200)

# ========================================
# SIDEBAR LEGEND
# ========================================

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Map Legend")
st.sidebar.markdown("""
**Colors:**
- 🟢 Green: High Score (8-10)
- 🟡 Yellow/Orange: Medium (4-7)
- 🔴 Red: Low Score (0-3)
- 🟣 Purple: Top Match

**Voice Features:**
-  Simple: Quick recording
""")