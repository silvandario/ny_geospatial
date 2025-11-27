# AI Recommendations Feature
# This file adds AI-powered recommendations to app.py
# Insert this code after line 389 (after st.markdown("---"))


# ========================================
# AI RECOMMENDATIONS
# ========================================

st.markdown("### 🤖 AI Recommendations")

# Function to generate AI recommendations
def generate_ai_recommendations(neighborhood_name, user_criteria, text_input=None, audio_input=None):
    high_priority_criteria = []
    for key, value in user_criteria.items():
        if value >= 7:
            criterion_name = key.replace('w_', '').replace('_', ' ').title()
            high_priority_criteria.append(f"{criterion_name} ({value}/10)")
    
    criteria_text = ", ".join(high_priority_criteria) if high_priority_criteria else "No specific priorities set"
    
    prompt = f"""You are a NYC neighborhood expert. A visitor is interested in {neighborhood_name}, NYC.

Their priorities are: {criteria_text}"""
    
    if text_input:
        prompt += f"\n\nThey also mentioned: \"{text_input}\""
    if audio_input:
        prompt += f"\n\nFrom their voice input: \"{audio_input}\""
    
    prompt += """

Provide 4-5 specific, actionable recommendations for activities, places to visit, or things to do in this neighborhood.

Format as bullet points starting with •
Focus on real places that exist in this neighborhood.
Be specific and enthusiastic.
Keep each recommendation to one concise line."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful NYC neighborhood guide providing specific local recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"• Unable to generate recommendations: {str(e)}"

# Get user criteria
user_criteria = {}
for criterion in ['w_nightlife', 'w_culture', 'w_restaurants', 'w_green', 'w_shopping', 'w_safety', 'w_quiet', 'w_clean', 'w_mobility']:
    user_criteria[criterion] = st.session_state.get(criterion, 0)

# Check for optional inputs
text_input = st.session_state.get('ai_text_input', None)
audio_input = st.session_state.get('audio_transcript', None)

# Generate & cache recommendations
@st.cache_data(show_spinner=False)
def cached_recommendations(neighborhood, criteria_str, text, audio):
    return generate_ai_recommendations(neighborhood, json.loads(criteria_str), text, audio)

with st.spinner("🤖 Generating personalized recommendations..."):
    recommendations = cached_recommendations(neighborhood_name, json.dumps(user_criteria), text_input, audio_input)

st.markdown(recommendations)
st.markdown("---")
