"""
VAPI Voice Agent Integration for NYC Vibe & Value Finder
=========================================================
This module handles voice interactions using VAPI for the neighborhood search.

Author: Silvan Ladner
Date: 2025-11-27
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, Optional

class VAPIAgent:
    """
    VAPI Voice Agent for NYC Neighborhood Search
    Handles voice-based queries and converts them to filter updates.
    """
    
    def __init__(self, api_key: str, assistant_id: Optional[str] = None):
        """
        Initialize VAPI agent.
        
        Args:
            api_key: VAPI API key
            assistant_id: Optional existing VAPI assistant ID
        """
        self.api_key = api_key
        self.assistant_id = assistant_id
        self.base_url = "https://api.vapi.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def create_assistant(self) -> str:
        """
        Create a new VAPI assistant configured for NYC neighborhood search.
        
        Returns:
            Assistant ID
        """
        assistant_config = {
            "name": "NYC Neighborhood Finder",
            "voice": {
                "provider": "11labs",
                "voiceId": "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                "stability": 0.5,
                "similarityBoost": 0.75
            },
            "model": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "systemPrompt": """You are a friendly NYC neighborhood search assistant. 
                
Your job is to help users find their perfect NYC neighborhood by understanding their preferences.

Ask natural questions like:
- "What's your budget for accommodation per night?"
- "Are you looking for a lively nightlife scene or something quieter?"
- "Do you prefer being near parks and green spaces?"
- "How important is safety to you?"
- "Do you want to be close to restaurants and cafés?"

Based on their answers, extract preferences for:
- max_price (budget in dollars)
- w_nightlife (0-10): importance of bars, clubs, music venues
- w_culture (0-10): importance of museums, theaters, galleries
- w_restaurants (0-10): importance of dining options
- w_green (0-10): importance of parks and nature
- w_shopping (0-10): importance of shopping
- w_safety (0-10): importance of low crime
- w_quiet (0-10): importance of low noise
- w_clean (0-10): importance of cleanliness
- w_mobility (0-10): importance of public transport

After gathering preferences, call the update_filters function with the values."""
            },
            "functions": [
                {
                    "name": "update_filters",
                    "description": "Update the NYC neighborhood search filters based on user preferences",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_price": {
                                "type": "integer",
                                "description": "Maximum price per night in dollars"
                            },
                            "w_nightlife": {
                                "type": "integer",
                                "description": "Nightlife importance (0-10)"
                            },
                            "w_culture": {
                                "type": "integer",
                                "description": "Culture importance (0-10)"
                            },
                            "w_restaurants": {
                                "type": "integer",
                                "description": "Dining importance (0-10)"
                            },
                            "w_green": {
                                "type": "integer",
                                "description": "Parks/nature importance (0-10)"
                            },
                            "w_shopping": {
                                "type": "integer",
                                "description": "Shopping importance (0-10)"
                            },
                            "w_safety": {
                                "type": "integer",
                                "description": "Safety importance (0-10)"
                            },
                            "w_quiet": {
                                "type": "integer",
                                "description": "Quietness importance (0-10)"
                            },
                            "w_clean": {
                                "type": "integer",
                                "description": "Cleanliness importance (0-10)"
                            },
                            "w_mobility": {
                                "type": "integer",
                                "description": "Public transport importance (0-10)"
                            }
                        }
                    }
                }
            ]
        }
        
        response = requests.post(
            f"{self.base_url}/assistant",
            headers=self.headers,
            json=assistant_config
        )
        
        if response.status_code == 201:
            assistant_data = response.json()
            self.assistant_id = assistant_data["id"]
            return self.assistant_id
        else:
            raise Exception(f"Failed to create assistant: {response.text}")
    
    def start_call(self, phone_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a VAPI phone call or web call.
        
        Args:
            phone_number: Optional phone number for phone calls
            
        Returns:
            Call information
        """
        if not self.assistant_id:
            raise Exception("No assistant configured. Call create_assistant() first.")
        
        call_config = {
            "assistantId": self.assistant_id,
            "type": "webCall" if phone_number is None else "phoneCall"
        }
        
        if phone_number:
            call_config["customer"] = {"number": phone_number}
        
        response = requests.post(
            f"{self.base_url}/call",
            headers=self.headers,
            json=call_config
        )
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            raise Exception(f"Failed to start call: {response.text}")
    
    def get_web_call_token(self) -> str:
        """
        Get a web call token for browser-based voice interaction.
        
        Returns:
            Web call token
        """
        if not self.assistant_id:
            raise Exception("No assistant configured. Call create_assistant() first.")
        
        response = requests.post(
            f"{self.base_url}/call/web",
            headers=self.headers,
            json={"assistantId": self.assistant_id}
        )
        
        if response.status_code in [200, 201]:
            return response.json()["webCallUrl"]
        else:
            raise Exception(f"Failed to get web call token: {response.text}")


def render_vapi_widget():
    """
    Render VAPI voice widget in Streamlit.
    """
    st.markdown("### 🎙️ Voice Assistant")
    st.markdown("Click the button below to start a voice conversation with our AI assistant.")
    
    # Check if VAPI credentials exist
    if "VAPI_API_KEY" not in st.secrets:
        st.error("❌ VAPI_API_KEY not found in secrets.toml")
        st.info("Add your VAPI API key to .streamlit/secrets.toml")
        return
    
    # Initialize VAPI agent
    if "vapi_agent" not in st.session_state:
        st.session_state.vapi_agent = VAPIAgent(
            api_key=st.secrets["VAPI_API_KEY"],
            assistant_id=st.secrets.get("VAPI_ASSISTANT_ID")
        )
    
    # Create assistant if needed
    if not st.session_state.vapi_agent.assistant_id:
        with st.spinner("Setting up voice assistant..."):
            try:
                assistant_id = st.session_state.vapi_agent.create_assistant()
                st.success(f"✓ Voice assistant ready! ID: {assistant_id}")
            except Exception as e:
                st.error(f"Failed to setup assistant: {str(e)}")
                return
    
    # Voice interaction button
    if st.button("🎙️ Start Voice Search", key="start_voice"):
        try:
            web_call_url = st.session_state.vapi_agent.get_web_call_token()
            
            # Embed VAPI widget using iframe
            vapi_widget = f"""
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; text-align: center; background: #f9f9f9;">
                <p style="font-size: 18px; color: #333; margin-bottom: 15px;">
                    <strong>Voice Assistant Active</strong>
                </p>
                <iframe 
                    src="{web_call_url}" 
                    width="100%" 
                    height="400" 
                    frameborder="0"
                    allow="microphone"
                ></iframe>
                <p style="font-size: 14px; color: #666; margin-top: 15px;">
                    Speak naturally and describe your ideal NYC neighborhood
                </p>
            </div>
            """
            
            st.markdown(vapi_widget, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Failed to start voice call: {str(e)}")
    
    # Instructions
    with st.expander("ℹ️ How to use voice search"):
        st.markdown("""
        **Tips for best results:**
        
        1. **Be specific**: "I want a quiet neighborhood with lots of parks, budget under $200"
        2. **Mention priorities**: "Safety and public transport are very important to me"
        3. **Express negatives**: "I don't care about nightlife"
        4. **Ask questions**: "What's a good area for families?"
        
        **Example conversation:**
        - 🎙️ You: "I'm looking for a safe, quiet area with good restaurants"
        - 🤖 Assistant: "What's your budget per night?"
        - 🎙️ You: "Around $150"
        - 🤖 Assistant: "Great! How important is being near parks?"
        - 🎙️ You: "Very important, I love nature"
        
        The assistant will automatically update your search filters!
        """)


def handle_vapi_callback(function_name: str, parameters: Dict[str, Any]):
    """
    Handle function calls from VAPI assistant.
    
    Args:
        function_name: Name of the function called
        parameters: Function parameters from VAPI
    """
    if function_name == "update_filters":
        # Update session state with new filter values
        for key, value in parameters.items():
            if key in st.session_state:
                st.session_state[key] = value
                st.success(f"Updated {key} to {value}")
        
        # Trigger rerun to update the map
        st.rerun()


# Alternative: Simple Audio Recording Widget (if VAPI is too complex)
def render_simple_audio_widget():
    """
    Simpler alternative: Audio recording widget with OpenAI Whisper transcription.
    """
    st.markdown("### 🎙️ Voice Search (Simple)")
    
    # Audio recording
    st.markdown("Click 'Record' and describe your ideal neighborhood:")
    
    # Use streamlit-webrtc for audio recording
    audio_component = st.audio_input("Record your voice")
    
    if audio_component:
        if st.button("🔍 Analyze Voice Input"):
            with st.spinner("Transcribing and analyzing..."):
                try:
                    # Transcribe audio using OpenAI Whisper
                    import openai
                    openai.api_key = st.secrets["OPENAI_API_KEY"]
                    
                    # Save audio temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_component.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Transcribe
                    with open(tmp_file_path, "rb") as audio_file:
                        transcript = openai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    
                    transcribed_text = transcript.text
                    st.info(f"📝 You said: {transcribed_text}")
                    
                    # Use existing OpenAI analysis from app.py
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": """
                            Extract user preferences from their voice input and return JSON with filter values.
                            Return only JSON with keys: max_price, w_nightlife, w_culture, w_restaurants, 
                            w_green, w_shopping, w_safety, w_quiet, w_clean, w_mobility (all 0-10 scale).
                            """},
                            {"role": "user", "content": transcribed_text}
                        ]
                    )
                    
                    updates = json.loads(response.choices[0].message.content)
                    
                    # Update filters
                    for key, value in updates.items():
                        if key in st.session_state:
                            st.session_state[key] = value
                    
                    st.success("✓ Filters updated from voice input!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    st.title("VAPI Voice Agent Test")
    render_vapi_widget()