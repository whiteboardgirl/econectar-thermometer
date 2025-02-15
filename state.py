# state.py
import streamlit as st
from typing import Dict, Any, List
from config import Box

def initialize_state() -> None:
    """Initialize application state with defaults."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.boxes = [
            Box(id=i+1, width=22, height=9, cooling_effect=ce)
            for i, ce in enumerate([2, 0, 0, 8])
        ]
        st.session_state.saved_state = {}
        st.session_state.realtime = False
        st.session_state.alerts = []
        st.session_state.notes = ""

def save_current_state(params: Dict[str, Any], boxes: List[Box], 
                      is_daytime: bool, altitude: float, 
                      gps_coordinates: str) -> None:
    """Save current state."""
    st.session_state.saved_state = {
        'params': params,
        'boxes': boxes,
        'is_daytime': is_daytime,
        'altitude': altitude,
        'gps_coordinates': gps_coordinates
    }
    st.success("State saved successfully.")
