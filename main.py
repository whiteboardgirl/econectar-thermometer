# main.py
import streamlit as st
import numpy as np
from typing import Dict, Any
from config import ThermalConfig, Box
from thermal import calculate_hive_temperature
from weather import get_temperature_from_coordinates
from visualization import create_temperature_chart, create_altitude_chart
from state import initialize_state, save_current_state

def create_sidebar() -> Dict[str, Any]:
    """Create and handle sidebar inputs."""
    st.sidebar.title("🐝 Configuration")
    
    params = {
        'colony_size': st.sidebar.slider("Colony Size (%)", 0, 100, 50),
        'wood_thickness': st.sidebar.slider("Wood Thickness (cm)", 1.0, 5.0, 2.0),
        'wood_thermal_conductivity': st.sidebar.slider(
            "Thermal Conductivity (W/(m⋅K))", 
            0.1, 0.3, 0.13, 
            step=0.01
        ),
        'air_film_resistance_outside': ThermalConfig.AIR_FILM_RESISTANCE_OUTSIDE,
        'ideal_hive_temperature': ThermalConfig.IDEAL_HIVE_TEMPERATURE,
        'bee_metabolic_heat': ThermalConfig.BEE_METABOLIC_HEAT
    }
    
    return params

def render_box_controls() -> None:
    """Render box configuration controls."""
    st.subheader("📦 Box Configuration")
    for i, box in enumerate(st.session_state.boxes):
        with st.expander(f"Box {box.id}", expanded=True):
            box.width = st.slider(f"Width for Box {box.id} (cm)", 10, 50, int(box.width))
            box.height = st.slider(f"Height for Box {box.id} (cm)", 5, 20, int(box.height))
            box.cooling_effect = st.number_input(
                "Cooling Effect (°C)", 
                0.0, 20.0, 
                float(box.cooling_effect), 
                0.5, 
                key=f"cooling_effect_{i}"
            )

def main():
    """Main application entry point."""
    # Initialize state
    initialize_state()
    
    # Page setup
    st.set_page_config(
        page_title="Hive Thermal Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🐝 Hive Thermal Dashboard")
    st.markdown("---")
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Input parameters
        params = create_sidebar()
        
        # Location and environmental inputs
        gps_coordinates = st.text_input("Enter GPS Coordinates (lat, lon)", "4.6097, -74.0817")
        try:
            lat, lon = map(float, gps_coordinates.split(','))
            ambient_temperature = get_temperature_from_coordinates(lat, lon)
            if ambient_temperature is None:
                ambient_temperature = 25.0
        except ValueError:
            st.error("Please enter valid coordinates in the format 'lat, lon'")
            ambient_temperature = 25.0
        
        is_daytime = st.radio("Time of Day", ['Day', 'Night'], index=0) == 'Day'
        st.write(f"Current Ambient Temperature: {ambient_temperature}°C")
        
        altitude = st.slider("Altitude (meters)", 0, 3800, 0, 100)
        
        # Box configuration
        render_box_controls()
        
    with col2:
        # Calculate results
        results = calculate_hive_temperature(
            params, 
            st.session_state.boxes, 
            ambient_temperature, 
            is_daytime, 
            altitude
        )
        
        # Display metrics
        st.subheader("📊 Analysis Results")
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Base Hive Temperature", f"{results['base_temperature']:.1f}°C")
            st.metric("Ambient Temperature", f"{results['ambient_temperature']:.1f}°C")
        with col2b:
            st.metric("Colony Size", f"{int(results['calculated_colony_size']):,} bees")
            st.metric("Metabolic Heat", f"{results['colony_metabolic_heat']:.3f} kW")
        
        # Box temperatures
        st.subheader("📊 Box Temperatures")
        for i, temp in enumerate(results['box_temperatures']):
            st.markdown(f"**Box {i+1}:** {temp:.1f}°C")
            st.progress(max(0.0, min(1.0, temp / 50)))
        
        # Visualizations
        temp_chart = create_temperature_chart(results['box_temperatures'])
        st.markdown(f'<img src="data:image/png;base64,{temp_chart}"/>', unsafe_allow_html=True)
        
        # Altitude effect
        altitude_temps = []
        altitude_range = np.arange(0, 4000, 100)
        for alt in altitude_range:
            alt_results = calculate_hive_temperature(
                params, 
                st.session_state.boxes, 
                ambient_temperature, 
                is_daytime, 
                alt
            )
            altitude_temps.append(alt_results['base_temperature'])
        
        altitude_chart = create_altitude_chart(altitude_range, altitude_temps)
        st.markdown(f'<img src="data:image/png;base64,{altitude_chart}"/>', unsafe_allow_html=True)
        
        # Additional metrics
        st.metric("Total Hive Volume", f"{results['total_volume']:.2f} m³")
        st.metric("Total Surface Area", f"{results['total_surface_area']:.2f} m²")
        st.metric("Heat Transfer", f"{results['heat_transfer']:.3f} kW")
        
        # Save
