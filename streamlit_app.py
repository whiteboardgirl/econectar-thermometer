import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import requests
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# ========================
# Configuration & Constants
# ========================

@dataclass
class ThermalConfig:
    """Configuration constants for thermal calculations."""
    AIR_FILM_RESISTANCE_OUTSIDE: float = 0.04  # m²K/W
    BEE_METABOLIC_HEAT: float = 0.0040  # Watts per bee
    IDEAL_HIVE_TEMPERATURE: float = 35.0  # °C

# ========================
# Data Models
# ========================

@dataclass
class Box:
    """Represents a single hive box."""
    id: int
    width: float
    height: float
    cooling_effect: float

# ========================
# External API Calls
# ========================

@st.cache_data(show_spinner=False)
def get_temperature_from_coordinates(lat: float, lon: float) -> Optional[float]:
    """
    Retrieve the current temperature from the Open-Meteo API for the provided coordinates.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "current_weather" in data:
            return data["current_weather"]["temperature"]
        else:
            st.error("No current weather data found in the response.")
            return None
    except requests.HTTPError as e:
        st.error(f"Error fetching weather data: {str(e)}. Status code: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching weather data: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def get_altitude_from_coordinates(lat: float, lon: float) -> Optional[float]:
    """
    Retrieve the altitude (elevation in meters) for the provided coordinates using the Open-Elevation API.
    """
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # The API returns a list of results; we take the first one.
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["elevation"]
        else:
            st.error("No elevation data found for the provided coordinates.")
            return None
    except requests.HTTPError as e:
        st.error(f"Error fetching elevation data: {str(e)}. Status code: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching elevation data: {str(e)}")
        return None

# ========================
# Thermal Calculations
# ========================

def calculate_oxygen_factor(altitude_m: float) -> float:
    """Calculate oxygen factor based on altitude."""
    H = 7400  # Scale height for Earth's atmosphere (m)
    pressure_ratio = math.exp(-altitude_m / H)
    return max(0.6, pressure_ratio)

def calculate_box_surface_area(width_cm: float, height_cm: float) -> float:
    """
    Calculate the total surface area for a hexagonal box in square meters.
    Assumes a hexagon-based design where the width defines the distance between parallel sides.
    """
    width_m, height_m = width_cm / 100, height_cm / 100
    side_length = width_m / math.sqrt(3)
    hexagon_area = (3 * math.sqrt(3) / 2) * (side_length ** 2)
    sides_area = 6 * side_length * height_m
    return (2 * hexagon_area) + sides_area

def calculate_relative_humidity_at_altitude(base_humidity: float, base_temp: float, altitude: float) -> float:
    """
    Calculate relative humidity at altitude based on temperature and pressure changes.
    Uses the Magnus-Tetens formula for vapor pressure.
    """
    A = 17.27
    B = 237.7  # °C
    LAPSE_RATE = 6.5  # °C per 1000m (used for humidity only)
    temp_at_altitude = base_temp - (altitude * LAPSE_RATE / 1000)
    
    def vapor_pressure(T: float) -> float:
        return 0.611 * math.exp((A * T) / (T + B))  # in kPa
    
    base_es = vapor_pressure(base_temp)
    altitude_es = vapor_pressure(temp_at_altitude)
    pressure_ratio = math.exp(-altitude / 7400)
    base_ea = base_es * (base_humidity / 100)
    altitude_ea = base_ea * pressure_ratio
    new_rh = (altitude_ea / altitude_es) * 100
    
    altitude_factor = (min(1.2, 1 + (altitude / 2000) * 0.2)
                       if altitude <= 2000 else 1.2 * (1 - (altitude - 2000) / 6000 * 0.3))
    
    return min(100, max(0, new_rh * altitude_factor))

def adjust_temperature_for_conditions(
    base_temp: float,
    altitude: float,
    is_daytime: bool,
    rain_intensity: float = 0.0,
    wind_speed: float = 0.0,
    humidity: float = 50.0,
    apply_altitude_temp_correction: bool = True,
    lapse_rate: float = 6.5,
    debug: bool = False
) -> float:
    """
    Adjust temperature based on environmental conditions.
    
    If apply_altitude_temp_correction is True, subtract an altitude-dependent lapse rate.
    The lapse_rate (°C per 1000 m) is adjustable via a slider.
    """
    if debug:
        st.write(f"[DEBUG] Base temp: {base_temp}, Altitude: {altitude}, Daytime: {is_daytime}, "
                 f"Rain: {rain_intensity}, Wind: {wind_speed}, Humidity: {humidity}, Lapse rate: {lapse_rate}")
    
    altitude_humidity = calculate_relative_humidity_at_altitude(humidity, base_temp, altitude)
    
    if apply_altitude_temp_correction:
        altitude_adjusted_temp = base_temp - (altitude * lapse_rate / 1000)
    else:
        altitude_adjusted_temp = base_temp  # Use the GPS temperature as is
    
    DAY_NIGHT_DIFFERENCE = 8.0  # °C difference between day and night
    time_adjusted_temp = altitude_adjusted_temp + (DAY_NIGHT_DIFFERENCE / 2 if is_daytime else -DAY_NIGHT_DIFFERENCE / 2)
    
    if time_adjusted_temp > 25 and altitude_humidity > 40:
        humidity_adjustment = (altitude_humidity - 40) / 100 * (time_adjusted_temp - 25) * 0.5
    else:
        humidity_adjustment = 0
    
    RAIN_COOLING_EFFECT = 5.0  # Maximum cooling effect of heavy rain
    rain_adjustment = -RAIN_COOLING_EFFECT * rain_intensity * (1 - altitude_humidity / 200)
    
    if time_adjusted_temp <= 10 and wind_speed > 1.3:
        wind_chill = (13.12 + 0.6215 * time_adjusted_temp - 11.37 * (wind_speed ** 0.16) +
                      0.3965 * time_adjusted_temp * (wind_speed ** 0.16))
        wind_adjustment = wind_chill - time_adjusted_temp
    else:
        wind_adjustment = 0
    
    final_temp = time_adjusted_temp + humidity_adjustment + rain_adjustment + wind_adjustment
    if debug:
        st.write(f"[DEBUG] Final adjusted temp: {final_temp}")
    return final_temp

def calculate_hive_temperature(
    params: Dict[str, float],
    boxes: List[Box],
    ambient_temp_c: float,
    is_daytime: bool,
    altitude: float,
    rain_intensity: float = 0.0,
    wind_speed: float = 0.0,
    humidity: float = 50.0,
    apply_altitude_temp_correction: bool = True,
    lapse_rate: float = 6.5,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Calculate hive temperature taking into account environmental factors and hive configuration.
    If the "bypass" option is active, the hive base temperature is set equal to the GPS ambient temperature.
    """
    if debug:
        st.write("[DEBUG] Input params:", params)
        st.write(f"[DEBUG] Ambient temp: {ambient_temp_c}, Daytime: {is_daytime}, Altitude: {altitude}")
    
    # Use altitude for oxygen factor and humidity calculations even if we bypass temperature adjustment.
    oxygen_factor = calculate_oxygen_factor(altitude)
    
    # Check if we are bypassing hive adjustments
    if params.get("bypass_hive", False):
        adjusted_ambient_temp = ambient_temp_c
    else:
        adjusted_ambient_temp = adjust_temperature_for_conditions(
            ambient_temp_c, altitude, is_daytime, rain_intensity, wind_speed, humidity,
            apply_altitude_temp_correction=apply_altitude_temp_correction,
            lapse_rate=lapse_rate,
            debug=debug
        )
    
    if debug:
        st.write(f"[DEBUG] Adjusted ambient temp: {adjusted_ambient_temp}")
    
    altitude_humidity = calculate_relative_humidity_at_altitude(humidity, ambient_temp_c, altitude)
    
    # Adjust simulation parameters based on time of day (skip if bypassing)
    if not params.get("bypass_hive", False):
        if is_daytime:
            params['ideal_hive_temperature'] += 1.0
            params['bee_metabolic_heat'] *= 1.2
            params['air_film_resistance_outside'] *= 0.9
        else:
            params['ideal_hive_temperature'] -= 0.5
            params['bee_metabolic_heat'] *= 0.8
            params['air_film_resistance_outside'] *= 1.2

    if debug:
        st.write("[DEBUG] Time-adjusted params:", params)
    
    # When bypassing, skip further adjustments and use the ambient temperature directly.
    if params.get("bypass_hive", False):
        estimated_temp_c = ambient_temp_c
    else:
        if altitude_humidity > 70:
            evaporative_cooling_factor = 1 - ((altitude_humidity - 70) / 100)
            params['bee_metabolic_heat'] *= (1 + (1 - evaporative_cooling_factor) * 0.2)
    
        if rain_intensity > 0:
            params['bee_metabolic_heat'] *= (1 + rain_intensity * 0.3)
            params['air_film_resistance_outside'] *= (1 - rain_intensity * 0.2)
    
        if wind_speed > 0:
            wind_factor = max(0.5, 1 - (wind_speed / 20))
            params['air_film_resistance_outside'] *= wind_factor
            params['bee_metabolic_heat'] *= (1 + (wind_speed / 20) * 0.4)
    
        calculated_colony_size = 2000 * (params['colony_size'] / 100)
        colony_metabolic_heat = calculated_colony_size * params['bee_metabolic_heat'] * oxygen_factor
    
        total_volume = sum(
            (3 * math.sqrt(3) / 2) * ((box.width / (100 * math.sqrt(3))) ** 2) * (box.height / 100)
            for box in boxes
        )
        total_surface_area = sum(calculate_box_surface_area(box.width, box.height) for box in boxes)
    
        wood_resistance = (params['wood_thickness'] / 100) / params['wood_thermal_conductivity']
        total_resistance = wood_resistance + params['air_film_resistance_outside']
    
        if adjusted_ambient_temp >= params['ideal_hive_temperature']:
            cooling_effort = min(1.0, (adjusted_ambient_temp - params['ideal_hive_temperature']) / 10)
            temp_decrease = (2.0 if is_daytime else 1.0) * cooling_effort
            estimated_temp_c = max(params['ideal_hive_temperature'], adjusted_ambient_temp - temp_decrease)
        else:
            heat_contribution = min(
                params['ideal_hive_temperature'] - adjusted_ambient_temp,
                (colony_metabolic_heat * total_resistance) / total_surface_area
            )
            heat_contribution *= 0.9 if not is_daytime else 1.0
            estimated_temp_c = adjusted_ambient_temp + heat_contribution
    
        estimated_temp_c = min(50, max(0, estimated_temp_c))
    
    if debug:
        st.write(f"[DEBUG] Estimated hive base temperature: {estimated_temp_c}")
    
    if params.get("bypass_hive", False):
        box_temperatures = [ambient_temp_c for _ in boxes]
    else:
        box_temperatures = [
            max(0, min(50, estimated_temp_c - box.cooling_effect))
            for box in boxes
        ]
    
    if debug:
        st.write("[DEBUG] Box temperatures:", box_temperatures)
    
    heat_transfer = (total_surface_area * abs(estimated_temp_c - adjusted_ambient_temp)) / total_resistance / 1000 if not params.get("bypass_hive", False) else 0
    
    return {
        'calculated_colony_size': calculated_colony_size if not params.get("bypass_hive", False) else 0,
        'colony_metabolic_heat': (colony_metabolic_heat / 1000) if not params.get("bypass_hive", False) else 0,
        'base_temperature': estimated_temp_c,
        'box_temperatures': box_temperatures,
        'total_volume': total_volume if not params.get("bypass_hive", False) else 0,
        'total_surface_area': total_surface_area if not params.get("bypass_hive", False) else 0,
        'thermal_resistance': total_resistance if not params.get("bypass_hive", False) else 0,
        'ambient_temperature': adjusted_ambient_temp,
        'oxygen_factor': oxygen_factor,
        'altitude_humidity': altitude_humidity,
        'heat_transfer': heat_transfer
    }

# ========================
# Visualization Functions
# ========================

def create_temperature_chart(box_temperatures: List[float]) -> None:
    """Create a bar chart showing temperature distribution across hive boxes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([f'Box {i+1}' for i in range(len(box_temperatures))], box_temperatures, color='skyblue')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Distribution Across Hive Boxes')
    st.pyplot(fig)

def create_altitude_chart(altitude_range: np.ndarray, temperatures: List[float]) -> None:
    """Plot hive temperature versus altitude."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(altitude_range, temperatures, marker='o', linestyle='-')
    ax.set_title("Hive Temperature vs. Altitude")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Hive Temperature (°C)")
    ax.grid(True)
    st.pyplot(fig)

# ========================
# State Initialization and UI Components
# ========================

def initialize_state() -> None:
    """Initialize the application state if not already set."""
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

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar inputs and return the parameters."""
    st.sidebar.title("🐝 Configuration")
    
    debug = st.sidebar.checkbox("Enable Debug Output", value=False)
    
    lapse_rate = st.sidebar.slider("Lapse Rate (°C per 1000m)", 0.0, 10.0, 6.5, step=0.1)
    
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
        'bee_metabolic_heat': ThermalConfig.BEE_METABOLIC_HEAT,
        'lapse_rate': lapse_rate
    }
    
    st.sidebar.write("Current params:", params)
    
    # Default is False so that the ambient temperature remains as measured by GPS.
    apply_altitude_correction = st.sidebar.checkbox(
        "Apply altitude correction to ambient temp", value=False,
        help="Disable if GPS temperature is already altitude-adjusted."
    )
    params['apply_altitude_correction'] = apply_altitude_correction
    
    bypass_hive = st.sidebar.checkbox(
        "Bypass hive calculation adjustments", value=False,
        help="If enabled, hive temperature will equal the GPS ambient temperature."
    )
    params['bypass_hive'] = bypass_hive
    
    return params, debug

def render_box_controls() -> None:
    """Render controls for modifying each hive box's parameters."""
    st.subheader("📦 Box Configuration")
    for i, box in enumerate(st.session_state.boxes):
        with st.expander(f"Box {box.id}", expanded=True):
            box.width = st.slider(f"Width for Box {box.id} (cm)", 10, 50, int(box.width))
            box.height = st.slider(f"Height for Box {box.id} (cm)", 5, 20, int(box.height))
            box.cooling_effect = st.number_input(
                f"Cooling Effect for Box {box.id} (°C)", 
                0.0, 20.0, 
                float(box.cooling_effect), 
                0.5, 
                key=f"cooling_effect_{i}"
            )

# ========================
# Main Application
# ========================

def main():
    """Main application entry point."""
    initialize_state()
    
    st.set_page_config(
        page_title="Hive Thermal Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🐝 Hive Thermal Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        params, debug = render_sidebar()
        
        gps_coordinates = st.text_input("Enter GPS Coordinates (lat, lon)", "4.6097, -74.0817")
        try:
            lat, lon = map(float, gps_coordinates.split(','))
            ambient_temperature = get_temperature_from_coordinates(lat, lon)
            if ambient_temperature is None:
                ambient_temperature = 25.0
            
            # Retrieve altitude automatically
            altitude = get_altitude_from_coordinates(lat, lon)
            if altitude is None:
                st.warning("Could not retrieve altitude automatically. Please use the slider below.")
                altitude = st.slider("Simulated Altitude (meters)", 0, 3800, 0, 100)
            else:
                st.info(f"Retrieved altitude: {altitude:.1f} m")
        except ValueError:
            st.error("Please enter valid coordinates in the format 'lat, lon'")
            ambient_temperature = 25.0
            altitude = st.slider("Simulated Altitude (meters)", 0, 3800, 0, 100)
        
        st.subheader("🌍 Environmental Conditions")
        is_daytime = st.radio("Time of Day", ['Day', 'Night'], index=0) == 'Day'
        
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            humidity = st.slider("Relative Humidity (%)", 0, 100, 50,
                                 help="Base humidity at current location")
            altitude_humidity = calculate_relative_humidity_at_altitude(humidity, ambient_temperature, altitude)
            st.info(f"Humidity at altitude: {altitude_humidity:.1f}%")
        with col1b:
            rain_intensity = st.slider("Rain Intensity", 0.0, 1.0, 0.0, 
                                       help="0 = No rain, 1 = Heavy rain")
        with col1c:
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 0.0,
                                   help="Typical range: Light breeze (2-5), Strong wind (10-15)")
        
        st.write(f"Current Ambient Temperature: {ambient_temperature}°C")
        
        render_box_controls()
        
    with col2:
        results = calculate_hive_temperature(
            params.copy(),  # Use a copy so the original params remain unchanged
            st.session_state.boxes, 
            ambient_temperature, 
            is_daytime, 
            altitude,
            rain_intensity,
            wind_speed,
            humidity,
            apply_altitude_temp_correction=params.get('apply_altitude_correction', False),
            lapse_rate=params.get('lapse_rate', 6.5),
            debug=debug
        )
        
        if debug and st.checkbox("Show Detailed Debug Info"):
            st.write("[DEBUG] Calculation Results:", results)
        
        st.subheader("📊 Analysis Results")
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Base Hive Temperature", f"{results['base_temperature']:.1f}°C")
            st.metric("Ambient Temperature", f"{results['ambient_temperature']:.1f}°C")
        with col2b:
            st.metric("Colony Size", f"{int(results['calculated_colony_size']):,} bees")
            st.metric("Metabolic Heat", f"{results['colony_metabolic_heat']:.3f} kW")
        
        st.subheader("📊 Box Temperatures")
        for i, temp in enumerate(results['box_temperatures']):
            st.markdown(f"**Box {i+1}:** {temp:.1f}°C")
            st.progress(max(0.0, min(1.0, temp / 50)))
        
        st.subheader("📈 Temperature Distribution")
        create_temperature_chart(results['box_temperatures'])
        
        st.subheader("📈 Altitude Effect")
        altitude_range = np.arange(0, 4000, 100)
        altitude_temps = []
        with st.spinner('Calculating altitude effects...'):
            for alt in altitude_range:
                alt_results = calculate_hive_temperature(
                    params.copy(),
                    st.session_state.boxes, 
                    ambient_temperature, 
                    is_daytime, 
                    alt,
                    rain_intensity,
                    wind_speed,
                    humidity,
                    apply_altitude_temp_correction=params.get('apply_altitude_correction', False),
                    lapse_rate=params.get('lapse_rate', 6.5),
                    debug=False
                )
                altitude_temps.append(alt_results['base_temperature'])
        create_altitude_chart(altitude_range, altitude_temps)
        
        st.metric("Total Hive Volume", f"{results['total_volume']:.2f} m³")
        st.metric("Total Surface Area", f"{results['total_surface_area']:.2f} m²")
        st.metric("Heat Transfer", f"{results['heat_transfer']:.3f} kW")
    
    if 'last_params' not in st.session_state or st.session_state.last_params != params:
        st.session_state.last_params = params
        st.experimental_rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please check the input parameters or try refreshing the page.")
