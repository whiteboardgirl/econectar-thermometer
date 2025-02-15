import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class ThermalConfig:
    """Configuration constants for thermal calculations."""
    KELVIN_CONVERSION: float = 273.15
    AIR_FILM_RESISTANCE_OUTSIDE: float = 0.04  # m²K/W
    BEE_METABOLIC_HEAT: float = 0.0040  # Watts per bee
    IDEAL_HIVE_TEMPERATURE: float = 35.0  # °C

@dataclass
class Box:
    """Represents a single hive box."""
    id: int
    width: float
    height: float
    cooling_effect: float

# Thermal calculations
def calculate_oxygen_factor(altitude_m: float) -> float:
    """Calculate oxygen factor based on altitude."""
    P0 = 1013.25  # Standard atmospheric pressure at sea level (hPa)
    H = 7400  # Scale height for Earth's atmosphere (m)
    pressure_ratio = math.exp(-altitude_m / H)
    return max(0.6, pressure_ratio)

def calculate_box_surface_area(width_cm: float, height_cm: float) -> float:
    """Calculate surface area for a hexagonal box in square meters."""
    width_m, height_m = width_cm / 100, height_cm / 100
    side_length = width_m / math.sqrt(3)
    hexagon_area = (3 * math.sqrt(3) / 2) * (side_length ** 2)
    sides_area = 6 * side_length * height_m
    return (2 * hexagon_area) + sides_area

def calculate_relative_humidity_at_altitude(base_humidity: float, base_temp: float, 
                                            altitude: float) -> float:
    """
    Calculate relative humidity at altitude based on temperature and pressure changes.
    
    The Magnus-Tetens formula is used for vapor pressure calculations.
    Humidity typically increases with altitude until about 2000m, then decreases.
    """
    A = 17.27
    B = 237.7  # °C
    
    LAPSE_RATE = 6.5  # °C per 1000m
    temp_at_altitude = base_temp - (altitude * LAPSE_RATE / 1000)
    
    def vapor_pressure(T):
        return 0.611 * math.exp((A * T) / (T + B))  # kPa
    
    base_es = vapor_pressure(base_temp)
    altitude_es = vapor_pressure(temp_at_altitude)
    
    pressure_ratio = math.exp(-altitude / 7400)  # scale height of 7400m
    
    base_ea = base_es * (base_humidity / 100)
    altitude_ea = base_ea * pressure_ratio
    
    new_rh = (altitude_ea / altitude_es) * 100
    
    altitude_factor = min(1.2, 1 + (altitude / 2000) * 0.2) if altitude <= 2000 else \
                     1.2 * (1 - (altitude - 2000) / 6000 * 0.3)
    
    return min(100, max(0, new_rh * altitude_factor))

def adjust_temperature_for_conditions(base_temp: float, altitude: float, is_daytime: bool,
                                      rain_intensity: float = 0.0, wind_speed: float = 0.0,
                                      humidity: float = 50.0) -> float:
    """
    Adjust temperature based on environmental conditions including humidity.
    """
    altitude_humidity = calculate_relative_humidity_at_altitude(humidity, base_temp, altitude)
    
    LAPSE_RATE = 6.5  # °C per 1000m
    altitude_adjusted_temp = base_temp - (altitude * LAPSE_RATE / 1000)
    
    DAY_NIGHT_DIFFERENCE = 8.0  # °C difference between day and night
    time_adjusted_temp = altitude_adjusted_temp + (DAY_NIGHT_DIFFERENCE / 2 if is_daytime else -DAY_NIGHT_DIFFERENCE / 2)
    
    if time_adjusted_temp > 25 and altitude_humidity > 40:
        humidity_adjustment = (altitude_humidity - 40) / 100 * (time_adjusted_temp - 25) * 0.5
    else:
        humidity_adjustment = 0
    
    RAIN_COOLING_EFFECT = 5.0  # Maximum temperature reduction from heavy rain
    rain_adjustment = -RAIN_COOLING_EFFECT * rain_intensity * (1 - altitude_humidity / 200)
    
    if time_adjusted_temp <= 10 and wind_speed > 1.3:
        wind_chill = 13.12 + 0.6215 * time_adjusted_temp - 11.37 * (wind_speed ** 0.16) + \
                    0.3965 * time_adjusted_temp * (wind_speed ** 0.16)
        wind_adjustment = wind_chill - time_adjusted_temp
    else:
        wind_adjustment = 0
    
    final_temp = time_adjusted_temp + humidity_adjustment + rain_adjustment + wind_adjustment
    return final_temp

def calculate_hive_temperature(params: Dict[str, float], boxes: List[Box], 
                               ambient_temp_c: float, is_daytime: bool, 
                               altitude: float, rain_intensity: float = 0.0,
                               wind_speed: float = 0.0, humidity: float = 50.0) -> Dict[str, Any]:
    """Calculate hive temperature with all environmental factors."""
    config = ThermalConfig()
    oxygen_factor = calculate_oxygen_factor(altitude)
    
    adjusted_ambient_temp = adjust_temperature_for_conditions(
        ambient_temp_c, altitude, is_daytime, rain_intensity, wind_speed, humidity
    )
    
    altitude_humidity = calculate_relative_humidity_at_altitude(humidity, ambient_temp_c, altitude)
    
    # Adjust parameters for time of day
    if is_daytime:
        params['ideal_hive_temperature'] += 1.0
        params['bee_metabolic_heat'] *= 1.2
        params['air_film_resistance_outside'] *= 0.9
    else:
        params['ideal_hive_temperature'] -= 0.5
        params['bee_metabolic_heat'] *= 0.8
        params['air_film_resistance_outside'] *= 1.2
    
    # Humidity effects on bee thermoregulation
    if altitude_humidity > 70:
        evaporative_cooling_factor = 1 - ((altitude_humidity - 70) / 100)
        params['bee_metabolic_heat'] *= (1 + (1 - evaporative_cooling_factor) * 0.2)
    
    # Rain effects
    if rain_intensity > 0:
        params['bee_metabolic_heat'] *= (1 + rain_intensity * 0.3)
        params['air_film_resistance_outside'] *= (1 - rain_intensity * 0.2)
    
    # Wind effects
    if wind_speed > 0:
        wind_factor = max(0.5, 1 - (wind_speed / 20))
        params['air_film_resistance_outside'] *= wind_factor
        params['bee_metabolic_heat'] *= (1 + (wind_speed / 20) * 0.4)
    
    # Colony calculations
    calculated_colony_size = 50000 * (params['colony_size'] / 100)
    colony_metabolic_heat = calculated_colony_size * params['bee_metabolic_heat'] * oxygen_factor

    # Volume and surface calculations
    total_volume = sum(
        (3 * math.sqrt(3) / 2) * ((box.width / (100 * math.sqrt(3))) ** 2) * (box.height / 100)
        for box in boxes
    )
    
    total_surface_area = sum(calculate_box_surface_area(box.width, box.height) for box in boxes)

    # Thermal resistance
    wood_resistance = (params['wood_thickness'] / 100) / params['wood_thermal_conductivity']
    total_resistance = wood_resistance + params['air_film_resistance_outside']

    # Temperature calculations
    if adjusted_ambient_temp >= params['ideal_hive_temperature']:
        cooling_effort = min(1.0, (adjusted_ambient_temp - params['ideal_hive_temperature']) / 10)
        temp_decrease = 2.0 * cooling_effort if is_daytime else 1.0 * cooling_effort
        estimated_temp_c = max(params['ideal_hive_temperature'], adjusted_ambient_temp - temp_decrease)
    else:
        heat_contribution = min(
            params['ideal_hive_temperature'] - adjusted_ambient_temp,
            (colony_metabolic_heat * total_resistance) / total_surface_area
        )
        heat_contribution *= 0.9 if not is_daytime else 1.0
        estimated_temp_c = adjusted_ambient_temp + heat_contribution

    # Final adjustments
    estimated_temp_c = min(50, max(0, estimated_temp_c))

    # Calculate box temperatures
    box_temperatures = [
        max(0, min(50, estimated_temp_c - box.cooling_effect))
        for box in boxes
    ]

    return {
        'calculated_colony_size': calculated_colony_size,
        'colony_metabolic_heat': colony_metabolic_heat / 1000,  # Convert to kW
        'base_temperature': estimated_temp_c,
        'box_temperatures': box_temperatures,
        'total_volume': total_volume,
        'total_surface_area': total_surface_area,
        'thermal_resistance': total_resistance,
        'ambient_temperature': adjusted_ambient_temp,
        'oxygen_factor': oxygen_factor,
        'altitude_humidity': altitude_humidity,
        'heat_transfer': (total_surface_area * abs(estimated_temp_c - adjusted_ambient_temp)) / total_resistance / 1000
    }

# Weather API integration
def get_temperature_from_coordinates(lat: float, lon: float) -> Optional[float]:
    """Retrieve temperature data from coordinates."""
    url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['current_weather']['temperature']
    except requests.HTTPError as e:
        st.error(f"Error fetching weather data: {str(e)}. Status code: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching weather data: {str(e)}")
        return None

# Visualization functions
def create_temperature_chart(box_temperatures: List[float]) -> None:
    """Create temperature distribution chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([f'Box {i+1}' for i in range(len(box_temperatures))], box_temperatures)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Distribution Across Hive Boxes')
    st.pyplot(fig)

def create_altitude_chart(altitude_range: np.ndarray, temperatures: List[float]) -> None:
    """Create altitude effect visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(altitude_range, temperatures)
    ax.set_title("Hive Temperature vs. Altitude")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Hive Temperature (°C)")
    ax.grid(True)
    st.pyplot(fig)

# State initialization
def initialize_state() -> None:
    """Initialize application state."""
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

# UI Components
def render_sidebar() -> Dict[str, Any]:
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
    """Main application."""
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
        # Parameters
        params = render_sidebar()
        
        # Location inputs
        gps_coordinates = st.text_input("Enter GPS Coordinates (lat, lon)", "4.6097, -74.0817")
        try:
            lat, lon = map(float, gps_coordinates.split(','))
            ambient_temperature = get_temperature_from_coordinates(lat, lon)
            if ambient_temperature is None:
                ambient_temperature = 25.0
        except ValueError:
            st.error("Please enter valid coordinates in the format 'lat, lon'")
            ambient_temperature = 25.0
        
        # Environmental conditions
        st.subheader("🌍 Environmental Conditions")
        is_daytime = st.radio("Time of Day", ['Day', 'Night'], index=0) == 'Day'
        altitude = st.slider("Altitude (meters)", 0, 3800, 0, 100)
        
        # Weather controls in three columns
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            humidity = st.slider("Relative Humidity (%)", 0, 100, 50,
                               help="Base humidity at current location")
            altitude_humidity = calculate_relative_humidity_at_altitude(
                humidity, ambient_temperature, altitude
            )
            st.info(f"Humidity at altitude: {altitude_humidity:.1f}%")
            
        with col1b:
            rain_intensity = st.slider("Rain Intensity", 0.0, 1.0, 0.0, 
                                     help="0 = No rain, 1 = Heavy rain")
        with col1c:
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 0.0,
                                 help="Typical range: Light breeze (2-5), Strong wind (10-15)")
        
        st.write(f"Current Ambient Temperature: {ambient_temperature}°C")
        
        # Box configuration
        render_box_controls()
        
    with col2:
        # Calculate results with new weather parameters
        results = calculate_hive_temperature(
            params, 
            st.session_state.boxes, 
            ambient_temperature, 
            is_daytime, 
            altitude,
            rain_intensity,
            wind_speed,
            humidity
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
        st.subheader("📈 Temperature Distribution")
        create_temperature_chart(results['box_temperatures'])
        
        # Altitude effect
        st.subheader("📈 Altitude Effect")
        altitude_temps = []
        altitude_range = np.arange(0, 4000, 100)
        
        # Calculate temperatures for different altitudes
        with st.spinner('Calculating altitude effects...'):
            for alt in altitude_range:
                alt_results = calculate_hive_temperature(
                    params.copy(),  # Use a copy to prevent parameter modification
                    st.session_state.boxes, 
                    ambient_temperature, 
                    is_daytime, 
                    alt
                )
                altitude_temps.append(alt_results['base_temperature'])
        
        create_altitude_chart(altitude_range, altitude_temps)
        
        # Additional metrics
        st.metric("Total Hive Volume", f"{results['total_volume']:.2f} m³")
        st.metric("Total Surface Area", f"{results['total_surface_area']:.2f} m²")
        st.metric("Heat Transfer", f"{results['heat_transfer']:.3f} kW")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please check the input parameters or try refreshing the page.")
