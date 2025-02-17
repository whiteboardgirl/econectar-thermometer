# Standard library imports
import datetime
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Third-party imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
import pytz
from timezonefinder import TimezoneFinder

# Data classes
@dataclass
class BeeSpecies:
    name: str
    metabolic_rate: float
    colony_size_factor: int
    ideal_temp: Tuple[float, float]
    humidity_range: Tuple[float, float]
    nest_conductivity: float
    max_cooling: float
    activity_profile: str

@dataclass
class HiveBox:
    id: int
    width: float
    height: float
    depth: float
    cooling_effect: float
    propolis_thickness: float = 1.5
    pdrc_coating: bool = False

    def adjusted_cooling(self, environment_factors):
        """Compute hive cooling effectiveness with or without CryoX coating."""
        base_cooling = self.cooling_effect
        if self.pdrc_coating:
            pdrc_performance = cryox_performance(**environment_factors) / 100
            return base_cooling * (1 + pdrc_performance)
        return base_cooling

# Configuration constants
SPECIES_CONFIG = {
    "Melipona": BeeSpecies(
        name="Melipona",
        metabolic_rate=0.0088,
        colony_size_factor=700,
        ideal_temp=(30.0, 33.0),
        humidity_range=(50.0, 70.0),
        nest_conductivity=0.09,
        max_cooling=1.5,
        activity_profile="Diurnal"
    ),
    "Scaptotrigona": BeeSpecies(
        name="Scaptotrigona",
        metabolic_rate=0.0105,
        colony_size_factor=1000,
        ideal_temp=(31.0, 35.0),
        humidity_range=(40.0, 70.0),
        nest_conductivity=0.11,
        max_cooling=1.8,
        activity_profile="Morning"
    ),

    "Trigona fulviventris": BeeSpecies(
        name="Trigona fulviventris",
        metabolic_rate=0.0095,
        colony_size_factor=800,
        ideal_temp=(32.0, 36.0),
        humidity_range=(45.0, 65.0),
        nest_conductivity=0.10,
        max_cooling=1.6,
        activity_profile="Diurnal"
    ),
    "Cephalotrigona femorata": BeeSpecies(
        name="Cephalotrigona femorata",
        metabolic_rate=0.0110,
        colony_size_factor=600,
        ideal_temp=(29.0, 33.0),
        humidity_range=(50.0, 70.0),
        nest_conductivity=0.095,
        max_cooling=1.55,
        activity_profile="Diurnal"
    ),
    "Melipona eburnea": BeeSpecies(
        name="Melipona eburnea",
        metabolic_rate=0.0090,
        colony_size_factor=750,
        ideal_temp=(30.5, 33.5),
        humidity_range=(50.0, 70.0),
        nest_conductivity=0.085,
        max_cooling=1.6,
        activity_profile="Diurnal"
    ),
    "Melipona compressipes": BeeSpecies(
        name="Melipona compressipes",
        metabolic_rate=0.0100,
        colony_size_factor=800,
        ideal_temp=(31.0, 34.0),
        humidity_range=(50.0, 68.0),
        nest_conductivity=0.088,
        max_cooling=1.7,
        activity_profile="Diurnal"
    ),
    "Trigona spinipes": BeeSpecies(
        name="Trigona spinipes",
        metabolic_rate=0.0092,
        colony_size_factor=850,
        ideal_temp=(31.0, 35.0),
        humidity_range=(45.0, 65.0),
        nest_conductivity=0.095,
        max_cooling=1.65,
        activity_profile="Diurnal"
    )
}

# API endpoints
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"

# Utility functions
def parse_gps_input(gps_str: str) -> Tuple[float, float] | None:
    try:
        lat, lon = map(float, gps_str.strip().split(','))
        return lat, lon
    except ValueError:
        return None

@st.cache_data(show_spinner=False)
def get_weather_data(lat: float, lon: float) -> Dict | None:
    """Fetches weather data from Open-Meteo API."""
    try:
        response = requests.get(OPEN_METEO_URL.format(lat=lat, lon=lon), timeout=5)
        response.raise_for_status()
        data = response.json()
        current = data.get("current_weather")
        if current:
            return {
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed")
            }
    except Exception as e:
        st.error(f"Failed to retrieve weather data: {e}")
    return None

@st.cache_data(show_spinner=False)
def get_altitude(lat: float, lon: float) -> float | None:
    """Fetches altitude data from Open Elevation API."""
    try:
        response = requests.get(OPEN_ELEVATION_URL.format(lat=lat, lon=lon), timeout=5)
        response.raise_for_status()
        data = response.json()
        results = data.get("results")
        if results and isinstance(results, list):
            return results[0].get("elevation")
    except Exception as e:
        st.error(f"Failed to retrieve altitude data: {e}")
    return None

def cryox_performance(cloud_cover, humidity, wind_speed, temperature, solar_angle, air_quality, uv_index):
    """Calculate CryoX PDRC surface effectiveness."""
    w_h, w_w, w_t, w_s, w_q, w_u = 0.35, 0.25, 0.15, 0.15, 0.10, 0.15

    P_h = max(0, 1 - humidity / 100)
    P_w = max(0, 1 - abs(wind_speed - 5) / 10)
    P_t = min(1, temperature / 100)
    P_s = min(1, solar_angle / 90)
    P_q = air_quality
    P_u = max(0, 1 - uv_index / 11)

    performance = (1 - cloud_cover**1.5) * (
        w_h * P_h + w_w * P_w + w_t * P_t + w_s * P_s + w_q * P_q + w_u * P_u
    ) * 100

    return max(0, performance)

# Simulation functions
@st.cache_data(ttl=1, show_spinner=False)
def simulate_hive_temperature(species: BeeSpecies, colony_size_pct: float, nest_thickness: float,
                            lid_thickness: float, boxes: List[HiveBox], ambient_temp: float,
                            is_daytime: bool, altitude: float, rain_intensity: float,
                            surface_area_exponent: float, lat: float, lon: float,
                            day_of_year: int, **environment_factors) -> Dict:
    """Simulates hive temperature with all environmental factors."""
    # Adjust ambient temperature for altitude, species behavior, and rain
    temp_adj = adjust_temperature(ambient_temp, altitude, species, is_daytime)
    temp_adj -= (rain_intensity * 3)  # Enhanced rain cooling effect

    # Heat contributions
    metabolic_heat = calculate_metabolic_heat(species, colony_size_pct, altitude)
    solar_heat_gain = calculate_solar_heat_gain(lat, lon, is_daytime, day_of_year)
    
    HONEY_HEAT_FACTOR = 0.25
    honey_heat = len(boxes) * HONEY_HEAT_FACTOR * metabolic_heat
    
    ENCLOSURE_HEAT_FACTOR = 1.5
    BASE_HEAT_RETENTION = 2.0
    
    total_heat = (metabolic_heat + solar_heat_gain + honey_heat) * ENCLOSURE_HEAT_FACTOR + BASE_HEAT_RETENTION

    # Thermal resistances calculation
    nest_resistance = (nest_thickness / 1000) / species.nest_conductivity
    propolis_resistance = sum(box.propolis_thickness * 0.02 for box in boxes)
    
    LID_CONDUCTIVITY = 0.012
    lid_resistance = (lid_thickness / 1000) / LID_CONDUCTIVITY
    LID_INSULATION_FACTOR = 1.5
    lid_resistance *= LID_INSULATION_FACTOR * len(boxes)

    total_resistance = nest_resistance + propolis_resistance + lid_resistance + 0.1

    # Surface area and heat calculations
    total_surface_area = sum(
        2 * ((box.width * box.height) + (box.width * box.depth) + (box.height * box.depth)) / 10000
        for box in boxes
    )
    adjusted_surface = total_surface_area ** surface_area_exponent
    adjusted_surface = max(adjusted_surface, 0.0001)
    
    HEAT_RETENTION_FACTOR = 1.4
    heat_gain = (total_heat * total_resistance * HEAT_RETENTION_FACTOR) / adjusted_surface

    # Enhanced cooling effect calculation
    MAX_COOLING_TEMP = 8.0  # Maximum cooling of 8°C
    cooling_factor = min(1.0, colony_size_pct / 80.0)
    
    # Calculate total cooling effect from all boxes
    total_cooling_effect = sum(box.cooling_effect for box in boxes)
    
    # Calculate box temperatures with enhanced cooling effects
    box_temps = []
    for i, box in enumerate(boxes):
        # Base height factor
        height_factor = 1.0 + (i * 0.15)
        
        # Base temperature with height consideration
        box_temp = temp_adj * height_factor
        
        # Solar heating (stronger for upper boxes)
        if is_daytime:
            solar_factor = 1.0 + (i * 0.1)
            box_temp += solar_heat_gain * solar_factor * 0.1
        
        # Enhanced cooling effect calculation with CryoX consideration
        adjusted_cooling = box.adjusted_cooling(environment_factors)
        cooling_temp = (adjusted_cooling / 5.0) * MAX_COOLING_TEMP
        
        if box_temp > species.ideal_temp[1]:
            temp_excess = box_temp - species.ideal_temp[1]
            cooling_multiplier = 1.0 + (temp_excess / 10.0)
            cooling_temp *= cooling_multiplier
        
        # Apply cooling effect
        box_temp -= cooling_temp
        
        # Add propolis heating
        propolis_heating = box.propolis_thickness * 0.06
        box_temp += propolis_heating
        
        # Temperature bounds
        max_temp = species.ideal_temp[1] + 3
        box_temp = max(species.ideal_temp[0], min(max_temp, box_temp))
        box_temps.append(box_temp)

    hive_temp = box_temps[-1]

    # Calculate the average temperature reduction from cooling
    avg_cooling = sum([(box.cooling_effect / 5.0) * MAX_COOLING_TEMP for box in boxes]) / len(boxes)
    
    # Apply cooling effect to base temperature
    if hive_temp > species.ideal_temp[1]:
        temp_excess = hive_temp - species.ideal_temp[1]
        cooling_multiplier = 1.0 + (temp_excess / 10.0)
        hive_temp -= (avg_cooling * cooling_multiplier)

    return {
        "base_temp": hive_temp,
        "box_temps": box_temps,
        "metabolic_heat": metabolic_heat,
        "solar_heat_gain": solar_heat_gain,
        "thermal_resistance": total_resistance,
        "heat_gain": heat_gain,
        "cryox_performance": cryox_performance(**environment_factors)
    }

def calculate_metabolic_heat(species: BeeSpecies, colony_size_pct: float, altitude: float) -> float:
    """
    Calculates the metabolic heat generated by the bee colony.
    """
    OXYGEN_ALTITUDE_SCALE = 7400
    oxygen_factor = max(0.5, np.exp(-altitude / OXYGEN_ALTITUDE_SCALE))
    colony_size = species.colony_size_factor * (colony_size_pct / 100.0)
    base_metabolic = colony_size * species.metabolic_rate * oxygen_factor
    ACTIVITY_MULTIPLIER = 2.5
    return base_metabolic * ACTIVITY_MULTIPLIER

def adjust_temperature(ambient_temp: float, altitude: float, species: BeeSpecies, is_daytime: bool) -> float:
    """
    Adjusts the ambient temperature based on altitude, bee species, and daytime.
    """
    ALTITUDE_TEMP_DROP = 6.5 / 1000  # Temperature drop per meter of altitude
    temp_adj = ambient_temp - (altitude * ALTITUDE_TEMP_DROP)

    if species.activity_profile == "Diurnal":
        temp_adj += 3 if is_daytime else -1
    elif species.activity_profile == "Morning":
        temp_adj += 4 if is_daytime else 0
    else:
        temp_adj += 2 if is_daytime else -0.5
    return temp_adj

def calculate_solar_heat_gain(lat: float, lon: float, is_daytime: bool, day_of_year: int) -> float:
    """
    Estimates solar heat gain in Watts based on location, time of day, and day of year.
    """
    if not is_daytime:
        return 0.0

    SOLAR_CONSTANT = 1367  # W/m^2
    solar_angle = np.cos(np.radians(23.45 * np.sin(np.radians(360 * (day_of_year + 284) / 365))))
    solar_radiation = SOLAR_CONSTANT * solar_angle * 0.7
    HIVE_SURFACE_AREA = 0.25  # m^2
    solar_heat_gain = solar_radiation * HIVE_SURFACE_AREA
    return solar_heat_gain

def create_hive_boxes(species):
    """Creates and configures hive boxes with UI controls."""
    if species.name == "Melipona":
        default_boxes = [
            HiveBox(1, 23, 6, 23, 1.0),
            HiveBox(2, 23, 6, 23, 0.5),
            HiveBox(3, 23, 6, 23, 2.0),
            HiveBox(4, 23, 6, 23, 1.5)
        ]
    else:
        default_boxes = [
            HiveBox(1, 13, 5, 13, 1.0),
            HiveBox(2, 13, 5, 13, 0.5),
            HiveBox(3, 13, 5, 13, 2.0),
            HiveBox(4, 13, 5, 13, 1.5),
            HiveBox(5, 13, 5, 13, 1.0)
        ]
    boxes = []
    for box in default_boxes:
        cols = st.columns(5)  # Changed to 5 columns to include CryoX toggle
        with cols[0]:
            box.width = st.number_input(
                f"Box {box.id} Width (cm)", 
                min_value=10, 
                max_value=50, 
                value=int(box.width),
                help="Width of the hive box in centimeters. Affects heat distribution and colony space."
            )
        with cols[1]:
            box.height = st.number_input(
                f"Box {box.id} Height (cm)", 
                min_value=5, 
                max_value=30, 
                value=int(box.height),
                help="Height of the hive box in centimeters. Affects vertical heat distribution."
            )
        with cols[2]:
            box.depth = st.number_input(
                f"Box {box.id} Depth (cm)", 
                min_value=10, 
                max_value=50, 
                value=int(box.depth),
                help="Depth of the hive box in centimeters. Affects heat retention and colony space."
            )
        with cols[3]:
            box.cooling_effect = st.number_input(
                f"Box {box.id} Cooling Effect (0-5)", 
                min_value=0.0, 
                max_value=5.0,
                value=min(box.cooling_effect, 5.0),
                step=0.5,
                help="Cooling capability of the box (0-5). Higher values mean more cooling (up to -8°C at maximum)."
            )
        with cols[4]:
            box.pdrc_coating = st.checkbox(
                f"CryoX Coating Box {box.id}",
                value=False,
                help="Enable CryoX coating for enhanced cooling performance based on environmental conditions."
            )
        boxes.append(box)
    return boxes

def plot_box_temperatures(boxes: List[HiveBox], box_temps: List[float], species: BeeSpecies) -> go.Figure:
    """
    Creates a bar chart of box temperatures with clear visual indicators for ideal range
    and temperature status.
    """
    labels = [f"Box {box.id}" for box in boxes]
    
    # Create color scale based on temperature ranges
    colors = []
    for temp in box_temps:
        if temp < species.ideal_temp[0]:
            colors.append('blue')  # Too cold
        elif temp > species.ideal_temp[1]:
            colors.append('red')   # Too hot
        else:
            colors.append('green') # Just right

    # Create the bar chart
    fig = go.Figure()
    
    # Add ideal temperature range as a rectangular shape
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=len(boxes) - 0.5,
        y0=species.ideal_temp[0],
        y1=species.ideal_temp[1],
        fillcolor="lightgreen",
        opacity=0.2,
        line=dict(width=0),
        layer="below"
    )

    # Add temperature bars
    fig.add_trace(go.Bar(
        x=labels,
        y=box_temps,
        marker_color=colors,
        text=[f"{temp:.1f}°C" for temp in box_temps],
        textposition='auto',
    ))

    # Add horizontal lines for ideal temperature range
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(boxes) - 0.5,
        y0=species.ideal_temp[0],
        y1=species.ideal_temp[0],
        line=dict(color="green", width=2, dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(boxes) - 0.5,
        y0=species.ideal_temp[1],
        y1=species.ideal_temp[1],
        line=dict(color="green", width=2, dash="dash"),
    )

    # Update layout with more detailed information
    fig.update_layout(
        title=dict(
            text=f"Temperature Distribution in Hive Boxes<br>"
                 f"<sup>Ideal range: {species.ideal_temp[0]}°C - {species.ideal_temp[1]}°C</sup>",
            x=0.5
        ),
        xaxis_title="Box Position",
        yaxis_title="Temperature (°C)",
        yaxis=dict(
            range=[
                min(min(box_temps), species.ideal_temp[0]) - 1,
                max(max(box_temps), species.ideal_temp[1]) + 1
            ]
        ),
        showlegend=False
    )

    # Add annotations for temperature status
    for i, temp in enumerate(box_temps):
        status = "Too Cold" if temp < species.ideal_temp[0] else \
                "Too Hot" if temp > species.ideal_temp[1] else \
                "Ideal"
        fig.add_annotation(
            x=labels[i],
            y=temp,
            text=status,
            yshift=20,
            showarrow=False,
            font=dict(size=10)
        )

    return fig

def plot_hive_3d_structure(boxes: List[HiveBox], box_temps: List[float], species: BeeSpecies) -> go.Figure:
    """
    Creates a 3D visualization of the hive boxes with temperature mapping.
    """
    fig = go.Figure()
    z_offset = 0
    for i, (box, temp) in enumerate(zip(boxes, box_temps)):
        x = [-box.width/2, box.width/2, box.width/2, -box.width/2]
        y = [-box.depth/2, -box.depth/2, box.depth/2, box.depth/2]
        z = [z_offset] * 4
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0], j=[1], k=[2],
            colorscale=[[0, 'blue'], [0.5, 'yellow'], [1, 'red']],
            intensity=[temp],
            name=f'Box {box.id}'
        ))
        z_top = [z_offset + box.height] * 4
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z_top,
            colorscale=[[0, 'blue'], [0.5, 'yellow'], [1, 'red']],
            intensity=[temp],
            showscale=False
        ))
        z_offset += box.height + 2
    fig.update_layout(
        title="3D Hive Structure with Temperature Distribution",
        scene=dict(
            xaxis_title="Width (cm)",
            yaxis_title="Depth (cm)",
            zaxis_title="Height (cm)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=True
    )
    return fig

def main():
    """Main application function."""
    st.set_page_config(page_title="Stingless Bee Hive Thermal Simulator", layout="wide")
    st.title("🍯 Stingless Bee Hive Thermal Simulator")

    # Environmental conditions
    st.sidebar.header("Environmental Conditions")
    environment_factors = {
        "cloud_cover": st.sidebar.slider("Cloud Cover (%)", 0, 100, 50) / 100,
        "humidity": st.sidebar.slider("Humidity (%)", 0, 100, 50),
        "wind_speed": st.sidebar.slider("Wind Speed (mph)", 0, 20, 5),
        "temperature": st.sidebar.slider("Temperature (°F)", 50, 120, 85),
        "solar_angle": st.sidebar.slider("Solar Angle (degrees)", 0, 90, 45),
        "air_quality": st.sidebar.slider("Air Quality Index (0-1)", 0.0, 1.0, 0.5),
        "uv_index": st.sidebar.slider("UV Index (0-11)", 0, 11, 5)
    }

    # Hive configuration
    st.sidebar.header("Hive Configuration")
    species_key = st.sidebar.selectbox(
        "Select Bee Species", 
        list(SPECIES_CONFIG.keys()),
        help="Choose the stingless bee species. Each species has different temperature preferences and colony characteristics."
    )
    species = SPECIES_CONFIG[species_key]
    
    st.sidebar.markdown(f"**{species.name} Characteristics:**")
    st.sidebar.write(f"Ideal Temperature: {species.ideal_temp[0]}–{species.ideal_temp[1]} °C")
    st.sidebar.write(f"Humidity Range: {species.humidity_range[0]}–{species.humidity_range[1]} %")
    st.sidebar.write(f"Activity Profile: {species.activity_profile}")

    colony_size_pct = st.sidebar.slider(
        "Colony Size (%)", 
        0, 100, 50,
        help="Percentage of maximum colony size. Larger colonies generate more heat and have better temperature regulation."
    )
    
    nest_thickness = st.sidebar.slider(
        "Nest Wall Thickness (mm)", 
        1.0, 10.0, 5.0,
        help="Thickness of the nest walls. Thicker walls provide better insulation."
    )
    
    lid_thickness = st.sidebar.slider(
        "Lid Thickness (mm)", 
        1.0, 10.0, 5.0,
        help="Thickness of the hive lid. Thicker lids help retain heat and provide better insulation."
    )
    
    rain_intensity = st.sidebar.slider(
        "Rain Intensity (0 to 1)", 
        0.0, 1.0, 0.0, 
        step=0.1,
        help="Intensity of rainfall. Higher values mean more cooling effect on the hive."
    )
    
    surface_area_exponent = st.sidebar.slider(
        "Surface Area Exponent", 
        1.0, 2.0, 1.0, 
        step=0.1,
        help="Affects how heat dissipates based on hive surface area. Higher values mean more heat loss through surfaces."
    )

    # CryoX comparison
    st.subheader("CryoX Cooling Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Cooling Effect Comparison")
        # Create two sample hives for comparison
        hive_no_cryox = HiveBox(id=1, width=30, height=40, depth=50, cooling_effect=1.0, pdrc_coating=False)
        hive_with_cryox = HiveBox(id=2, width=30, height=40, depth=50, cooling_effect=1.0, pdrc_coating=True)

        cooling_no_cryox = hive_no_cryox.adjusted_cooling(environment_factors)
        cooling_with_cryox = hive_with_cryox.adjusted_cooling(environment_factors)

        st.write(f"Cooling effect **without CryoX**: {cooling_no_cryox:.2f}")
        st.write(f"Cooling effect **with CryoX**: {cooling_with_cryox:.2f}")
        st.write(f"Performance improvement: **{((cooling_with_cryox/cooling_no_cryox - 1) * 100):.1f}%**")

    with col2:
        # Plotting the comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Without CryoX", "With CryoX"], 
            y=[cooling_no_cryox, cooling_with_cryox],
            name="Cooling Effect",
            marker_color=['lightblue', 'royalblue']
        ))
        fig.update_layout(
            title="Comparison of Cooling Effect",
            yaxis_title="Cooling Effect Factor",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Advanced Configuration
    with st.expander("Advanced Hive Configuration"):
        st.info("Configure detailed parameters for each hive box. These settings affect heat distribution and retention.")
        boxes = create_hive_boxes(species)
        gps_input = st.text_input(
            "Enter GPS Coordinates (lat,lon)", 
            "-3.4653,-62.2159",
            help="Geographic coordinates of the hive location. Used to calculate solar exposure and day/night cycles."
        )
        
        gps = parse_gps_input(gps_input)
        if gps is None:
            st.error("Invalid GPS input. Please enter coordinates as 'lat,lon'.")
            return
            
        lat, lon = gps
        altitude = get_altitude(lat, lon)
        if altitude is None:
            st.warning("Could not retrieve altitude. Please enter altitude manually.")
            altitude = st.slider(
                "Altitude (m)", 
                0, 5000, 100,
                help="Height above sea level. Affects air temperature and density."
            )
        else:
            st.write(f"Altitude: {altitude} m")
            
        is_daytime = is_daytime_calc(lat, lon)
        st.write(f"It is daytime: {is_daytime}")
        
        weather = get_weather_data(lat, lon)
        if weather and weather.get("temperature") is not None:
            ambient_temp = weather["temperature"]
            st.write(f"Current Ambient Temperature: {ambient_temp} °C")
        else:
            st.warning("Weather data unavailable. Please use the slider below.")
            ambient_temp = st.slider(
                "Ambient Temperature (°C)", 
                15.0, 40.0, 28.0,
                help="Outside air temperature. Major factor in hive temperature regulation."
            )

    if st.button("Run Simulation", help="Calculate hive temperatures based on current parameters and display results."):
        # Add current timestamp to force update
        st.session_state.simulation_time = datetime.datetime.now().timestamp()
        
        results = simulate_hive_temperature(
            species=species,
            colony_size_pct=colony_size_pct,
            nest_thickness=nest_thickness,
            lid_thickness=lid_thickness,
            boxes=boxes,
            ambient_temp=(weather["temperature"] - 32) * 5/9,  # Convert °F to °C
            is_daytime=is_daytime,
            altitude=altitude,
            rain_intensity=rain_intensity,
            surface_area_exponent=surface_area_exponent,
            lat=lat,
            lon=lon,
            day_of_year=datetime.datetime.now().timetuple().tm_yday,
            **environment_factors
        )
        
        # Store results in session state
        st.session_state.last_results = results

    # Display results if they exist
    if 'last_results' in st.session_state:
        results = st.session_state.last_results
        st.subheader("Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Base Hive Temperature", 
                f"{results['base_temp']:.1f} °C",
                help="The baseline temperature inside the hive before considering individual box variations."
            )
            st.metric(
                "Metabolic Heat Output", 
                f"{results['metabolic_heat']:.2f} W",
                help="Heat generated by the bee colony through their metabolic activity."
            )
        with col2:
            st.metric(
                "Solar Heat Gain", 
                f"{results['solar_heat_gain']:.2f} W",
                help="Heat absorbed from sunlight exposure."
            )
        with col3:
            st.write(
                "Thermal Resistance:", 
                f"{results['thermal_resistance']:.3f}",
                help="The hive's ability to resist heat flow. Higher values mean better insulation."
            )
            st.write(
                "Heat Gain:", 
                f"{results['heat_gain']:.3f}",
                help="Total heat accumulation in the hive from all sources."
            )
            
        st.subheader("Temperature Status")
        if results['base_temp'] < species.ideal_temp[0]:
            st.error(f"⚠️ Alert: Hive is too cold! Current temperature ({results['base_temp']:.1f}°C) is below the ideal range ({species.ideal_temp[0]}-{species.ideal_temp[1]}°C).")
        elif results['base_temp'] > species.ideal_temp[1]:
            st.error(f"⚠️ Alert: Hive is too hot! Current temperature ({results['base_temp']:.1f}°C) is above the ideal range ({species.ideal_temp[0]}-{species.ideal_temp[1]}°C).")
        else:
            st.success(f"✅ Hive temperature ({results['base_temp']:.1f}°C) is within the ideal range ({species.ideal_temp[0]}-{species.ideal_temp[1]}°C).")
            
        # Force graph updates by adding simulation time to the key
        st.plotly_chart(
            plot_box_temperatures(boxes, results["box_temps"], species), 
            use_container_width=True,
            key=f"temp_plot_{st.session_state.get('simulation_time', 0)}",
            help="Visual representation of temperature distribution across hive boxes."
        )
        st.plotly_chart(
            plot_hive_3d_structure(boxes, results["box_temps"], species), 
            use_container_width=True,
            key=f"3d_plot_{st.session_state.get('simulation_time', 0)}",
            help="3D visualization of the hive structure with temperature mapping."
        )

if __name__ == "__main__":
    main()

