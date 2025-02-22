import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime
import numpy as np
import plotly.graph_objects as go
import pytz
from timezonefinder import TimezoneFinder
from dataclasses import dataclass
from typing import List, Tuple, Dict

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
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"

# Utility functions
def parse_gps_input(gps_str: str) -> Tuple[float, float] | None:
    """Parse GPS coordinates from string input."""
    try:
        lat, lon = map(float, gps_str.strip().split(','))
        return lat, lon
    except ValueError:
        return None

@st.cache_data
def get_historical_weather_data(lat: float, lon: float, start_date: str, end_date: str):
    """Fetch historical hourly temperature data from Open-Meteo API."""
    url = OPEN_METEO_ARCHIVE_URL.format(lat=lat, lon=lon, start_date=start_date, end_date=end_date)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        times = data['hourly']['time']
        temperatures = data['hourly']['temperature_2m']
        df = pd.DataFrame({'time': times, 'temperature': temperatures})
        df['time'] = pd.to_datetime(df['time'])
        return df
    except requests.RequestException as e:
        st.error(f"Failed to fetch historical weather data: {e}")
        return None

@st.cache_data
def get_current_weather_data(lat: float, lon: float):
    """Fetch current weather data from Open-Meteo API."""
    url = OPEN_METEO_URL.format(lat=lat, lon=lon)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        current = data.get("current_weather")
        if current:
            return {
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed")
            }
    except requests.RequestException as e:
        st.error(f"Failed to fetch current weather data: {e}")
    return None

@st.cache_data
def get_altitude(lat: float, lon: float):
    """Fetch altitude data from Open Elevation API."""
    url = OPEN_ELEVATION_URL.format(lat=lat, lon=lon)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        results = data.get("results")
        if results and isinstance(results, list):
            return results[0].get("elevation")
    except requests.RequestException as e:
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

def is_daytime_calc(lat: float, lon: float):
    """Determine if it's currently daytime at the given coordinates."""
    try:
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        if not timezone_str:
            return True  # Default to daytime
        timezone = pytz.timezone(timezone_str)
        current_time = datetime.now(timezone)
        return 6 <= current_time.hour < 18
    except Exception as e:
        st.warning(f"Could not determine daytime status: {e}")
        return True

# Simple Calculation Mode
def adjust_k_for_volume(base_k, actual_volume, standard_volume=2.0):
    """Adjust thermal transfer factor based on hive volume."""
    if actual_volume <= 0:
        raise ValueError("Hive volume must be greater than zero.")
    scaling_factor = standard_volume / actual_volume
    adjusted_k = base_k * scaling_factor
    return min(max(adjusted_k, 0.0), 1.0)

def simple_calculation(lat, lon):
    """Perform simple hive temperature calculation."""
    st.subheader("Simple Hive Temperature Calculation")
    start_date = st.date_input("Start Date", value=date(2023, 1, 1), help="Select start date for weather data.", key="simple_start_date")
    end_date = st.date_input("End Date", value=date(2023, 1, 31), help="Select end date for weather data.", key="simple_end_date")
    base_k = st.slider("Base Thermal Transfer Factor (k)", 0.0, 1.0, 0.5, step=0.1, help="Base factor for cooling effect.", key="simple_base_k")
    delta_T_roof = st.slider("Roof Temperature Reduction (°C)", 0.0, 10.0, 8.5, step=0.5, help="Cooling from hive roof.", key="simple_delta_T_roof")
    hive_volume = st.number_input("Hive Internal Volume (liters)", 0.5, 5.0, 2.0, step=0.1, help="Hive volume in liters.", key="simple_hive_volume")

    if st.button("Calculate", key="simple_calculate"):
        weather_df = get_historical_weather_data(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if weather_df is not None:
            try:
                adjusted_k = adjust_k_for_volume(base_k, hive_volume)
            except ValueError as e:
                st.error(e)
                return
            weather_df['internal_temperature'] = weather_df['temperature'] - adjusted_k * delta_T_roof
            st.write("#### Weather Data and Calculated Internal Temperature")
            st.dataframe(weather_df.style.format({"temperature": "{:.2f}", "internal_temperature": "{:.2f}"}))
            st.write("#### Temperature Over Time")
            plt.figure(figsize=(10, 5))
            plt.plot(weather_df['time'], weather_df['temperature'], label='External Temperature', color='red')
            plt.plot(weather_df['time'], weather_df['internal_temperature'], label='Internal Temperature', color='blue')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°C)')
            plt.title('External vs. Internal Hive Temperature')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            avg_external = weather_df['temperature'].mean()
            avg_internal = weather_df['internal_temperature'].mean()
            st.write(f"**Average External Temperature:** {avg_external:.2f} °C")
            st.write(f"**Average Internal Temperature:** {avg_internal:.2f} °C")
            st.write(f"**Average Temperature Reduction:** {avg_external - avg_internal:.2f} °C")
            st.write(f"**Adjusted k (based on hive volume):** {adjusted_k:.2f}")
        else:
            st.write("No data available. Please check your inputs and try again.")

# Detailed Simulation Mode
def create_hive_boxes(species):
    """Create and configure hive boxes with UI controls."""
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
        cols = st.columns(5)
        with cols[0]:
            box.width = st.number_input(f"Box {box.id} Width (cm)", min_value=10, max_value=50, value=int(box.width), key=f"box_{box.id}_width")
        with cols[1]:
            box.height = st.number_input(f"Box {box.id} Height (cm)", min_value=5, max_value=30, value=int(box.height), key=f"box_{box.id}_height")
        with cols[2]:
            box.depth = st.number_input(f"Box {box.id} Depth (cm)", min_value=10, max_value=50, value=int(box.depth), key=f"box_{box.id}_depth")
        with cols[3]:
            box.cooling_effect = st.number_input(f"Box {box.id} Cooling Effect (0-5)", min_value=0.0, max_value=5.0, value=min(box.cooling_effect, 5.0), step=0.5, key=f"box_{box.id}_cooling")
        with cols[4]:
            box.pdrc_coating = st.checkbox(f"CryoX Coating Box {box.id}", value=False, key=f"box_{box.id}_cryox")
        boxes.append(box)
    return boxes

def simulate_hive_temperature(species, colony_size_pct, nest_thickness, lid_thickness, boxes, ambient_temp, is_daytime, altitude, rain_intensity, surface_area_exponent, lat, lon, day_of_year, **env_factors):
    """Simulate hive internal temperature."""
    colony_size = species.colony_size_factor * (colony_size_pct / 100)
    metabolic_heat = species.metabolic_rate * colony_size * (1.2 if is_daytime else 0.8)
    total_volume = sum(box.width * box.height * box.depth for box in boxes) / 1000  # Convert cm³ to liters
    surface_area = sum(2 * (box.width * box.height + box.width * box.depth + box.height * box.depth) for box in boxes) / 10000  # Convert cm² to m²
    adjusted_surface_area = surface_area ** surface_area_exponent
    heat_loss = species.nest_conductivity * adjusted_surface_area * (ambient_temp - species.ideal_temp[0]) / (nest_thickness / 1000)
    cooling = sum(box.adjusted_cooling(env_factors) for box in boxes) * (1 - rain_intensity * 0.5)
    base_temp = ambient_temp + (metabolic_heat - heat_loss - cooling) / total_volume
    box_temps = []
    for box in boxes:
        box_cooling = box.adjusted_cooling(env_factors)
        box_temp = base_temp - box_cooling * (box.width * box.height * box.depth / 1000) / total_volume
        box_temps.append(max(species.ideal_temp[0], min(species.ideal_temp[1], box_temp)))
    return {"base_temp": base_temp, "box_temps": box_temps}

def plot_box_temperatures(boxes: List[HiveBox], box_temps: List[float], species: BeeSpecies):
    """Plot temperature per hive box."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Box {box.id}" for box in boxes],
        y=box_temps,
        name="Box Temperature",
        marker_color='blue'
    ))
    fig.add_hline(y=species.ideal_temp[0], line_dash="dash", line_color="green", annotation_text="Min Ideal Temp")
    fig.add_hline(y=species.ideal_temp[1], line_dash="dash", line_color="red", annotation_text="Max Ideal Temp")
    fig.update_layout(
        title="Temperature per Hive Box",
        xaxis_title="Hive Box",
        yaxis_title="Temperature (°C)",
        bargap=0.2
    )
    return fig

def plot_hive_3d_structure(boxes: List[HiveBox], box_temps: List[float], species: BeeSpecies):
    """Plot 3D hive structure with temperature coloring."""
    x, y, z = [], [], []
    colors = []
    for i, box in enumerate(boxes):
        x.extend([0, box.width, box.width, 0, 0])
        y.extend([0, 0, box.depth, box.depth, 0])
        z.extend([i * box.height, i * box.height, i * box.height, i * box.height, i * box.height])
        colors.append(box_temps[i])
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(color=colors, colorscale='Viridis', width=5),
        marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True)
    )])
    fig.update_layout(
        title="3D Hive Structure with Temperature Gradient",
        scene=dict(
            xaxis_title="Width (cm)",
            yaxis_title="Depth (cm)",
            zaxis_title="Height (cm)"
        )
    )
    return fig

def detailed_simulation(lat, lon):
    """Perform detailed hive thermal simulation."""
    st.subheader("Detailed Hive Thermal Simulation")
    species_key = st.selectbox("Select Bee Species", list(SPECIES_CONFIG.keys()), key="detailed_species")
    species = SPECIES_CONFIG[species_key]
    colony_size_pct = st.slider("Colony Size (%)", 0, 100, 50, key="detailed_colony_size")
    nest_thickness = st.slider("Nest Wall Thickness (mm)", 1.0, 10.0, 5.0, key="detailed_nest_thickness")
    lid_thickness = st.slider("Lid Thickness (mm)", 1.0, 10.0, 5.0, key="detailed_lid_thickness")
    rain_intensity = st.slider("Rain Intensity (0 to 1)", 0.0, 1.0, 0.0, step=0.1, key="detailed_rain_intensity")
    surface_area_exponent = st.slider("Surface Area Exponent", 1.0, 2.0, 1.0, step=0.1, key="detailed_surface_area_exponent")

    with st.expander("Advanced Hive Configuration"):
        boxes = create_hive_boxes(species)
        altitude = get_altitude(lat, lon)
        if altitude is None:
            altitude = st.slider("Altitude (m)", 0, 5000, 100, key="detailed_altitude")
        is_daytime = is_daytime_calc(lat, lon)
        weather = get_current_weather_data(lat, lon)
        ambient_temp = weather["temperature"] if weather else st.slider("Ambient Temperature (°C)", 15.0, 40.0, 28.0, key="detailed_ambient_temp")

    environment_factors = {
        "cloud_cover": st.slider("Cloud Cover (%)", 0, 100, 50, key="detailed_cloud_cover") / 100,
        "humidity": st.slider("Humidity (%)", 0, 100, 50, key="detailed_humidity"),
        "wind_speed": st.slider("Wind Speed (mph)", 0, 20, 5, key="detailed_wind_speed"),
        "temperature": ambient_temp,
        "solar_angle": st.slider("Solar Angle (degrees)", 0, 90, 45, key="detailed_solar_angle"),
        "air_quality": st.slider("Air Quality Index (0-1)", 0.0, 1.0, 0.5, key="detailed_air_quality"),
        "uv_index": st.slider("UV Index (0-11)", 0, 11, 5, key="detailed_uv_index")
    }

    if st.button("Run Simulation", key="detailed_run_simulation"):
        results = simulate_hive_temperature(
            species=species,
            colony_size_pct=colony_size_pct,
            nest_thickness=nest_thickness,
            lid_thickness=lid_thickness,
            boxes=boxes,
            ambient_temp=ambient_temp,
            is_daytime=is_daytime,
            altitude=altitude,
            rain_intensity=rain_intensity,
            surface_area_exponent=surface_area_exponent,
            lat=lat,
            lon=lon,
            day_of_year=datetime.now().timetuple().tm_yday,
            **environment_factors
        )
        st.session_state.last_results = results

    if 'last_results' in st.session_state:
        results = st.session_state.last_results
        st.subheader("Simulation Results")
        st.metric("Base Hive Temperature", f"{results['base_temp']:.1f} °C")
        st.plotly_chart(plot_box_temperatures(boxes, results["box_temps"], species), use_container_width=True)
        st.plotly_chart(plot_hive_3d_structure(boxes, results["box_temps"], species), use_container_width=True)

# Main function
def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(page_title="Hive Thermal Simulator", layout="wide")
    st.title("Hive Thermal Simulator")
    st.sidebar.header("Location")
    gps_input = st.sidebar.text_input("Enter GPS Coordinates (lat,lon)", "-3.4653,-62.2159")
    gps = parse_gps_input(gps_input)
    if gps is None:
        st.error("Invalid GPS input. Please enter coordinates as 'lat,lon'.")
        return
    lat, lon = gps

    mode = st.sidebar.selectbox("Select Mode", ["Simple Calculation", "Detailed Simulation"])
    if mode == "Simple Calculation":
        simple_calculation(lat, lon)
    elif mode == "Detailed Simulation":
        detailed_simulation(lat, lon)

    st.write("""
    *Note: This is a simplified model assuming a linear relationship between roof cooling, hive size, and internal temperature. 
    Actual hive temperatures may vary due to bee activity, insulation, and other environmental factors.*
    """)

if __name__ == "__main__":
    main()
