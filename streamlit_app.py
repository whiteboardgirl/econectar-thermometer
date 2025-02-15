import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests

# Constants
KELVIN_CONVERSION = 273.15
AIR_FILM_RESISTANCE_OUTSIDE = 0.04  # m²K/W
BEE_METABOLIC_HEAT = 0.0040  # Watts per bee
IDEAL_HIVE_TEMPERATURE = 35.0  # °C

# Utility Functions
def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + KELVIN_CONVERSION

def kelvin_to_celsius(kelvin):
    """Convert Kelvin to Celsius."""
    return kelvin - KELVIN_CONVERSION

def calculate_oxygen_factor(altitude_m):
    """Calculate oxygen factor based on altitude."""
    P0 = 1013.25  # Standard atmospheric pressure at sea level (hPa)
    H = 7400  # Scale height for Earth's atmosphere (m)
    pressure_ratio = math.exp(-altitude_m / H)
    return max(0.6, pressure_ratio)

def calculate_box_surface_area(width_cm, height_cm):
    """Calculate surface area for a hexagonal box in square meters."""
    width_m, height_m = width_cm / 100, height_cm / 100
    side_length = width_m / math.sqrt(3)
    hexagon_area = (3 * math.sqrt(3) / 2) * (side_length ** 2)
    sides_area = 6 * side_length * height_m
    return (2 * hexagon_area) + sides_area

def adjust_for_time_of_day(is_daytime, params, altitude):
    """Adjust parameters based on time of day and altitude."""
    if is_daytime:
        params['ideal_hive_temperature'] += 1.0  
        params['bee_metabolic_heat'] *= 1.1  
    else:
        params['ideal_hive_temperature'] -= 0.5  
        params['air_film_resistance_outside'] *= 1.1  
    
    oxygen_factor = calculate_oxygen_factor(altitude)
    params['bee_metabolic_heat'] *= oxygen_factor  
    params['air_film_resistance_outside'] *= (1 + (altitude / 1000) * 0.05)  
    params['ideal_hive_temperature'] -= (altitude / 1000) * 0.5  
    return params

def calculate_hive_temperature(params, boxes, ambient_temp_c, is_daytime, altitude):
    """Calculate hive temperature with adjustments for conditions."""
    params = adjust_for_time_of_day(is_daytime, params, altitude)
    
    ambient_temp_k = celsius_to_kelvin(ambient_temp_c)
    ideal_temp_k = celsius_to_kelvin(params['ideal_hive_temperature'])
    
    calculated_colony_size = 50000 * (params['colony_size'] / 100)
    oxygen_factor = calculate_oxygen_factor(altitude)
    colony_metabolic_heat = calculated_colony_size * params['bee_metabolic_heat'] * oxygen_factor

    total_volume = sum(
        (3 * math.sqrt(3) / 2) * ((box['width'] / (100 * math.sqrt(3))) ** 2) * (box['height'] / 100)
        for box in boxes
    )
    total_surface_area = sum(calculate_box_surface_area(box['width'], box['height']) for box in boxes)
    
    wood_resistance = (params['wood_thickness'] / 100) / params['wood_thermal_conductivity']
    total_resistance = wood_resistance + params['air_film_resistance_outside']

    if ambient_temp_c >= params['ideal_hive_temperature']:
        cooling_effort = min(1.0, (ambient_temp_c - params['ideal_hive_temperature']) / 10)  
        temp_decrease = 2.0 * cooling_effort if is_daytime else 1.0 * cooling_effort  
        estimated_temp_c = max(params['ideal_hive_temperature'], ambient_temp_c - temp_decrease)
    else:
        heat_contribution = min(
            params['ideal_hive_temperature'] - ambient_temp_c,
            (colony_metabolic_heat * total_resistance) / total_surface_area
        )
        heat_contribution *= 0.9 if not is_daytime else 1.0  
        estimated_temp_c = ambient_temp_c + heat_contribution
    
    estimated_temp_c -= (altitude / 1000) * 0.5  
    estimated_temp_c = min(50, max(0, estimated_temp_c))
    estimated_temp_k = celsius_to_kelvin(estimated_temp_c)

    final_heat_transfer = (total_surface_area * abs(estimated_temp_k - ambient_temp_k)) / total_resistance

    box_temperatures = [
        max(0, min(50, estimated_temp_c - box['cooling_effect']))
        for box in boxes
    ]

    return {
        'calculated_colony_size': calculated_colony_size,
        'colony_metabolic_heat': colony_metabolic_heat / 1000,
        'base_temperature': estimated_temp_c,
        'box_temperatures': box_temperatures,
        'total_volume': total_volume,
        'total_surface_area': total_surface_area,
        'thermal_resistance': total_resistance,
        'ambient_temperature': ambient_temp_c,
        'oxygen_factor': oxygen_factor,
        'heat_transfer': final_heat_transfer / 1000
    }

# Streamlit Setup
st.set_page_config(
    page_title="Hive Thermal Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

def get_temperature_from_coordinates(lat, lon):
    url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        data = response.json()
        return data['current_weather']['temperature']
    except requests.exceptions.HTTPError as errh:
        st.error(f"Http Error: {errh}")
        st.write(f"Debug Info - URL: {url}")
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
        st.write(f"Debug Info - URL: {url}")
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
        st.write(f"Debug Info - URL: {url}")
    except requests.exceptions.RequestException as err:
        st.error(f"Unknown Error: {err}")
        st.write(f"Debug Info - URL: {url}")
    except KeyError:
        st.error("API response does not contain expected 'current_weather' key")
        st.write(f"Debug Info - URL: {url}")
    return None

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.boxes = [
        {'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': ce}
        for i, ce in enumerate([2, 0, 0, 8])
    ]

# Page header
st.title("🐝 Hive Thermal Dashboard")
st.markdown("---")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Input Parameters")
    
    gps_coordinates = st.text_input("Enter GPS Coordinates (lat, lon)", "4.6097, -74.0817")
    try:
        lat, lon = map(float, gps_coordinates.split(','))
        ambient_temperature = get_temperature_from_coordinates(lat, lon)
        if ambient_temperature is None:
            ambient_temperature = 25.0  
    except ValueError:
        st.error("Please enter valid coordinates in the format 'lat, lon'")
        ambient_temperature = 25.0  

    is_daytime = st.radio("Time of Day", ['Day', 'Night'], index=0)
    st.write(f"Current Ambient Temperature: {ambient_temperature}°C")
    
    colony_size = st.slider("Colony Size (%)", 0, 100, 50)
    altitude = st.slider("Altitude (meters)", 0, 3800, 0, 100)
    oxygen_factor = calculate_oxygen_factor(altitude)
    progress_value = (oxygen_factor - 0.6) / (1.0 - 0.6)  
    
    st.progress(progress_value)
    st.caption(f"Oxygen Factor: {oxygen_factor:.2f}")

    st.subheader("📦 Box Configuration")
    for i, box in enumerate(st.session_state.boxes):
        with st.expander(f"Box {box['id']}", expanded=True):
            box['width'] = st.slider(f"Width for Box {box['id']} (cm)", 10, 50, box['width'])
            box['height'] = st.slider(f"Height for Box {box['id']} (cm)", 5, 20, box['height'])
            box['cooling_effect'] = st.number_input(
                "Cooling Effect (°C)", 0.0, 20.0, float(box['cooling_effect']), 0.5, key=f"cooling_effect_{i}"
            )

# Parameters dictionary
params = {
    'colony_size': colony_size,
    'bee_metabolic_heat': BEE_METABOLIC_HEAT,
    'wood_thickness': st.slider("Wood Thickness (cm)", 1.0, 5.0, 2.0),
    'wood_thermal_conductivity': st.slider("Thermal Conductivity (W/(m⋅K))", 0.1, 0.3, 0.13, step=0.01),
    'air_film_resistance_outside': AIR_FILM_RESISTANCE_OUTSIDE,
    'ideal_hive_temperature': IDEAL_HIVE_TEMPERATURE
}

# Convert radio selection to boolean
is_daytime = is_daytime == 'Day'

# Calculate results
results = calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, is_daytime, altitude)

# Display results
with col2:
    st.subheader("📈 Analysis Results")
    
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
        progress_value = max(0.0, min(1.0, temp / 50))
        st.progress(progress_value)

    # Add a graph for temperature distribution
    try:
        fig, ax = plt.subplots()
        ax.bar([f'Box {i+1}' for i in range(len(results['box_temperatures']))], results['box_temperatures'])
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Distribution Across Hive Boxes')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{b64}"/>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while generating the graph: {e}")

    st.metric("Total Hive Volume", f"{results['total_volume']:.2f} m³")
    st.metric("Total Surface Area", f"{results['total_surface_area']:.2f} m²")
    st.metric("Heat Transfer", f"{results['heat_transfer']:.3f} kW")

        # Additional features for interactivity
    if st.button("Refresh Weather Data"):
        ambient_temperature = get_temperature_from_coordinates(lat, lon) or 25.0
        results = calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, is_daytime, altitude)
    
    # Advanced Settings
    with st.expander("Advanced Settings"):
        params['ideal_hive_temperature'] = st.slider("Ideal Hive Temperature (°C)", 30.0, 40.0, IDEAL_HIVE_TEMPERATURE, 0.1)
        params['bee_metabolic_heat'] = st.slider("Bee Metabolic Heat (W/bee)", 0.003, 0.005, BEE_METABOLIC_HEAT, 0.0001)
        
        # Option to change the number of boxes dynamically
        num_boxes = st.slider("Number of Boxes", 1, 10, len(st.session_state.boxes))
        if num_boxes != len(st.session_state.boxes):
            st.session_state.boxes = st.session_state.boxes[:num_boxes] + [{'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': 0} for i in range(len(st.session_state.boxes), num_boxes)]
    
    # Error handling and user feedback
    if 'error' in st.session_state:
        st.error(st.session_state.error)
        st.session_state.pop('error', None)  # Clear error message after display

    # Save/Load State
    if st.button("Save State"):
        st.session_state.saved_state = {
            'params': params,
            'boxes': st.session_state.boxes,
            'is_daytime': is_daytime,
            'altitude': altitude,
            'gps_coordinates': gps_coordinates
        }
        st.success("State saved successfully.")

    if st.button("Load State"):
        if 'saved_state' in st.session_state:
            saved = st.session_state.saved_state
            params.update(saved['params'])
            st.session_state.boxes = saved['boxes']
            is_daytime = saved['is_daytime']
            altitude = saved['altitude']
            gps_coordinates = saved['gps_coordinates']
            st.success("State loaded successfully.")
        else:
            st.error("No saved state found.")

    # Debugging and Developer Tools
    if st.checkbox("Show Debug Info"):
        st.write("Current Parameters:", params)
        st.write("Current Boxes:", st.session_state.boxes)
        st.write("Other Data:", {
            'is_daytime': is_daytime,
            'altitude': altitude,
            'gps_coordinates': gps_coordinates,
            'ambient_temperature': ambient_temperature
        })

# Main execution
if __name__ == "__main__":
    st.session_state.setdefault('initialized', False)
    st.session_state.setdefault('boxes', [])
    st.session_state.setdefault('saved_state', {})
    
    if not st.session_state.initialized:
        st.session_state.initialized = True
        st.session_state.boxes = [
            {'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': ce}
            for i, ce in enumerate([2, 0, 0, 8])
        ]

    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the input parameters or try refreshing the page.")

            # Visualization Enhancements
    st.subheader("📊 Advanced Visualizations")

    # Temperature vs. Altitude
    altitude_range = np.arange(0, 4000, 100)
    temp_changes = [calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, is_daytime, alt)['base_temperature'] for alt in altitude_range]
    
    fig_alt, ax_alt = plt.subplots()
    ax_alt.plot(altitude_range, temp_changes)
    ax_alt.set_title("Hive Temperature vs. Altitude")
    ax_alt.set_xlabel("Altitude (m)")
    ax_alt.set_ylabel("Hive Temperature (°C)")
    
    buf_alt = BytesIO()
    fig_alt.savefig(buf_alt, format='png')
    buf_alt.seek(0)
    b64_alt = base64.b64encode(buf_alt.read()).decode()
    st.image(f"data:image/png;base64,{b64_alt}")

    # Temperature vs. Time of Day
    temp_day = calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, True, altitude)['base_temperature']
    temp_night = calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, False, altitude)['base_temperature']

    fig_day_night, ax_day_night = plt.subplots()
    ax_day_night.bar(['Day', 'Night'], [temp_day, temp_night])
    ax_day_night.set_title("Hive Temperature by Time of Day")
    ax_day_night.set_ylabel("Hive Temperature (°C)")

    buf_day_night = BytesIO()
    fig_day_night.savefig(buf_day_night, format='png')
    buf_day_night.seek(0)
    b64_day_night = base64.b64encode(buf_day_night.read()).decode()
    st.image(f"data:image/png;base64,{b64_day_night}")

    # Interactive Exploration of Hive Parameters
    with st.expander("Explore Hive Parameters"):
        param_to_explore = st.selectbox("Select Parameter to Explore", ['colony_size', 'wood_thickness', 'wood_thermal_conductivity'])
        range_values = st.slider(f"Range for {param_to_explore}", 
                                 min_value=0.0 if param_to_explore in ['colony_size', 'wood_thickness'] else 0.1, 
                                 max_value=100.0 if param_to_explore == 'colony_size' else 5.0 if param_to_explore == 'wood_thickness' else 0.3, 
                                 value=(0.1, 1.0) if param_to_explore == 'wood_thermal_conductivity' else (10.0, 90.0) if param_to_explore == 'colony_size' else (1.0, 3.0),
                                 step=0.01 if param_to_explore == 'wood_thermal_conductivity' else 1.0)

        explore_range = np.linspace(*range_values, num=20)
        explore_temps = []
        for val in explore_range:
            temp_params = params.copy()
            temp_params[param_to_explore] = val
            explore_temps.append(calculate_hive_temperature(temp_params, st.session_state.boxes, ambient_temperature, is_daytime, altitude)['base_temperature'])

        fig_explore, ax_explore = plt.subplots()
        ax_explore.plot(explore_range, explore_temps)
        ax_explore.set_title(f"Hive Temperature vs. {param_to_explore}")
        ax_explore.set_xlabel(f"{param_to_explore}")
        ax_explore.set_ylabel("Hive Temperature (°C)")
        
        buf_explore = BytesIO()
        fig_explore.savefig(buf_explore, format='png')
        buf_explore.seek(0)
        b64_explore = base64.b64encode(buf_explore.read()).decode()
        st.image(f"data:image/png;base64,{b64_explore}")

    # Educational Content
    st.subheader("📖 Learn More")
    st.markdown("""
    - **Hive Temperature Regulation**: Bees regulate hive temperature through various behaviors like fanning, clustering, or water evaporation. 
    - **Impact of Altitude**: Higher altitudes mean less oxygen, which can affect bee metabolism and thus heat generation.
    - **Time of Day**: Bees are more active during the day, leading to higher internal hive temperatures. At night, clustering helps retain heat.
    """)

    # Feedback or Questions
    with st.expander("Leave Feedback or Ask Questions"):
        feedback = st.text_area("Your feedback or questions here")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback! We'll review it soon.")
            # Here you would typically save or send this feedback somewhere

# Main execution
if __name__ == "__main__":
    st.session_state.setdefault('initialized', False)
    st.session_state.setdefault('boxes', [])
    st.session_state.setdefault('saved_state', {})
    
    if not st.session_state.initialized:
        st.session_state.initialized = True
        st.session_state.boxes = [
            {'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': ce}
            for i, ce in enumerate([2, 0, 0, 8])
        ]

    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the input parameters or try refreshing the page.")

            # Real-time Data Updates
    if st.button("Start Real-time Updates"):
        st.session_state.realtime = not st.session_state.get('realtime', False)
        if st.session_state.realtime:
            st.info("Real-time updates started. Click again to stop.")
        else:
            st.info("Real-time updates stopped.")

    if 'realtime' in st.session_state and st.session_state.realtime:
        import time
        st.write("Updating in real-time...")
        for _ in range(5):
            ambient_temperature = get_temperature_from_coordinates(lat, lon) or 25.0
            results = calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, is_daytime, altitude)
            
            st.metric("Current Ambient Temperature", f"{ambient_temperature:.1f}°C")
            st.metric("Current Hive Temperature", f"{results['base_temperature']:.1f}°C")
            time.sleep(10)  # Update every 10 seconds
            if not st.session_state.realtime:
                break

    # Simulation Over Time
    with st.expander("Simulate Over Time"):
        days_to_simulate = st.slider("Days to Simulate", 1, 30, 7)
        simulate_button = st.button("Run Simulation")

        if simulate_button:
            import pandas as pd
            from datetime import datetime, timedelta
            
            dates = [datetime.now() + timedelta(days=i) for i in range(days_to_simulate)]
            temperatures = []
            for date in dates:
                # Here, you'd need an API or model to predict weather for a future date
                # For simplicity, we'll simulate with random fluctuations:
                simulated_ambient_temp = ambient_temperature + np.random.uniform(-5, 5)
                is_day = date.hour > 6 and date.hour < 18  # Assuming daytime from 6 AM to 6 PM
                
                temp_result = calculate_hive_temperature(params, st.session_state.boxes, simulated_ambient_temp, is_day, altitude)
                temperatures.append(temp_result['base_temperature'])

            df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})
            st.line_chart(df.set_index('Date'))

    # Hive Health Insights
    st.subheader("🐝 Hive Health Insights")
    health_score = 100 - ((abs(results['base_temperature'] - params['ideal_hive_temperature']) / params['ideal_hive_temperature']) * 100)
    st.metric("Hive Health Score", f"{health_score:.2f}%")
    st.markdown("""
    - **Temperature Stability**: Bees prefer a stable temperature close to 35°C. Deviations might stress the colony.
    - **Colony Size**: A larger colony can better regulate temperature but might overheat in small hives.
    - **Material & Design**: Good insulation and hive design are crucial for maintaining ideal conditions.
    """)

    # Community Features
    with st.expander("Share & Community"):
        st.write("Share your hive configuration or check out others:")
        if st.button("Share Config"):
            config_to_share = {
                'params': params,
                'boxes': st.session_state.boxes,
                'altitude': altitude,
                'is_daytime': is_daytime
            }
            # Here, you would typically implement sharing functionality, e.g., saving to a database or generating a shareable link
            st.success("Configuration shared. Link to share: [placeholder for link]")
        
        # Placeholder for community features - would need backend support
        st.write("Community Hive Configurations:")
        st.write("Coming soon...")

    # Documentation and Help
    with st.expander("Documentation & Help"):
        st.markdown("""
        ### How to Use
        - **Input Parameters**: Adjust these to match your hive's conditions.
        - **Box Configuration**: Modify dimensions and cooling effects to see impacts on temperature.
        - **Visualizations**: Use charts to understand trends and impacts of different variables.

        ### Troubleshooting
        - **Invalid Coordinates**: Ensure your input is in the correct format (latitude, longitude).
        - **API Errors**: If weather data isn't loading, check your internet connection or try different coordinates.

        For more help or to report issues, please use the feedback section or contact support.
        """)

# Main execution
if __name__ == "__main__":
    st.session_state.setdefault('initialized', False)
    st.session_state.setdefault('boxes', [])
    st.session_state.setdefault('saved_state', {})
    st.session_state.setdefault('realtime', False)
    
    if not st.session_state.initialized:
        st.session_state.initialized = True
        st.session_state.boxes = [
            {'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': ce}
            for i, ce in enumerate([2, 0, 0, 8])
        ]

    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the input parameters or try refreshing the page.")

            # Historical Data
    st.subheader("📊 Historical Data")
    if st.checkbox("Show Historical Data"):
        # Placeholder for historical data - would require actual data collection over time
        import pandas as pd
        historical_data = pd.DataFrame({
            'date': pd.date_range(start="2023-01-01", periods=100, freq='D'),
            'ambient_temp': np.random.uniform(5, 30, 100),
            'hive_temp': np.random.uniform(30, 38, 100),
            'colony_size': np.random.uniform(10000, 60000, 100)
        })
        
        st.line_chart(historical_data.set_index('date'))

    # Interactive Map for Location Selection
    st.subheader("🌍 Select Location on Map")
    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_data, zoom=10)

    # Hive Design Optimization
    st.subheader("🔧 Hive Design Optimization")
    with st.expander("Optimize Hive Design"):
        objective = st.radio("Optimization Objective", ["Temperature Stability", "Minimal Heat Loss", "Maximize Volume"])
        
        if objective == "Temperature Stability":
            optimization_param = 'wood_thickness'
        elif objective == "Minimal Heat Loss":
            optimization_param = 'wood_thermal_conductivity'
        else:  # Maximize Volume
            optimization_param = 'height'
        
        def objective_function(x):
            if objective == "Temperature Stability":
                return abs(calculate_hive_temperature({**params, optimization_param: x}, st.session_state.boxes, ambient_temperature, is_daytime, altitude)['base_temperature'] - params['ideal_hive_temperature'])
            elif objective == "Minimal Heat Loss":
                return calculate_hive_temperature({**params, optimization_param: x}, st.session_state.boxes, ambient_temperature, is_daytime, altitude)['heat_transfer']
            else:  # Maximize Volume
                return -sum([(3 * math.sqrt(3) / 2) * ((box['width'] / (100 * math.sqrt(3))) ** 2) * (x / 100) for box in st.session_state.boxes])

        from scipy.optimize import minimize_scalar
        
        # Placeholder for optimization result - in real scenarios, you'd need bounds and possibly more sophisticated methods
        result = minimize_scalar(objective_function, bounds=(0.1, 5.0) if optimization_param == 'wood_thickness' else (0.1, 0.3) if optimization_param == 'wood_thermal_conductivity' else (5, 20))
        
        st.write(f"Optimal {optimization_param}: {result.x:.2f}")
        st.write(f"Objective Value: {result.fun:.2f}")
        
        # Update params with the optimized value
        params[optimization_param] = result.x

    # Alerts and Notifications
    st.subheader("🔔 Alerts & Notifications")
    if st.button("Set Alerts"):
        alert_temp = st.slider("Alert if Hive Temperature goes below", 0.0, 35.0, 33.0)
        alert_colony_size = st.slider("Alert if Colony Size drops below", 0, 50000, 20000)
        
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        
        if results['base_temperature'] < alert_temp:
            st.session_state.alerts.append(f"Alert: Hive temperature ({results['base_temperature']:.1f}°C) is below {alert_temp}°C!")
        if results['calculated_colony_size'] < alert_colony_size:
            st.session_state.alerts.append(f"Alert: Colony size ({results['calculated_colony_size']}) is below {alert_colony_size} bees!")
        
        if st.session_state.alerts:
            for alert in st.session_state.alerts:
                st.warning(alert)
            st.session_state.alerts = []

    # Integration with External Tools
    st.subheader("🔗 External Tools Integration")
    with st.expander("Integrate with External Tools"):
        st.write("Coming Soon:")
        st.write("- Sync with weather apps for better forecasting.")
        st.write("- Connect to IoT devices for real hive monitoring.")
        st.write("- Integration with beekeeping management software.")

# Main execution
if __name__ == "__main__":
    st.session_state.setdefault('initialized', False)
    st.session_state.setdefault('boxes', [])
    st.session_state.setdefault('saved_state', {})
    st.session_state.setdefault('realtime', False)
    st.session_state.setdefault('alerts', [])
    
    if not st.session_state.initialized:
        st.session_state.initialized = True
        st.session_state.boxes = [
            {'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': ce}
            for i, ce in enumerate([2, 0, 0, 8])
        ]

    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the input parameters or try refreshing the page.")

            # Hive Management Recommendations
    st.subheader("📝 Management Recommendations")
    recommendations = []
    
    if results['base_temperature'] < params['ideal_hive_temperature'] - 2:
        recommendations.append("Consider insulating the hive further or clustering bees for warmth.")
    elif results['base_temperature'] > params['ideal_hive_temperature'] + 2:
        recommendations.append("Ensure adequate ventilation or apply cooling methods like shading or misting.")
    
    if results['calculated_colony_size'] < 20000:
        recommendations.append("Monitor for signs of disease or absconding. Consider merging hives or introducing new bees.")
    
    if altitude > 2000:
        recommendations.append("At this altitude, oxygen levels are lower; monitor bee activity and health closely.")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.info("No specific recommendations at this time. Continue monitoring hive conditions.")

    # Bee Activity Simulation
    st.subheader("🐝 Bee Activity Simulation")
    with st.expander("Simulate Bee Activity"):
        activity = st.slider("Simulate Bee Activity Level", 0, 100, 50)
        activity_factor = activity / 100.0
        
        # Adjust bee metabolic heat based on activity level
        params['bee_metabolic_heat'] = BEE_METABOLIC_HEAT * (1 + activity_factor * 0.5)  # Increase heat generation with activity
        
        sim_results = calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, is_daytime, altitude)
        
        st.write(f"Simulated Hive Temperature with Activity Level {activity}%: {sim_results['base_temperature']:.2f}°C")
        st.progress(sim_results['colony_metabolic_heat'] / 0.3)  # Assuming max metabolic heat at 0.3 kW for progress bar

    # Educational Quiz
    st.subheader("📚 Educational Quiz")
    with st.expander("Take a Quiz on Beekeeping"):
        questions = [
            {"question": "What is the ideal temperature for a bee hive?", "choices": ["25°C", "35°C", "45°C"], "correct": "35°C"},
            {"question": "Which part of the day do bees typically work harder?", "choices": ["Morning", "Afternoon", "Night"], "correct": "Afternoon"},
            {"question": "What does a high altitude affect in terms of beekeeping?", "choices": ["Bee vision", "Oxygen availability", "Honey production"], "correct": "Oxygen availability"}
        ]
        
        score = 0
        for i, q in enumerate(questions):
            user_answer = st.radio(q['question'], q['choices'], key=f"quiz_{i}")
            if user_answer == q['correct']:
                score += 1
        
        st.write(f"Your Score: {score}/{len(questions)}")
        if score == len(questions):
            st.success("Perfect Score! You're a beekeeping expert!")
        elif score > 0:
            st.info(f"Good job! You got {score} out of {len(questions)} correct.")
        else:
            st.warning("Keep learning! You might want to review some beekeeping basics.")

    # Hive Virtual Tour
    st.subheader("🌐 Virtual Hive Tour")
    with st.expander("Explore a Virtual Hive"):
        st.write("Here's a glimpse inside a typical bee hive:")
        # Placeholder for static images or video content
        st.image("path_to_hive_image.jpg", caption="Inside a Bee Hive", use_column_width=True)
        st.markdown("""
        - **Brood Chamber**: Where the queen lays eggs.
        - **Honey Supers**: Used for storing honey.
        - **Frames**: Provide structure for comb building.
        """)

    # Environmental Impact Section
    st.subheader("🌿 Environmental Impact")
    with st.expander("Learn About Bees and the Environment"):
        st.markdown("""
        - **Pollination**: Bees are vital for pollinating many crops and wild plants.
        - **Biodiversity**: They support a wide range of species in ecosystems.
        - **Climate**: Bee behavior can change with climate shifts, affecting local flora.
        """)

        st.write("Your hive's estimated pollination impact:")
        pollination_impact = results['calculated_colony_size'] * 0.0001  # Example calculation: each bee pollinates 0.0001 km² per day
        st.metric("Pollination Area (km²/day)", f"{pollination_impact:.2f}")

# Main execution
if __name__ == "__main__":
    st.session_state.setdefault('initialized', False)
    st.session_state.setdefault('boxes', [])
    st.session_state.setdefault('saved_state', {})
    st.session_state.setdefault('realtime', False)
    st.session_state.setdefault('alerts', [])
    
    if not st.session_state.initialized:
        st.session_state.initialized = True
        st.session_state.boxes = [
            {'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': ce}
            for i, ce in enumerate([2, 0, 0, 8])
        ]

    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the input parameters or try refreshing the page.")

            # Seasonal Adjustments
    st.subheader("🌸 Seasonal Adjustments")
    with st.expander("Adjust for Season"):
        season = st.selectbox("Select Season", ["Spring", "Summer", "Autumn", "Winter"])
        if season == "Winter":
            params['ideal_hive_temperature'] += 2  # Bees cluster more in winter
            st.info("Winter adjustments: Increased ideal temperature due to clustering.")
        elif season == "Summer":
            for box in st.session_state.boxes:
                box['cooling_effect'] *= 1.1  # More cooling needed in summer
            st.info("Summer adjustments: Enhanced cooling effect for boxes.")
        else:
            st.info(f"Adjustments for {season}: No specific changes needed.")

        # Recalculate with seasonal adjustments
        seasonal_results = calculate_hive_temperature(params, st.session_state.boxes, ambient_temperature, is_daytime, altitude)
        st.metric(f"Hive Temperature for {season}", f"{seasonal_results['base_temperature']:.1f}°C")

    # Resource Management
    st.subheader("🍯 Resource Management")
    with st.expander("Resource Management"):
        nectar_flow = st.slider("Current Nectar Flow (L/day)", 0.0, 10.0, 5.0)
        honey_consumption = results['calculated_colony_size'] * 0.00001  # Example: each bee uses 0.00001L of honey per day
        
        surplus_honey = nectar_flow - honey_consumption
        if surplus_honey > 0:
            st.success(f"Surplus Honey Production: {surplus_honey:.2f} L/day")
        else:
            st.warning(f"Deficit in Honey Production: {abs(surplus_honey):.2f} L/day. Consider supplemental feeding.")

    # Beekeeper's Notes
    st.subheader("📓 Beekeeper's Notes")
    if 'notes' not in st.session_state:
        st.session_state.notes = ""
    
    notes = st.text_area("Add or View Notes", st.session_state.notes, height=150)
    if st.button("Save Notes"):
        st.session_state.notes = notes
        st.success("Notes saved!")

    # Legal and Safety Information
    st.subheader("⚖️ Legal & Safety")
    with st.expander("Legal and Safety Information"):
        st.markdown("""
        - **Regulations**: Check local laws regarding beekeeping, hive placement, and honey production.
        - **Safety**: Use protective gear, be aware of allergies, and manage hive aggression.
        - **Pest Control**: Understand legal methods for controlling hive pests like mites or beetles.
        """)

    # Future Developments
    st.subheader("🔮 Future Developments")
    st.markdown("""
    We're always looking to enhance this tool. Future features might include:
    - **AI Predictions**: Using machine learning for weather and hive condition forecasts.
    - **IoT Integration**: Real-time data collection from sensors in hives.
    - **Augmented Reality**: AR experiences to interact with hive structures virtually.
    - **Community Features**: More interactive community sharing and discussion platforms.
    """)

    # Feedback Loop
    st.subheader("🔄 Feedback Loop")
    st.write("Your feedback helps us improve:")
    feedback = st.text_area("What would you like to see improved or added?", "")
    if st.button("Submit Feedback"):
        st.success("Thank you! Your feedback has been submitted.")
        # Here you'd typically handle sending feedback to a backend or database

# Main execution
if __name__ == "__main__":
    st.session_state.setdefault('initialized', False)
    st.session_state.setdefault('boxes', [])
    st.session_state.setdefault('saved_state', {})
    st.session_state.setdefault('realtime', False)
    st.session_state.setdefault('alerts', [])
    st.session_state.setdefault('notes', "")
    
    if not st.session_state.initialized:
        st.session_state.initialized = True
        st.session_state.boxes = [
            {'id': i+1, 'width': 22, 'height': 9, 'cooling_effect': ce}
            for i, ce in enumerate([2, 0, 0, 8])
        ]

    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the input parameters or try refreshing the page.")

        
