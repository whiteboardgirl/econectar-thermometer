import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# Function to fetch weather data from Open-Meteo API
@st.cache_data
def get_weather_data(lat, lon, start_date, end_date):
    """
    Fetch hourly temperature data from Open-Meteo API for a given location and date range.
    
    Args:
        lat (float): Latitude of the location
        lon (float): Longitude of the location
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: DataFrame with time and external temperature data, or None if the request fails
    """
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        times = data['hourly']['time']
        temperatures = data['hourly']['temperature_2m']
        df = pd.DataFrame({'time': times, 'temperature': temperatures})
        df['time'] = pd.to_datetime(df['time'])
        return df
    except requests.RequestException as e:
        st.error(f"Failed to fetch weather data: {e}")
        return None

# Streamlit app setup
st.title("Hive Temperature Calculator")
st.write("""
This app estimates the internal temperature of high-tech beehives based on local weather data and the cooling effect of the hive roof. 
Designed for melipolicultores, it helps assess how temperature reductions can benefit native bee longevity.
""")

# User input section
st.subheader("Input Parameters")
lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=4.0, help="Enter the latitude of the hive location (e.g., 4.0 for Bogotá, Colombia).")
lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=72.0, help="Enter the longitude of the hive location (e.g., 72.0 for Bogotá, Colombia).")
start_date = st.date_input("Start Date", value=date(2023, 1, 1), help="Select the start date for weather data.")
end_date = st.date_input("End Date", value=date(2023, 1, 31), help="Select the end date for weather data.")
k = st.slider("Thermal Transfer Factor (k)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Factor determining how much roof cooling affects internal temperature (0 to 1).")
delta_T_roof = st.slider("Roof Temperature Reduction (°C)", min_value=0.0, max_value=10.0, value=8.5, step=0.5, help="Temperature reduction provided by the hive roof (typically 8-9°C).")

# Calculate and display results when the button is clicked
if st.button("Calculate"):
    # Fetch weather data
    weather_df = get_weather_data(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    if weather_df is not None:
        # Calculate internal hive temperature
        weather_df['internal_temperature'] = weather_df['temperature'] - k * delta_T_roof
        
        # Display data table
        st.subheader("Results")
        st.write("#### Weather Data and Calculated Internal Temperature")
        st.dataframe(weather_df.style.format({"temperature": "{:.2f}", "internal_temperature": "{:.2f}"}))
        
        # Plot external vs. internal temperatures
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
        
        # Summary statistics
        st.write("#### Summary Statistics")
        avg_external = weather_df['temperature'].mean()
        avg_internal = weather_df['internal_temperature'].mean()
        st.write(f"**Average External Temperature:** {avg_external:.2f} °C")
        st.write(f"**Average Internal Temperature:** {avg_internal:.2f} °C")
        st.write(f"**Average Temperature Reduction:** {avg_external - avg_internal:.2f} °C")
    else:
        st.write("No data available. Please check your inputs and try again.")

# Add a disclaimer
st.write("""
*Note: This is a simplified model assuming a linear relationship between roof cooling and internal temperature. 
Actual hive temperatures may vary due to bee activity, insulation, and other environmental factors.*
""")
