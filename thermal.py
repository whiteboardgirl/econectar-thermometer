# thermal.py
import math
from typing import Dict, List, Any
from config import ThermalConfig, Box

def calculate_oxygen_factor(altitude_m: float) -> float:
    """Calculate oxygen factor based on altitude."""
    pressure_ratio = math.exp(-altitude_m / ThermalConfig.ATMOSPHERIC_SCALE_HEIGHT)
    return max(ThermalConfig.MIN_OXYGEN_FACTOR, pressure_ratio)

def calculate_box_surface_area(width_cm: float, height_cm: float) -> float:
    """Calculate surface area for a hexagonal box in square meters."""
    width_m, height_m = width_cm / 100, height_cm / 100
    side_length = width_m / math.sqrt(3)
    hexagon_area = (3 * math.sqrt(3) / 2) * (side_length ** 2)
    sides_area = 6 * side_length * height_m
    return (2 * hexagon_area) + sides_area

def calculate_hive_temperature(params: Dict[str, float], boxes: List[Box], 
                             ambient_temp_c: float, is_daytime: bool, 
                             altitude: float) -> Dict[str, Any]:
    """Calculate hive temperature with all environmental factors."""
    # Time of day adjustments
    if is_daytime:
        params['ideal_hive_temperature'] += 1.0
        params['bee_metabolic_heat'] *= 1.1
    else:
        params['ideal_hive_temperature'] -= 0.5
        params['air_film_resistance_outside'] *= 1.1

    # Altitude adjustments
    oxygen_factor = calculate_oxygen_factor(altitude)
    params['bee_metabolic_heat'] *= oxygen_factor
    params['air_film_resistance_outside'] *= (1 + (altitude / 1000) * 0.05)
    params['ideal_hive_temperature'] -= (altitude / 1000) * 0.5

    # Basic calculations
    calculated_colony_size = 50000 * (params['colony_size'] / 100)
    colony_metabolic_heat = calculated_colony_size * params['bee_metabolic_heat'] * oxygen_factor

    # Volume and surface calculations
    total_volume = sum(
        (3 * math.sqrt(3) / 2) * ((box.width / (100 * math.sqrt(3))) ** 2) * (box.height / 100)
        for box in boxes
    )
    total_surface_area = sum(calculate_box_surface_area(box.width, box.height) for box in boxes)

    # Thermal resistance calculations
    wood_resistance = (params['wood_thickness'] / 100) / params['wood_thermal_conductivity']
    total_resistance = wood_resistance + params['air_film_resistance_outside']

    # Temperature calculations
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

    # Final adjustments
    estimated_temp_c -= (altitude / 1000) * 0.5
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
        'ambient_temperature': ambient_temp_c,
        'oxygen_factor': oxygen_factor,
        'heat_transfer': (total_surface_area * abs(estimated_temp_c - ambient_temp_c)) / total_resistance / 1000
    }

# weather.py
import requests
from typing import Optional
import streamlit as st

def get_temperature_from_coordinates(lat: float, lon: float) -> Optional[float]:
    """Retrieve temperature data from coordinates."""
    url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['current_weather']['temperature']
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        st.write(f"Debug Info - URL: {url}")
        return None
