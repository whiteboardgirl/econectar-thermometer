# visualization.py
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from typing import List

def create_temperature_chart(box_temperatures: List[float]) -> str:
    """Create temperature distribution chart."""
    fig, ax = plt.subplots()
    ax.bar([f'Box {i+1}' for i in range(len(box_temperatures))], box_temperatures)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Distribution Across Hive Boxes')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def create_altitude_chart(altitude_range: np.ndarray, temperatures: List[float]) -> str:
    """Create altitude effect visualization."""
    fig, ax = plt.subplots()
    ax.plot(altitude_range, temperatures)
    ax.set_title("Hive Temperature vs. Altitude")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Hive Temperature (°C)")
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
