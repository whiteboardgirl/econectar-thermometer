# config.py
from dataclasses import dataclass
from typing import Dict, Any, List
import math

@dataclass
class ThermalConfig:
    """Configuration constants for thermal calculations."""
    KELVIN_CONVERSION: float = 273.15
    AIR_FILM_RESISTANCE_OUTSIDE: float = 0.04  # m²K/W
    BEE_METABOLIC_HEAT: float = 0.0040  # Watts per bee
    IDEAL_HIVE_TEMPERATURE: float = 35.0  # °C
    MIN_OXYGEN_FACTOR: float = 0.6
    ATMOSPHERIC_SCALE_HEIGHT: float = 7400  # m
    STD_ATMOSPHERIC_PRESSURE: float = 1013.25  # hPa

@dataclass
class Box:
    """Represents a single hive box."""
    id: int
    width: float
    height: float
    cooling_effect: float
