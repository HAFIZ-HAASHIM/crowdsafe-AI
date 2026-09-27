import pytest
import numpy as np

def calculate_density_metric(detected_people_count, zone_area_sqm):
    """
    Computes crowd density in persons per square meter and returns safety status.
    """
    if zone_area_sqm <= 0:
        raise ValueError("Zone area must be greater than zero.")
    
    density = detected_people_count / float(zone_area_sqm)
    
    if density < 1.0:
        status = "SAFE"
    elif density < 2.5:
        status = "MODERATE"
    elif density < 4.0:
        status = "WARNING"
    else:
        status = "CRITICAL_OVERCROWDING"
        
    return density, status

def test_safe_density():
    density, status = calculate_density_metric(10, 20)
    assert density == 0.5
    assert status == "SAFE"

def test_moderate_density():
    density, status = calculate_density_metric(30, 15)
    assert density == 2.0
    assert status == "MODERATE"

def test_critical_density_alert():
    density, status = calculate_density_metric(100, 20)
    assert density == 5.0
    assert status == "CRITICAL_OVERCROWDING"

def test_invalid_area():
    with pytest.raises(ValueError):
        calculate_density_metric(50, 0)
