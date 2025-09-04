# dryden.py
import numpy as np
import math
from scipy import signal

def _dryden_tf(height, airspeed, axis='u', turbulence_level=30):
    h = height
    V = airspeed

    sigma_w = 0.1 * turbulence_level  # Standard deviation of the wind speed in m/s, scaled by turbulence level
    coeff = (0.177 + 0.000823 * h) # Coefficient based on height
    Lv = h / coeff**0.2 # Length scale of the turbulence in m
    sigma = sigma_w / coeff**0.4 # Standard deviation of the wind speed in m/s, adjusted by the coefficient

    if axis == 'u':
        num = [sigma * np.sqrt(2 * Lv / (np.pi * V)) * V]
        den = [Lv, V]
    else:  # 'v' or 'w'
        c = sigma * np.sqrt(Lv / (np.pi * V))
        Lv_V = Lv / V
        num = [np.sqrt(3)*Lv_V*c, c]
        den = [Lv_V**2, 2*Lv_V, 1]

    return signal.TransferFunction(num, den)

def dryden_response(axis, height=100, airspeed=10, turbulence_level=30, time_steps=1000, seed=42):
    tf = _dryden_tf(height, airspeed, axis, turbulence_level)
    t = np.linspace(0, 10, time_steps)  # Time vector for simulation
    # Fix np random seed for reproducibility
    if seed:
        np.random.seed(seed)
    U = np.random.randn(len(t))  # Random input signal
    T, y, _ = signal.lsim(tf, U, t) # y is the wind speed in m/s
    return y



