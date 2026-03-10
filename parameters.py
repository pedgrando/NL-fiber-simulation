import numpy as np

# -------------------------------------------------------------
# -------- Physical Constants & Fiber Parameters --------------
# -------------------------------------------------------------

# Physical fiber parameters (Standard Single-Mode Fiber)
a_dB    = 0.2               # Attenuation coefficient [dB/km]
D       = 17e-6             # Dispersion parameter D [s/(m·m)] (17 ps/(nm·km))
gamma   = 1.27e-3           # Nonlinearity coefficient [W^-1 m^-1] (1.27 W^-1 km^-1)
n_sp    = 1                 # Spontaneous emission factor
h       = 6.626e-34         # Planck's constant [J·s]
lambda0 = 1.55e-6           # Operating wavelength [m] (C-band)
c       = 3e8               # Speed of light in vacuum [m/s]

# Derived: carrier frequency
f0 = c / lambda0            # [Hz]

# Dispersion coefficient beta2 [s^2/m]: beta2 = -(lambda0^2 / 2*pi*c) * D
beta2 = -(lambda0**2) / (2 * np.pi * c) * D

# Linear loss coefficient alpha [m^-1]
alpha = 1e-4 * np.log(10) * a_dB

# Noise Power Spectral Density sigma0_sq [W/Hz · 1/m] (per unit length)
sigma0_sq = n_sp * h * alpha * f0

# -------------------------------------------------------------
# ------- Normalization / Scale Parameters --------------------
# -------------------------------------------------------------

# Reference scales for dimensionless normalization
L_phys = 1e6                # Total distance (1,000 km)
B_phys = 5e9                # Bandwidth (5 GHz)

# Power scale P0: sets nonlinear coefficient to unity in NLSE
P0 = 2 / (gamma * L_phys)

# Time scale T0: sets dispersion coefficient to unity in NLSE
T0 = np.sqrt(np.abs(beta2) * L_phys / 2)

# Dimensionless distance (0 to 1)
L_norm = 1.0                                

# Dimensionless noise variance (scales physical PSD)
sigma_sq = sigma0_sq * L_phys / (P0 * T0)

# Dimensionless bandwidth
Bn = B_phys * T0 

# Explicit normalized NLSE coefficients
gamma_norm = -gamma * L_phys * P0
beta2_norm = -beta2 * L_phys / (T0 ** 2)
betav_norm = [0, 0, beta2_norm]

# -------------------------------------------------------------
# ------------- Constellation & Power Settings ----------------
# -------------------------------------------------------------

M = 16                      # Default QAM size
P = 1e-3                    # Target optical power [W] (1 mW)

# Normalized power per bandwidth unit
P_norm = P / (P0 * Bn)
power_scale_norm = np.sqrt(P_norm)

if __name__ == "__main__":
    print("=== Physical Parameters ===")
    print(f"  f0          = {f0:.4e} Hz")
    print(f"  beta2       = {beta2:.4e} s²/m")
    print(f"  sigma0_sq   = {sigma0_sq:.4e}")
    print("\n=== Scale Parameters ===")
    print(f"  P0          = {P0:.4e} W")
    print(f"  T0          = {T0:.4e} s")
    print("\n=== Normalized Parameters ===")
    print(f"  sigma_sq    = {sigma_sq:.4e}")
    print(f"  Bn          = {Bn:.4f}")
    print(f"  P_norm      = {P_norm:.4e}")
