import numpy as np

# -------------------------------------------------------------
# -------- Physical Constants & Fiber Parameters --------------
# -------------------------------------------------------------

# Table 1: Fiber and Noise Parameters
a_dB    = 0.2               # Fiber loss in dB/km (Standard single-mode fiber)
D       = 17e-6             # Chromatic dispersion D [s/(m·m)] -- 17 ps/(nm·km)
gamma   = 1.27e-3           # Nonlinearity parameter [W^-1 m^-1] -- 1.27 W^-1 km^-1
n_sp    = 1                 # Spontaneous emission factor for optical amplifiers
h       = 6.626e-34         # Planck's constant [J·s]
lambda0 = 1.55e-6           # Carrier wavelength [m] (Standard C-band)
c       = 3e8               # Speed of light in vacuum [m/s]

# Derived: carrier frequency f0 = c / lambda
f0 = c / lambda0            # [Hz]

# Dispersion coefficient beta2 [s^2/m]:
# Relationship: beta2 = -(lambda^2 / 2*pi*c) * D
beta2 = -(lambda0**2) / (2 * np.pi * c) * D    # [s²/m]

# Loss coefficient alpha [m^-1]:
# Convert dB/km to nepers/m: alpha_linear = (ln(10)/10) * alpha_dB / 1000
alpha = 1e-4 * np.log(10) * a_dB               # [m^-1]

# Noise Power Spectral Density sigma0_sq [W/Hz · 1/m]:
# This represents the ASE (Amplified Spontaneous Emission) noise PSD added per unit length.
sigma0_sq = n_sp * h * alpha * f0               # [W/Hz · 1/m] → [W·s/m]

# -------------------------------------------------------------
# ------- Normalization / Scale Parameters --------------------
# -------------------------------------------------------------

L_phys = 1e6             # Total physical propagation distance (10,000 km)
B_phys = 5e9                # Physical bandwidth for simulation (5 GHz)

# Scale parameters for dimensionless NLSE normalization (Equation 3)
# Normalization: L_norm = L_phys, T_norm = T0, P_norm = P0
# These scales simplify the NLSE to: dU/dz = -j*sgn(beta2)*d^2U/dt^2 + j*|U|^2*U

# Power scale P0: Chosen to make the nonlinear term coefficient unity.
P0 = 2 / (gamma * L_phys)                      # [W]

# Time scale T0: Chosen to make the dispersion term coefficient unity (ignoring sgn).
T0 = np.sqrt(np.abs(beta2) * L_phys / 2)       # [s]

# Normalized distance z is now distance / L_phys (z ∈ [0, 1])
L_norm = L_phys                                

# Normalized noise variance sigma_sq:
# Scales the physical noise PSD sigma0_sq into the dimensionless domain.
sigma_sq = sigma0_sq * L_phys / (P0 * T0)

# Normalized bandwidth Bn:
Bn = B_phys * T0                                # Normalized bandwidth

# Normalized gamm
gamma_norm = -gamma*L_phys*P0

# Normlaized beta2
beta2_norm = -beta2*L_phys/(T0**2)
betav_norm = [0, 0, beta2_norm]

# -------------------------------------------------------------
# ------------- Constellation Parameters ----------------------
# -------------------------------------------------------------

M = 16                      # M-QAM constellation size
P = 1e-3                    # Target average physical optical power [W] (1 mW)

# To scale unit-power symbols to E|s|² = P, multiply by √P
power_scale = np.sqrt(P)

# In the normalized system, all powers are divided by P0.
# The target normalized power constraint is E|s|² = P / (P0*Bn).
P_norm = P / (P0*Bn)
power_scale_norm = np.sqrt(P_norm)

# -------------------------------------------------------------
# ---------------------- Check Print --------------------------
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=== Physical Parameters ===")
    print(f"  f0          = {f0:.4e} Hz")
    print(f"  β₂          = {beta2:.4e} s²/m")
    print(f"  α           = {alpha:.4e} m⁻¹")
    print(f"  σ₀²         = {sigma0_sq:.4e}")
    print()
    print("=== Scale Parameters ===")
    print(f"  P₀          = {P0:.4e} W")
    print(f"  T₀          = {T0:.4e} s")
    print(f"  L           = {L_phys:.0f} m")
    print()
    print("=== Normalized Parameters ===")
    print(f"  σ²          = {sigma_sq:.4e}")
    print(f"  Bₙ          = {Bn:.4f}")
    print(f"  P_norm      = {P_norm:.4e}")
