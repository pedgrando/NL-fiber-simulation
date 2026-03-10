import numpy as np
import math

# -------------------------------------------------------------
# ---------------- Channel Functions --------------------------
# -------------------------------------------------------------

# takes q(t,z') and outputs q(t,z'+z)
def lin_step(qt, f, beta2, z):
    """
    Applies the linear dispersive step of the fiber channel.
    
    Expects f in natural frequency order (np.fft.fftfreq).
    """
    # Angular frequency w = 2*pi*f
    w = 2 * np.pi * f
    if np.isscalar(beta2):
        H = np.exp(1j * beta2 * (w ** 2) * z / 2)
    else:
        # beta2 is the full dispersion polynomial beta(omega)
        H = np.exp(1j * beta2 * z)
    return np.fft.ifft(np.fft.fft(qt) * H)

# takes q(t,z') and outputs q(t,z'+z)
def non_lin_step(qt, gamma, z):
    """
    Applies the nonlinear phase shift (Self-Phase Modulation) step of the fiber.
    
    This solves the nonlinear part of the NLSE:
    dq/dz = j * gamma * |q|^2 * q
    
    Solution: q(z+dz, t) = q(z, t) * exp(j * gamma * |q|^2 * dz)
    
    Args:
        qt (np.ndarray): Input complex signal.
        gamma (float): Nonlinearity coefficient.
        z (float): Propagation distance.
        
    Returns:
        np.ndarray: Signal after nonlinear phase shift.
    """
    return qt * np.exp(1j * gamma * z * (np.abs(qt)**2))


def _build_beta_w(betav, w):
    """
    Evaluates the dispersion polynomial  beta(omega)  from a coefficient
    vector betav = [beta_0, beta_1, beta_2, ...].

        beta(omega) = sum_{m=0}^{M} beta_m * omega^m / m!
    """
    beta_w = np.zeros_like(w, dtype=float)
    for m, coeff in enumerate(betav):
        beta_w = beta_w + coeff * (w ** m) / math.factorial(m)
    return beta_w

# -------------------------------------------------------------
# ---------------- Non Linear Channel -------------------------
# -------------------------------------------------------------


# non linear channel simulation
def nl_channel(t, qt, z, Nsteps, gamma, sigma2, betav, f, B):
    """
    Simulates nonlinear fiber propagation using the Split-Step Fourier Method (SSFM).
    
    The SSFM alternates between linear steps (in frequency domain) and 
    nonlinear steps (in time domain) to solve the NLSE over distance L.
    
    Args:
        L (float): Total propagation distance.
        Nsteps (int): Number of small steps dz to take.
        gamma (float): Nonlinearity coefficient.
        beta2 (float): Dispersion coefficient.
        qt (np.ndarray): Input complex signal.
        t (np.ndarray): Time vector.
        f (np.ndarray): Frequency vector.
        
    Returns:
        np.ndarray: Propagated complex signal.
    """
    dt = t[1] - t[0]
    dz = z / Nsteps
    N = len(t)
    w = 2 * np.pi * f

    beta_w = _build_beta_w(betav, w)

    for i in range(0, Nsteps):
        # 1. Dispersive step
        qt = lin_step(qt, f, beta_w, dz)

        # 2. Nonlinear step
        qt = non_lin_step(qt, gamma, dz)

        # 3. Add noise
        # Generate bandlimited noise in the frequency domain
        noise_f = np.zeros(N, dtype=complex)
        band_mask = (np.abs(f) <= B / 2).astype(bool)
        n_in_band = np.sum(band_mask)
        if n_in_band > 0:
            # Variance per frequency bin: N * sigma2 * dz / dt
            # (so that the time‑domain variance becomes sigma2 * dz * B)
            scale = np.sqrt(N * sigma2 * dz / (2 * dt))
            noise_f[band_mask] = scale * (np.random.randn(int(n_in_band)) + 1j * np.random.randn(int(n_in_band)))
        
        # Convert to time domain and add
        noise_t = np.fft.ifft(noise_f)
        qt = qt + noise_t

    qf = np.fft.fft(qt)

    return qt, qf

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------



# -------------------------------------------------------------
# ------------------ Linear Channel ---------------------------
# -------------------------------------------------------------

def lin_channel(qt, f, beta2, z, sigma2, B):
    """
    Models a purely linear fiber channel with Additive White Gaussian Noise (AWGN).
    
    Expects f in natural order (np.fft.fftfreq).
    """
    N = len(qt)
    w = 2 * np.pi * f

    # 1. Linear Propagation (Dispersion)
    q0f = np.fft.fft(qt)
    qzf = q0f * np.exp(1j * beta2 * (w**2) * z / 2)

    # 2. Noise Addition
    # Standardize frequency order for masking (f is natural order)
    band_mask = (np.abs(f) <= B / 2).astype(float)

    # df is the frequency resolution
    df = f[1] - f[0] if len(f) > 1 else 0
    # Sampling interval dt = 1 / (N * abs(df)) fallback if f is ffreq
    dt = 1 / (N * np.abs(f[1] - f[0])) if len(f) > 1 else 1.0
    
    # Generate complex Gaussian noise. 
    # Scaling factor ensures mean power of demodulated noise is sigma2.
    noise_f = np.sqrt(sigma2 * z * N / (2 * dt)) * (np.random.randn(N) + 1j * np.random.randn(N)) * band_mask

    # Add noise in frequency domain
    qzf = qzf + noise_f
    # Return to time domain
    qzt = np.fft.ifft(qzf)

    return qzt, qzf


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------


# -------------------------------------------------------------
# ---------------- Equalization Functions ---------------------
# -------------------------------------------------------------

def lin_equalizer(qt, t, f, beta2, L):
    """
    Compensates for linear dispersion by applying the inverse transfer function.
    
    Expects f in natural order.
    """
    # 1. Move to frequency domain
    qzf = np.fft.fft(qt)
    w = 2 * np.pi * f
    
    # 2. Apply inverse transfer function (conjugate of dispersion filter)
    qzfe = qzf * np.exp(-1j * beta2 * (w**2) * L / 2)

    # 3. Return to time domain
    return np.fft.ifft(qzfe), qzfe

    
def dbp_equalizer(t, qtL, z, Nsteps, gamma, betav, f, B):
    """
    Digital Back-Propagation (DBP) equalizer.

    Reverses the NLSE by negating dispersion (beta_w -> -beta_w) and
    nonlinearity (gamma -> -gamma). No noise is added during back-propagation.
    """
    w      = 2 * np.pi * f
    dz     = z / Nsteps
    beta_w = _build_beta_w(betav, w)

    for _ in range(Nsteps):
        qtL = lin_step(qtL, f, -beta_w, dz)
        qtL = non_lin_step(qtL, -gamma, dz)

    qf = np.fft.fft(qtL)
    return qf, qtL