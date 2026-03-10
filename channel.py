import numpy as np
import math

# -------------------------------------------------------------
# ---------------- Channel Functions --------------------------
# -------------------------------------------------------------

def lin_step(qt, f, beta2, z):
    """
    Applies the linear dispersive step of the fiber channel in the frequency domain.
    
    This function solves the linear part of the Nonlinear Schrödinger Equation (NLSE):
    dq/dz = -j * (beta2/2) * d^2q/dt^2
    
    Args:
        qt (np.ndarray): Input signal in time domain.
        f (np.ndarray): Frequency vector (natural order: np.fft.fftfreq).
        beta2 (float or np.ndarray): Dispersion coefficient or full beta(omega) polynomial.
        z (float): Propagation distance step dz.
        
    Returns:
        np.ndarray: Signal after linear propagation.
    """
    w = 2 * np.pi * f
    if np.isscalar(beta2):
        # Using a single beta2 coefficient
        H = np.exp(1j * beta2 * (w ** 2) * z / 2)
    else:
        # Using the full dispersion polynomial beta(omega)
        H = np.exp(1j * beta2 * z)
    return np.fft.ifft(np.fft.fft(qt) * H)

def non_lin_step(qt, gamma, z):
    """
    Applies the nonlinear phase shift (Self-Phase Modulation - SPM) in time domain.
    
    This solves the nonlinear part of the NLSE:
    dq/dz = j * gamma * |q|^2 * q
    
    Args:
        qt (np.ndarray): Input complex signal.
        gamma (float): Nonlinearity coefficient.
        z (float): Propagation distance dz.
        
    Returns:
        np.ndarray: Signal after nonlinear phase shift.
    """
    return qt * np.exp(1j * gamma * z * (np.abs(qt)**2))

def _build_beta_w(betav, w):
    """
    Evaluates the dispersion polynomial beta(omega) from a vector of coefficients.
    
    beta(omega) = sum_{m=0}^{M} beta_m * omega^m / m!
    
    Args:
        betav (list): Coefficients [beta0, beta1, beta2, beta3...].
        w (np.ndarray): Angular frequency vector (omega = 2*pi*f).
        
    Returns:
        np.ndarray: Evaluated beta(omega) at each frequency.
    """
    beta_w = np.zeros_like(w, dtype=float)
    for m, coeff in enumerate(betav):
        beta_w = beta_w + coeff * (w ** m) / math.factorial(m)
    return beta_w

# -------------------------------------------------------------
# ---------------- Non Linear Channel -------------------------
# -------------------------------------------------------------

def nl_channel(t, qt, z, Nsteps, gamma, sigma2, betav, f, B):
    """
    Simulates nonlinear fiber propagation using the Split-Step Fourier Method (SSFM).
    
    The SSFM alternates between linear steps (frequency domain) and 
    nonlinear steps (time domain) to solve the NLSE over distance z.
    Distributed noise is added at each step.
    
    Args:
        t (np.ndarray): Time vector.
        qt (np.ndarray): Input time-domain signal.
        z (float): Total propagation distance.
        Nsteps (int): Number of steps for SSFM integration.
        gamma (float): Nonlinearity coefficient.
        sigma2 (float): Noise power spectral density.
        betav (list): Dispersion coefficients [beta0, beta1, beta2...].
        f (np.ndarray): Frequency vector (natural order).
        B (float): Bandwidth for noise filtering.
        
    Returns:
        tuple: (qt, qf) time and frequency domain signals after propagation.
    """
    dt = t[1] - t[0]
    dz = z / Nsteps
    N = len(t)
    w = 2 * np.pi * f

    # Build the dispersion polynomial beta(omega)
    beta_w = _build_beta_w(betav, w)

    for _ in range(Nsteps):
        # 1. Linear (Dispersive) step
        qt = lin_step(qt, f, beta_w, dz)

        # 2. Nonlinear step (Kerr effect)
        qt = non_lin_step(qt, gamma, dz)

        # 3. Add distributed noise
        noise_f = np.zeros(N, dtype=complex)
        band_mask = (np.abs(f) <= B / 2).astype(bool)
        n_in_band = np.sum(band_mask)
        if n_in_band > 0:
            # Scale factor for frequency-domain noise variance
            scale = np.sqrt(N * sigma2 * dz / (2 * dt))
            noise_f[band_mask] = scale * (np.random.randn(int(n_in_band)) + 1j * np.random.randn(int(n_in_band)))
        
        qt = qt + np.fft.ifft(noise_f)

    qf = np.fft.fft(qt)
    return qt, qf

# -------------------------------------------------------------
# ------------------ Linear Channel ---------------------------
# -------------------------------------------------------------

def lin_channel(qt, f, beta2, z, sigma2, B):
    """
    Models a purely linear fiber channel with dispersion and AWGN (noiseless linear NLSE).
    
    Args:
        qt (np.ndarray): Input time-domain signal.
        f (np.ndarray): Frequency vector (natural order).
        beta2 (float): Second-order dispersion coefficient.
        z (float): Fiber length.
        sigma2 (float): Noise power spectral density.
        B (float): Channel bandwidth.
        
    Returns:
        tuple: (qzt, qzf) time and frequency domain signals.
    """
    N = len(qt)
    w = 2 * np.pi * f

    # 1. Linear Propagation
    q0f = np.fft.fft(qt)
    qzf = q0f * np.exp(1j * beta2 * (w**2) * z / 2)

    # 2. Add AWGN
    band_mask = (np.abs(f) <= B / 2).astype(float)
    dt = 1 / (N * np.abs(f[1] - f[0])) if len(f) > 1 else 1.0
    
    # Scale factor ensures mean power of demodulated noise matches sigma2
    noise_f = np.sqrt(sigma2 * z * N / (2 * dt)) * (np.random.randn(N) + 1j * np.random.randn(N)) * band_mask

    qzf = qzf + noise_f
    qzt = np.fft.ifft(qzf)

    return qzt, qzf

# -------------------------------------------------------------
# ---------------- Equalization Functions ---------------------
# -------------------------------------------------------------

def lin_equalizer(qt, t, f, beta2, L):
    """
    Compensates for linear dispersion by applying the inverse transfer function.
    
    Args:
        qt (np.ndarray): Received signal in time domain.
        t (np.ndarray): Time vector.
        f (np.ndarray): Frequency vector (natural order).
        beta2 (float): Dispersion coefficient.
        L (float): Length of fiber to compensate.
        
    Returns:
        tuple: (qzt_eq, qzf_eq) signals after dispersion compensation.
    """
    qzf = np.fft.fft(qt)
    w = 2 * np.pi * f
    
    # Inverse dispersion filter
    qzfe = qzf * np.exp(-1j * beta2 * (w**2) * L / 2)

    return np.fft.ifft(qzfe), qzfe

def dbp_equalizer(t, qtL, z, Nsteps, gamma, betav, f, B):
    """
    Digital Back-Propagation (DBP) equalizer.
    
    Compensates for both dispersion and nonlinearity by solving the 
    NLSE with negated coefficients (-beta, -gamma) over distance z.

    Args:
        t (np.ndarray): Time vector.
        qtL (np.ndarray): Received signal at z=L.
        z (float): Fiber length to back-propagate.
        Nsteps (int): Number of SSFM steps.
        gamma (float): Nonlinearity coefficient.
        betav (list): Dispersion coefficients [beta0, beta1, beta2...].
        f (np.ndarray): Frequency vector.
        B (float): Bandwidth for filtering.
        
    Returns:
        tuple: (qt_eq, qf_eq) signals after DBP.
    """
    w      = 2 * np.pi * f
    dz     = z / Nsteps
    beta_w = _build_beta_w(betav, w)

    for _ in range(Nsteps):
        # Reverse order: Non-Linear then Linear step with negated coefficients
        qtL = non_lin_step(qtL, -gamma, dz)
        qtL = lin_step(qtL, f, -beta_w, dz)

    qf = np.fft.fft(qtL)
    return qtL, qf