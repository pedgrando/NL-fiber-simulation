import numpy as np

# -------------------------------------------------------------
# ------------------- Modulation Functions --------------------
# -------------------------------------------------------------

def rrc(t, B, rho):
    """
    Computes the Root Raised Cosine (RRC) pulse shape.
    
    The RRC pulse is used for band-limited pulse shaping to satisfy the 
    Nyquist ISI criterion when combined with a matched filter.
    
    Args:
        t (np.ndarray): Time vector.
        B (float): Baud rate (symbol rate). Ts = 1/B.
        rho (float): Roll-off factor (0 <= rho <= 1).
        
    Returns:
        np.ndarray: The RRC pulse samples at times t.
    """
    Ts = 1 / B

    h_rrc = np.zeros_like(t)

    # RRC has 3 cases to avoid division by zero:
    # 1. t = 0
    # 2. t = +- Ts / (4 * rho) (singularities in the denominator)
    # 3. All other t
    
    idx_zero = np.isclose(t, 0)
    
    if rho == 0:
        idx_sing = np.zeros_like(t, dtype=bool)
    else:
        idx_sing = np.isclose(np.abs(t), Ts / (4 * rho))
        
    idx_other = ~(idx_zero | idx_sing)
    
    # Case 1: t = 0
    h_rrc[idx_zero] = 1.0 - rho + (4 * rho / np.pi)
    
    # Case 2: Singular points at t = +- Ts / (4 * rho)
    if rho != 0:
        val_sing = (rho / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rho)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rho)))
        h_rrc[idx_sing] = val_sing
        
    # Case 3: General formula
    t_other = t[idx_other]
    pi_t_Ts = np.pi * t_other / Ts
    num = np.sin(pi_t_Ts * (1 - rho)) + 4 * rho * (t_other / Ts) * np.cos(pi_t_Ts * (1 + rho))
    den = pi_t_Ts * (1 - (4 * rho * t_other / Ts)**2)
    h_rrc[idx_other] = num / den

    return h_rrc

def modul(t, s, B, use_rrc=False, rho=0.0):
    """
    Modulates a sequence of symbols onto a continuous-time signal.
    
    This function implements: q(t) = sum_i s_i * p(t - i/B)
    where p(t) is either a sinc pulse or an RRC pulse.
    
    The pulses are scaled by sqrt(B) to form an orthonormal basis, 
    ensuring that the energy of the signal q(t) is sum|s_i|^2.
    
    Args:
        t (np.ndarray): Time vector.
        s (np.ndarray): Sequence of complex symbols.
        B (float): Bandwidth / Baud rate.
        use_rrc (bool): Whether to use RRC pulse. Default is False (sinc).
        rho (float): Roll-off factor for RRC.
        
    Returns:
        np.ndarray: The modulated complex signal q(t).
    """
    Ns = len(s)
    N = len(t)

    # Initialize complexity-supporting signal array
    qt = np.zeros(N, dtype=complex)

    # Pulse shaping via summation
    # We iterate through each symbol and add its corresponding shifted pulse to the total signal.
    if use_rrc:
        for i in range(int(-np.floor(Ns / 2)), int(np.ceil(Ns / 2))):
            # Scale by sqrt(B) for orthonormal basis property
            qt = qt + s[i + (int(np.floor(Ns / 2)))] * np.sqrt(B) * rrc(t - i / B, B, rho)

    else:
        for i in range(int(-np.floor(Ns / 2)), int(np.ceil(Ns / 2))):
            # Scale by sqrt(B) for orthonormal basis property
            qt = qt + s[i + (int(np.floor(Ns / 2)))] * np.sqrt(B) * np.sinc(B * t - i)

    return qt

# a demodulator simply does a projection over the basis (inner prod -> integral of the product of those signals)
def demod(x, dt, B, Ns, t, use_rrc=False, rho=0.0):
    
    s = []
    
    # check lower limit of range -> may be buggy
    for i in range(int(-np.floor(Ns / 2)), int(np.ceil(Ns / 2))):
        if use_rrc:
            s.append( np.sqrt(B) * sum(x * rrc(t - i / B, B, rho)) * dt)
        else:
            s.append( np.sqrt(B) * sum(x * np.sinc(B * t - i)) * dt)

    return s