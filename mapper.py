import numpy as np

# -------------------------------------------------------------
# ------------------ Mapping Functions ------------------------
# -------------------------------------------------------------

def pam_gray_mapping(m):
    """
    Generates a 1-D Pulse Amplitude Modulation (PAM) Gray-coded mapping.
    
    The mapping assigns Gray-coded integers to equi-spaced amplitude levels.
    Gray coding ensures that adjacent symbols differ by only one bit, which 
    minimizes bit error rate for a given symbol error rate.
    
    Args:
        m (int): Number of levels in the PAM constellation.
        
    Returns:
        np.ndarray: A Look-Up Table (LUT) where index is the gray code and value is the amplitude.
    """
    if m == 1:
        return np.array([0.0])
    
    # 1. Define equi-spaced levels symmetric around zero.
    # For m=2: [-1, 1], For m=4: [-3, -1, 1, 3]
    levels = 2 * np.arange(m) - (m - 1)
    
    # 2. Generate Gray codes for integers 0 to m-1.
    # Gray code of n is: n ^ (n >> 1)
    gray = np.arange(m) ^ (np.arange(m) >> 1)
    
    # 3. Create a dictionary mapping gray code to amplitude.
    mapping = dict(zip(gray, levels))

    # 4. Construct the LUT such that lut[gray_code] = amplitude.
    lut = np.zeros(m)
    for g, level in mapping.items():
        lut[g] = level
    return lut

def generate_constellation(const):
    """
    Generates a 2-D QAM constellation with Gray (or quasi-Gray) mapping.
    
    This function handles:
    - Square QAM (4, 16, 64...): Generated as a product of two Gray-coded PAM mappings.
    - BPSK (2): Single-dimension mapping.
    - Cross QAM (8, 32): Hardcoded optimal quasi-Gray configurations.
    
    Args:
        const (int): Number of points in the constellation (M).
        
    Returns:
        np.ndarray: Complex array of constellation points.
    """
    k = int(np.log2(const))
    if 2**k != const:
        raise ValueError("const must be a power of 2 (2, 4, 8, 16, 32, ...)")
        
    if const == 8:
        # Standard Cross 8-QAM. 
        # Note: Perfect Gray mapping isn't possible for cross QAM. 
        # These indices are optimized for low average Hamming distance.
        return np.array([
            -3.0+1.0j, -3.0-1.0j, -1.0+3.0j, 1.0+3.0j,
            1.0-3.0j, -1.0-3.0j, 3.0-1.0j, 3.0+1.0j,
        ])
        
    if const == 32:
        # Standard Cross 32-QAM (6x6 square with 4 corners removed).
        # Quasi-Gray mapped via simulated annealing to minimize bit-error penalties.
        return np.array([
            3.0+5.0j, 3.0+3.0j, 1.0+5.0j, 1.0+3.0j,
            3.0-5.0j, -3.0+5.0j, 1.0-5.0j, -1.0+5.0j,
            5.0+3.0j, -3.0+1.0j, -1.0-1.0j, -1.0+1.0j,
            -1.0-5.0j, -3.0+3.0j, -1.0-3.0j, -1.0+3.0j,
            5.0-1.0j, 3.0+1.0j, 5.0-3.0j, 1.0+1.0j,
            3.0-3.0j, 3.0-1.0j, 1.0-3.0j, 1.0-1.0j,
            5.0+1.0j, -5.0+1.0j, -3.0-1.0j, -5.0-1.0j,
            -3.0-5.0j, -5.0+3.0j, -3.0-3.0j, -5.0-3.0j,
        ])
    
    # Generic square-like PAM product mapping.
    # Splits k bits into k_I and k_Q bits for the In-phase and Quadrature components.
    k_I = int(np.ceil(k / 2))
    k_Q = int(np.floor(k / 2))
    
    m_I = 2**k_I
    m_Q = 2**k_Q
    
    lut_I = pam_gray_mapping(m_I)
    lut_Q = pam_gray_mapping(m_Q)
    
    constellation = np.zeros(const, dtype=complex)
    for i in range(const):
        # Index i is split: upper bits for I, lower bits for Q.
        i_I = i >> k_Q
        i_Q = i & ((1 << k_Q) - 1)
        constellation[i] = lut_I[i_I] +  1j * lut_Q[i_Q]
        
    return constellation

def mapper(bits, const=16, normalize=True):
    """
    Maps a bit sequence into complex QAM symbols.
    
    Args:
        bits (array_like): Binary sequence (0s and 1s).
        const (int): Constellation size (M). Default is 16.
        normalize (bool): If True, scales symbols to have unit average power (E[|s|^2]=1).
        
    Returns:
        np.ndarray: Array of complex QAM symbols.
    """
    k = int(np.log2(const))

    bits = np.asarray(bits)
    if len(bits) % k != 0:
        raise ValueError("Bit sequence should be multiple of log2(const)")

    # 1. Group bits into symbols of size k.
    symbol_bits = bits.reshape(-1, k)
    
    # 2. Convert bit groups to integer indices.
    powers = 2**np.arange(k-1, -1, -1)
    indices = symbol_bits @ powers

    # 3. Map indices to complex points using the constellation LUT.
    constellation = generate_constellation(const)
    s = constellation[indices]

    # 4. Normalize if requested (divide by sqrt of average power).
    if normalize:
        P_avg = np.mean(np.abs(constellation)**2)
        s = s / np.sqrt(P_avg)

    return s

def demap_decode(s_hat, const=16, normalize=True):
    """
    Performs Maximum Likelihood (ML) detection and demapping of QAM symbols back to bits.
    
    Args:
        s_hat (array_like): Received complex symbols.
        const (int): Constellation size (M).
        normalize (bool): Must match the 'normalize' flag used in mapper().
        
    Returns:
        tuple: (detected_symbols, detected_bits)
    """
    k = int(np.log2(const))
    constellation = generate_constellation(const)
    
    if normalize:
        P_avg = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(P_avg)

    # 1. ML detection: Find the nearest neighbour in the constellation.
    s_hat = np.asarray(s_hat)
    # Calculate all-to-all Euclidean distances squared.
    dists = np.abs(s_hat[:, None] - constellation[None, :]) ** 2
    detected_idx = np.argmin(dists, axis=1)

    detected_symbols = constellation[detected_idx]

    # 2. Demapping: Recover bits from the detected symbol indices.
    bits = np.zeros(len(s_hat) * k, dtype=int)
    for i in range(k):
        # Extract bits using bitwise shifts and masking.
        bits[i::k] = (detected_idx >> (k - 1 - i)) & 1

    return detected_symbols, bits
