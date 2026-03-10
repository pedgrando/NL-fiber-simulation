import numpy as np

# -------------------------------------------------------------
# ------------------- Source Functions ------------------------
# -------------------------------------------------------------

def source_bernoulli(N, p):
    """
    Generates a sequence of i.i.d. Bernoulli bits.
    
    Bits are 1 with probability (1 - p) and 0 with probability p.
    This corresponds to the mapper's expectation of bit distributions.

    Args:
        N (int): Number of bits to generate.
        p (float): Bernoulli parameter (threshold for random generation).
        
    Returns:
        np.ndarray: Array of N binary integers (0s and 1s).
    """
    return np.where(np.random.rand(N) > p, 1, 0)

def source_prbs(N):
    """
    Generates a PRBS15 sequence using a Linear Feedback Shift Register (LFSR).
    
    This is useful for deterministic testing of the communication system.
    Generator polynomial: 1 + x^14 + x^15.
    
    Args:
        N (int): Number of bits to generate.
        
    Returns:
        np.ndarray: Array of N bits (0s and 1s).
    """
    bit_sequence = np.zeros(N, dtype=int)
    # LFSR Seed (must be non-zero)
    reg = 0x01f1 

    for i in range(N):
        bit_sequence[i] = reg & 1
        # PRBS15 feedback logic
        feedback = ((reg >> 14) ^ (reg >> 13)) & 1
        reg = ((reg << 1) | feedback) & 0x7fff
        
    return bit_sequence
