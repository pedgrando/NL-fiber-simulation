import numpy as np

# -------------------------------------------------------------
# ------------------- Source Functions ------------------------
# -------------------------------------------------------------

def source_bernoulli(N, p):
    """
    Generates a sequence of independent and identically distributed (i.i.d.) Bernoulli bits.
    
    Args:
        N (int): Number of bits to generate.
        p (float): Probability of a bit being 0. Bits are generated using np.random.rand(N) > p.
        
    Returns:
        np.ndarray: Array of N bits (0s and 1s).
    """
    return np.where(np.random.rand(N) > p, 1, 0)

def source_prbs(N):
    """
    Generates a Pseudo-Random Binary Sequence (PRBS) using a 15-bit Linear Feedback Shift Register (LFSR).
    This implements a PRBS15 sequence which is common in telecommunications testing.
    
    The generator polynomial is 1 + x^14 + x^15.
    
    Args:
        N (int): Number of bits to generate.
        
    Returns:
        np.ndarray: Array of N bits (0s and 1s).
    """
    bit_sequence = np.zeros(N, dtype=int)

    # Initial state of the 15-bit shift register (seed)
    # 0x01f1 is an arbitrary non-zero starting point.
    start = 0x01f1 
    a = start

    for i in range(0, N):
        # Extract the current output bit (least significant bit of the register)
        bit_sequence[i] = a & 1

        # Feedback logic for PRBS15: XOR the 15th bit (bit 14) and 14th bit (bit 13)
        # Note: Bits are 0-indexed, so x^15 corresponds to bit 14.
        new_bit = (((a >> 14) ^ (a >> 13)) & 1)
        
        # Shift the register to the left by 1 and insert the feedback bit at the beginning.
        # Mask with 0x7fff (15 ones) to keep the register at 15 bits.
        a = ((a << 1) | new_bit) & 0x7fff
        
    return bit_sequence
