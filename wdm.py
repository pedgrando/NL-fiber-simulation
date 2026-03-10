import numpy as np

# -------------------------------------------------------------
# --------- Wavelength Division Multiplexing (WDM) -----------
# -------------------------------------------------------------

def mux(t, Q, B0):
    """
    WDM Multiplexer: shifts Nu baseband user signals to their respective
    frequency slots and sums them into a single composite signal.

    Implements equation (20) from the lab notes:
        q(t, 0) = sum_{k=k1}^{k2} q_k(t, 0) * exp(j * 2*pi * k * B0 * t)

    User k is centred at frequency f = k * B0 and occupies the band
    [k*B0 - B0/2, k*B0 + B0/2].

    The user index runs symmetrically around zero:
        k1 = -floor(Nu/2),  k2 = ceil(Nu/2) - 1

    Args:
        t  (np.ndarray): Time vector of length N.
        Q  (np.ndarray): 2-D array of shape (Nu, N). Row k contains the
                         baseband signal of user k.
        B0 (float):      Per-user bandwidth (B0 = B / Nu).

    Returns:
        np.ndarray: Multiplexed composite signal, length N.
    """
    Nu = Q.shape[0]
    N  = Q.shape[1]

    k1 = -int(np.floor(Nu / 2))
    k2 =  int(np.ceil(Nu  / 2)) - 1

    q_mux = np.zeros(N, dtype=complex)

    for idx, k in enumerate(range(k1, k2 + 1)):
        # Frequency-shift user k to its carrier slot
        carrier = np.exp(1j * 2 * np.pi * k * B0 * t)
        q_mux  += Q[idx] * carrier

    return q_mux


def demux(t, qz, Nu, B0):
    """
    WDM Demultiplexer: recovers each user's baseband signal from the
    received composite signal.

    Implements equation (21) from the lab notes:
        q̃_k(t, z) = exp(-j * 2*pi * k * B0 * t) * F_{W_k}[ q(t, z) ]

    where F_{W_k} is an ideal brick-wall bandpass filter centred at k*B0
    with one-sided bandwidth B0/2. The filter is applied in the frequency
    domain; the signal is then frequency-shifted back to baseband.

    Args:
        t  (np.ndarray): Time vector of length N.
        qz (np.ndarray): Received composite signal, length N.
        Nu (int):        Number of WDM users.
        B0 (float):      Per-user bandwidth.

    Returns:
        np.ndarray: 2-D array of shape (Nu, N).  Row k contains the
                    baseband signal of user k.
    """
    N  = len(t)
    dt = t[1] - t[0]
    f  = np.fft.fftfreq(N, d=dt)   # natural-order frequency axis

    k1 = -int(np.floor(Nu / 2))
    k2 =  int(np.ceil(Nu  / 2)) - 1

    Qz  = np.fft.fft(qz)            # spectrum of received signal
    Q_demux = np.zeros((Nu, N), dtype=complex)

    for idx, k in enumerate(range(k1, k2 + 1)):
        # --- Step 1: bandpass filter centred at k * B0 ----------
        # Keep only the spectral slice W_k = [k*B0 - B0/2, k*B0 + B0/2]
        band_mask = np.abs(f - k * B0) <= B0 / 2
        Qz_filtered = Qz * band_mask          # frequency-domain filtering

        # Back to time domain (still passband-centred at k*B0)
        qz_filtered = np.fft.ifft(Qz_filtered)

        # --- Step 2: frequency-shift down to baseband -----------
        q_bb = qz_filtered * np.exp(-1j * 2 * np.pi * k * B0 * t)

        Q_demux[idx] = q_bb

    return Q_demux
