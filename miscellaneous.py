import numpy as np

from source import source_bernoulli, source_prbs
from mapper import mapper, demap_decode
from modulator import rrc, modul, demod
from channel import lin_channel, lin_equalizer
from parameters import (
    beta2, sigma_sq, Bn, P_norm, power_scale_norm, M, P
)

# -------------------------------------------------------------
# --------------- Miscellaneous Functions ---------------------
# -------------------------------------------------------------

def sig_spectrum(sig):
    """
    Computes the frequency-centered spectrum of a time-domain signal.
    Equivalent to fftshift(fft(sig)).
    """
    return np.fft.fftshift(np.fft.fft(sig))

# create function to compare whether symbols are correctly demod (sum the absolute squared error + remember to normalize by vector s norm)
def symbol_error(s, s_hat):
    return np.sum(np.isclose(s, s_hat))/len(s)


def run_snr_point(snr_dB, Ns, sps, B, p, beta2, z, power_scale_norm, M, n_trials=100, verbose=False):
    """
    Run Monte Carlo for a single SNR value.

    Parameters
    ----------
    snr_dB : float
        Signal‑to‑noise ratio in dB (SNR = P_signal / (sigma2 * B)).
    Ns : int
        Number of symbols per trial.
    sps : int
        Samples per symbol.
    B : float
        Normalized bandwidth.
    p : float
        Bernoulli parameter for source.
    beta2 : float
        Dispersion coefficient.
    z : float
        Normalized distance (usually 1).
    power_scale_norm : float
        Scaling factor to achieve desired signal power.
    M : int
        Constellation size.
    n_trials : int
        Number of independent trials.
    verbose : bool
        If True, print intermediate info.

    Returns
    -------
    ber : float
        Average bit error rate over all trials.
    ser : float
        Average symbol error rate.
    """
    # Compute noise variance from SNR
    snr_lin = 10 ** (snr_dB / 10)
    sigma2 = P_norm / (snr_lin * B)   # because total noise power = sigma2 * B

    # Time parameters (same as in main)
    T = (Ns + 10) / B
    dt = 1 / (B * sps)
    N = int(np.ceil(T / dt))
    N = int(2 ** np.ceil(np.log2(N)))   # power of two for FFT
    T = N * dt
    t = np.linspace(-T / 2, T / 2, N, endpoint=False)
    f = np.fft.fftfreq(N, d=dt)

    total_bit_errors = 0
    total_symbol_errors = 0
    total_bits = 0
    total_symbols = 0

    for trial in range(n_trials):
        # --- Transmitter ---
        nb = Ns * int(np.log2(M))
        b = source_bernoulli(nb, p)
        s = mapper(b, const=M, normalize=True) * power_scale_norm
        q0t = modul(t, s, B)

        # --- Channel ---
        qzt, _ = lin_channel(q0t, f, beta2=beta2, z=z, sigma2=sigma2, B=B)

        # --- Equalizer ---
        qzt_eq, _ = lin_equalizer(qzt, t, f, beta2=beta2, L=z)

        # --- Demodulator ---
        s_hat = demod(qzt_eq, dt, B, Ns, t)
        s_hat = np.array(s_hat)

        # --- Detector ---
        s_hat_unit = s_hat / power_scale_norm
        s_detected, b_hat = demap_decode(s_hat_unit, const=M, normalize=True)

        # --- Count errors ---
        bit_errors = np.sum(b != b_hat)
        symbol_errors = np.sum(s_detected != (s / power_scale_norm))
        total_bit_errors += bit_errors
        total_symbol_errors += symbol_errors
        total_bits += len(b)
        total_symbols += len(s_detected)

        if verbose and trial % 10 == 0:
            print(f"Trial {trial}: bit errors = {bit_errors}")

    ber = total_bit_errors / total_bits
    ser = total_symbol_errors / total_symbols
    return ber, ser
