import numpy as np
from source import source_bernoulli
from mapper import mapper, demap_decode
from modulator import modul, demod
from channel import nl_channel, dbp_equalizer, lin_channel, lin_equalizer

# -------------------------------------------------------------
# --------------- Monte Carlo Core Functions ------------------
# -------------------------------------------------------------

def run_nl_snr_point(snr_dB, Ns, sps, B, p, betav, z, Nsteps, gamma, sigma2, M, n_trials=200, verbose=False):
    """
    Runs a Monte Carlo simulation for a single SNR point in a nonlinear channel.
    
    This function simulates the entire transmission chain:
    Source -> Mapping -> Modulation -> NL Canal (SSFM) -> DBP -> Demodulation -> Detection.
    It returns both the equalized (DBP) and non-equalized BER.
    
    Args:
        snr_dB (float): Target SNR in decibels.
        Ns (int): Number of symbols per trial.
        sps (int): Samples per symbol.
        B (float): Bandwidth (Baud rate).
        p (float): Bernoulli probability for source.
        betav (list): Dispersion coefficients.
        z (float): Total propagation distance.
        Nsteps (int): Number of SSFM steps.
        gamma (float): Nonlinearity coefficient.
        sigma2 (float): Noise power spectral density.
        M (int): Constellation size.
        n_trials (int): Number of independent Monte Carlo trials.
        verbose (bool): If True, prints progress info.
        
    Returns:
        tuple: (ber_eq, ber_no_eq) bit error rates.
    """
    snr_lin = 10 ** (snr_dB / 10)
    # P_sig = SNR * noise_spectral_density * Bandwidth
    a = np.sqrt(snr_lin * sigma2 * B)

    # Time/Frequency mesh setup
    T = (Ns + 20) / B
    dt = 1 / (B * sps)
    N = int(2**np.ceil(np.log2(T / dt)))
    T = N * dt
    t = np.linspace(-T/2, T/2, N, endpoint=False)
    f = np.fft.fftfreq(N, d=dt)

    total_bit_errors_eq = 0
    total_bit_errors_no_eq = 0
    total_bits = 0

    for trial in range(n_trials):
        # 1. Transmitter
        nb = Ns * int(np.log2(M))
        b = source_bernoulli(nb, p)
        s = mapper(b, const=M, normalize=True) * a
        q0t = modul(t, s, B)

        # 2. Nonlinear Channel
        qzt, _ = nl_channel(t, q0t, z=z, Nsteps=Nsteps, gamma=gamma, sigma2=sigma2, betav=betav, f=f, B=B)

        # 3. Receiver - Case 1: DBP Equalizer
        qzt_eq, _ = dbp_equalizer(t, qzt, z=z, Nsteps=Nsteps, gamma=gamma, betav=betav, f=f, B=B)
        s_hat_eq = np.array(demod(qzt_eq, dt, B, Ns, t))
        s_hat_unit_eq = s_hat_eq / a
        _, b_hat_eq = demap_decode(s_hat_unit_eq, const=M, normalize=True)
        total_bit_errors_eq += np.sum(b != b_hat_eq)

        # 4. Receiver - Case 2: No Equalization
        s_hat_no_eq = np.array(demod(qzt, dt, B, Ns, t))
        s_hat_unit_no_eq = s_hat_no_eq / a
        _, b_hat_no_eq = demap_decode(s_hat_unit_no_eq, const=M, normalize=True)
        total_bit_errors_no_eq += np.sum(b != b_hat_no_eq)

        total_bits += len(b)

    ber_eq = total_bit_errors_eq / total_bits
    ber_no_eq = total_bit_errors_no_eq / total_bits
    return ber_eq, ber_no_eq

def run_snr_point(snr_dB, Ns, sps, B, p, beta2, z, power_scale_norm, M, n_trials=100, verbose=False):
    """
    Runs a Monte Carlo simulation for a single SNR point in a linear dispersive channel.
    
    Args:
        snr_dB (float): Target SNR in dB.
        Ns (int): Symbols per trial.
        sps (int): Samples per symbol.
        B (float): Bandwidth.
        p (float): Bernoulli source parameter.
        beta2 (float): Dispersion coefficient.
        z (float): Channel length.
        power_scale_norm (float): Normalization factor for signal power.
        M (int): QAM size.
        n_trials (int): Monte Carlo iterations.
        verbose (bool): If True, prints progress.
        
    Returns:
        tuple: (ber, ser) average bit and symbol error rates.
    """
    from parameters import sigma_sq, P_norm
    snr_lin = 10 ** (snr_dB / 10)
    sigma2 = P_norm / (snr_lin * B)

    T = (Ns + 10) / B
    dt = 1 / (B * sps)
    N = int(2 ** np.ceil(np.log2(T / dt)))
    t = np.linspace(-T / 2, T / 2, N, endpoint=False)
    f = np.fft.fftfreq(N, d=dt)

    total_bit_errors = 0
    total_symbol_errors = 0
    total_bits = 0
    total_symbols = 0

    for _ in range(n_trials):
        # Transmit
        nb = Ns * int(np.log2(M))
        b = source_bernoulli(nb, p)
        s = mapper(b, const=M, normalize=True) * power_scale_norm
        q0t = modul(t, s, B)

        # Channel & Equalize
        qzt, _ = lin_channel(q0t, f, beta2=beta2, z=z, sigma2=sigma2, B=B)
        qzt_eq, _ = lin_equalizer(qzt, t, f, beta2=beta2, L=z)

        # Receive
        s_hat = np.array(demod(qzt_eq, dt, B, Ns, t))
        s_hat_unit = s_hat / power_scale_norm
        s_detected, b_hat = demap_decode(s_hat_unit, const=M, normalize=True)

        total_bit_errors += np.sum(b != b_hat)
        total_symbol_errors += np.sum(s_detected != (s / power_scale_norm))
        total_bits += len(b)
        total_symbols += len(s_detected)

    return total_bit_errors / total_bits, total_symbol_errors / total_symbols

# -------------------------------------------------------------
# --------------- Helper Utilities ----------------------------
# -------------------------------------------------------------

def sig_spectrum(sig):
    """Computes the frequency-centered spectrum of a signal."""
    return np.fft.fftshift(np.fft.fft(sig))

def symbol_error(s, s_hat):
    """Computes the Symbol Error Rate (SER) between two symbol vectors."""
    return 1.0 - np.sum(np.isclose(s, s_hat)) / len(s)
