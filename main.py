import numpy as np
import matplotlib.pyplot as plt

from source import source_bernoulli
from mapper import mapper, demap_decode
from modulator import modul, demod
from channel import lin_channel, lin_equalizer
from miscellaneous import sig_spectrum, symbol_error
from parameters import (
    beta2_norm, sigma_sq, Bn, P_norm, power_scale_norm, M
)

# -------------------------------------------------------------------------
# --------- Main: Dispersive Linear Channel Pipeline ----------------------
# -------------------------------------------------------------------------
# This script demonstrates a single-channel transmission over a purely 
# dispersive fiber channel (no nonlinearity). It uses a standard 
# Transmitter -> Channel -> Equalizer -> Receiver chain.

def main():

    # --------------- 1. System parameters ------------------------
    B = Bn                          # Normalized bandwidth
    Ns = 2000                       # Number of symbols
    M_mod = 4                       # 4-QAM (QPSK)
    nb = Ns * int(np.log2(M_mod))   # Total number of source bits

    # --------------- 2. Simulation Grid Setup --------------------
    sps = 8                         # Samples per symbol
    T = (Ns + 10) / B               # Time window with symbol margin
    dt = 1 / (B * sps)
    N = int(2**np.ceil(np.log2(T / dt))) # Next power of 2 for FFT
    T = N * dt                      
    t = np.linspace(-T / 2, T / 2, N, endpoint=False)
    f = np.fft.fftfreq(N, d=dt)      # Natural order frequencies

    # --------------- 3. Channel Settings -------------------------
    snr_dB = 12                     # Target Signal-to-Noise Ratio (dB)
    snr_lin = 10 ** (snr_dB / 10)
    # sigma2 satisfies SNR = P_norm / (sigma2 * B)
    sigma2 = P_norm / (snr_lin * B)   
    z = 1.0                         # Normalized propagation distance

    # --------------- 4. Transmitter ------------------------------
    # Generate random bits
    b = source_bernoulli(nb, p=0.5)

    # Map bits to QAM symbols and scale to desired power
    s = mapper(b, const=M_mod, normalize=True) * power_scale_norm

    # Pulse shape modulation (Sinc pulses)
    q0t = modul(t, s, B)

    # --- Plot Transmitted Signal ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(q0t), label='Re')
    axes[0].plot(t, np.imag(q0t), label='Im', alpha=0.7)
    axes[0].set_title('Transmitted Signal q(t, 0)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend(); axes[0].grid(True)

    Q0f = np.fft.fft(q0t)
    axes[1].plot(np.fft.fftshift(f), np.abs(np.fft.fftshift(Q0f))**2)
    axes[1].set_title('Transmitted Spectrum |q̂(f, 0)|²')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-3*B, 3*B]); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('tx_signal.png', dpi=150)

    # --------------- 5. Fiber Channel (Linear) -------------------
    # Apply dispersion and add AWGN
    qzt, qzft = lin_channel(q0t, f, beta2=beta2_norm, z=z, sigma2=sigma2, B=B)

    # --- Plot Received Signal ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(qzt), label='Re')
    axes[0].plot(t, np.imag(qzt), label='Im', alpha=0.7)
    axes[0].set_title('Received Signal (Before Equalization)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(np.fft.fftshift(f), np.abs(np.fft.fftshift(qzft))**2)
    axes[1].set_title('Received Spectrum')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-3*B, 3*B]); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('rx_signal.png', dpi=150)

    # --------------- 6. Equalizer (Dispersion Compensation) ------
    # Negate original dispersion effect (Linear Equalizer)
    qzt_eq, _ = lin_equalizer(qzt, t, f, beta2=beta2_norm, L=z)

    # --- Plot Equalization Quality ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(q0t), label='TX (Re)', alpha=0.5)
    axes[0].plot(t, np.real(qzt_eq), '--', label='EQ (Re)')
    axes[0].set_title('Equalized vs transmitted (Real)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t, np.imag(q0t), label='TX (Im)', alpha=0.5)
    axes[1].plot(t, np.imag(qzt_eq), '--', label='EQ (Im)')
    axes[1].set_title('Equalized vs transmitted (Imag)')
    axes[1].set_xlabel('t (normalized)')
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('equalized_signal.png', dpi=150)

    # --------------- 7. Receiver Processing ----------------------
    # Projection onto pulse basis (Matched Filter / Sampling)
    s_hat = np.array(demod(qzt_eq, dt, B, Ns, t))

    # Normalize back to constellation units and perform detection
    s_hat_unit = s_hat / power_scale_norm
    s_detected, b_hat = demap_decode(s_hat_unit, const=M_mod, normalize=True)

    # --------------- 8. BER & Metrics ----------------------------
    # Compare transmitted symbols/bits with detected counterparts
    ser = symbol_error(s_detected, s_hat_unit)
    ber = np.sum(b != b_hat) / len(b)

    print(f"\n=== Simulation Results ({M_mod}-QAM) ===")
    print(f"  SNR  = {snr_dB} dB")
    print(f"  SER  = {ser:.4f}")
    print(f"  BER  = {ber:.6e}")
    print(f"  Errors: {np.sum(b != b_hat)} / {len(b)}")

    # --- Constellation Analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    s_unit = mapper(b, const=M_mod, normalize=True)
    axes[0].scatter(np.real(s_unit), np.imag(s_unit), marker='x', color='blue')
    axes[0].set_title(f'Ideal {M_mod}-QAM Constellation')
    axes[0].grid(True); axes[0].set_aspect('equal')

    axes[1].scatter(np.real(s_hat_unit), np.imag(s_hat_unit), marker='o',
                    color='red', alpha=0.4, label='Received')
    axes[1].scatter(np.real(s_unit), np.imag(s_unit), marker='x',
                    color='blue', label='Ideal')
    axes[1].set_title('Received Constellation after EQ')
    axes[1].legend(); axes[1].grid(True); axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig('constellation.png', dpi=150)
    # plt.show()

if __name__ == "__main__":
    main()
