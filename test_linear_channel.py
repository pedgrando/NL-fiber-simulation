import numpy as np
import matplotlib.pyplot as plt

from source import source_bernoulli, source_prbs
from mapper import mapper, demap_decode
from modulator import rrc, modul, demod
from channel import lin_channel, lin_equalizer
from miscellaneous import sig_spectrum, symbol_error
from parameters import (
    beta2, sigma_sq, Bn, P_norm, power_scale_norm, M, P
)

# -------------------------------------------------------------
# --------- Main: Dispersive Channel Pipeline -----------------
# -------------------------------------------------------------

def main():

    # --------------- System parameters ------------------------

    B = Bn                          # normalized bandwidth
    Ns = 1000                        # number of symbols
    nb = Ns * int(np.log2(M))       # number of bits

    # --------------- Simulation parameters --------------------
    # These are DERIVED from Ns and B — not independent!
    #   sps = samples per symbol (≥2 for Nyquist, use 4-8 in practice)
    #   dt  = 1/(B·sps)            — sampling interval
    #   T   = Ns/B + margin        — time window must fit all symbols
    #   N   = round(T/dt)          — total number of samples

    sps = 8                                     # oversampling factor
    T = (Ns + 10) / B                           # margin of ~10 symbol periods
    dt = 1 / (B * sps)
    N = int(np.ceil(T / dt))
    N = int(2**np.ceil(np.log2(N)))             # round up to power of 2 (FFT)
    T = N * dt                                  # adjust T to match exact N
    t = np.linspace(-T / 2, T / 2, N, endpoint=False)

    # frequency mesh
    F = 1 / dt
    df = 1 / T
    f = np.arange(-F / 2, F / 2, df)


    p = 0.5                         # Bernoulli parameter

    z = 1                           # normalized distance (full span)

    sigma2 = sigma_sq               # noise variance     

    # ---------------------- Transmitter -----------------------

    # Source
    b = source_bernoulli(nb, p)

    # Mapper: bits → symbols (unit power, then scale to P_norm)
    s = mapper(b, const=M, normalize=True) * power_scale_norm

    # Modulator: symbols → signal
    q0t = modul(t, s, B)

    # Check 1: Symbol power
    symbol_power = np.mean(np.abs(s)**2)
    print(f"  E|s|² = {symbol_power:.4f}  (should be P_norm = {P_norm:.4f})")

    # Check 2: Signal power (discrete approximation of continuous integral)
    signal_power = np.mean(np.abs(q0t)**2)
    expected_signal_power = symbol_power * (Ns / B) / T
    print(f"  mean|q|² = {signal_power:.6f}  (expected ≈ {expected_signal_power:.6f})")


    # --------------- Plot transmitted signal -------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(q0t), label='Re')
    axes[0].plot(t, np.imag(q0t), label='Im', alpha=0.7)
    axes[0].set_title('Transmitted signal q(t, 0)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend()
    axes[0].grid(True)

    Q0f = sig_spectrum(q0t)
    axes[1].plot(f, np.abs(Q0f)**2)
    axes[1].set_title('|q̂(ω, 0)|²')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-3*B, 3*B])
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('tx_signal.png', dpi=150)
    plt.show()

    # --------------- Channel: dispersive only -----------------

    qzt, qzft = lin_channel(q0t, f, beta2=-2, z=z, sigma2=sigma2, B=B)

    # --------------- Plot received signal ---------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(qzt), label='Re')
    axes[0].plot(t, np.imag(qzt), label='Im', alpha=0.7)
    axes[0].set_title('Received signal q(t, z) (before eq.)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(f, np.abs(np.fft.fftshift(qzft))**2)
    axes[1].set_title('|q̂(ω, z)|² (before eq.)')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-3*B, 3*B])
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('rx_signal.png', dpi=150)
    plt.show()

    # --------------- Equalizer: invert channel ----------------

    # For dispersive channel, equalization = inverse transfer function
    qzt_eq, qzfteq = lin_equalizer(qzt, t, f, beta2=-2, L=z)

    # --------------- Plot equalized signal --------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(q0t), label='TX (Re)', alpha=0.5)
    axes[0].plot(t, np.real(qzt_eq), '--', label='EQ (Re)')
    axes[0].set_title('Equalized vs transmitted')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, np.imag(q0t), label='TX (Im)', alpha=0.5)
    axes[1].plot(t, np.imag(qzt_eq), '--', label='EQ (Im)')
    axes[1].set_title('Equalized vs transmitted (Imag)')
    axes[1].set_xlabel('t (normalized)')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('equalized_signal.png', dpi=150)
    plt.show()

    # --------------------- Demodulator ------------------------

    s_hat = demod(qzt_eq, dt, B, Ns, t)
    s_hat = np.array(s_hat)

    # --------------------- Detector ---------------------------

    # Scale back to unit power for demapper, then detect
    s_hat_unit = s_hat / power_scale_norm
    s_detected, b_hat = demap_decode(s_hat_unit, const=M, normalize=True)

    # --------------- BER computation --------------------------

    ser = symbol_error(s_detected, s_hat_unit)
    ber = np.sum(b != b_hat) / len(b)
    print(f"\n=== Results ===")
    print(f"  σ² = {sigma2}")
    print(f"  SER = {ser}")
    print(f"  BER = {ber:.6f}")
    print(f"  Bit errors: {np.sum(b != b_hat)} / {len(b)}")

    # --------------- Constellation plot -----------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    s_unit = mapper(b, const=M, normalize=True)
    axes[0].scatter(np.real(s_unit), np.imag(s_unit), marker='x', color='blue')
    axes[0].set_title(f'{M}-QAM Constellation (unit power)')
    axes[0].set_xlabel('In-phase')
    axes[0].set_ylabel('Quadrature')
    axes[0].grid(True)
    axes[0].set_aspect('equal')

    axes[1].scatter(np.real(s_hat_unit), np.imag(s_hat_unit), marker='o',
                    color='red', alpha=0.6, label='Received')
    axes[1].scatter(np.real(s_unit), np.imag(s_unit), marker='x',
                    color='blue', label='TX')
    axes[1].set_title('Received constellation')
    axes[1].set_xlabel('In-phase')
    axes[1].set_ylabel('Quadrature')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig('constellation.png', dpi=150)
    plt.show()

    # --------------------- BER vs SNR -------------------------

    

if __name__ == "__main__":
    main()
