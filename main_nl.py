import numpy as np
import matplotlib.pyplot as plt

from source import source_bernoulli
from mapper import mapper, demap_decode
from modulator import modul, demod
from channel import nl_channel, dbp_equalizer
from miscellaneous import sig_spectrum, symbol_error
from parameters import (
    beta2_norm, gamma_norm, Bn, P_norm, power_scale_norm, M
)

def main_nl():
    """
    Simulates a Nonlinear Fiber Channel using the Split-Step Fourier Method (SSFM).
    This script demonstrates Self-Phase Modulation (SPM) and spectral broadening.
    """
    print("=== Nonlinear Channel Simulation ===")

    # --------------- System parameters ------------------------
    B = Bn                          # normalized bandwidth
    Ns = 1000                       # number of symbols
    nb = Ns * int(np.log2(M))       # number of bits
    
    # Simulation Window
    sps = 8
    T = (Ns + 20) / B               # Larger margin for NL broadening
    dt = 1 / (B * sps)
    N = int(2**np.ceil(np.log2(T / dt)))
    T = N * dt
    t = np.linspace(-T/2, T/2, N, endpoint=False)
    
    # Frequency mesh
    f = np.fft.fftfreq(N, d=dt)

    # --------------- Nonlinear Parameters ---------------------
    L = 1.0                         # normalized distance
    Nsteps = 5000                    # SSFM steps

    # Normalized coefficients for the NLSE scales used in parameters.py
    betav = [0, 0, beta2_norm]
    
    # We will test two power levels: Normal and High
    power_factors = [0.125, 0.25, 0.5, 1, 2]      # Multiplier for P_norm

    for factor in power_factors:
        curr_P = P_norm * factor
        curr_scale = np.sqrt(curr_P)
        print(f"\nSimulating Power Factor: {factor} (P_norm = {curr_P:.4e})")

        # 1. Transmitter
        b = source_bernoulli(nb, 0.5)
        s = mapper(b, const=M, normalize=True) * curr_scale
        q0t = modul(t, s, B)

        # 2. Nonlinear Channel (SSFM)
        # Note: nl_channel returns (qt, qf)
        qzt, qzf = nl_channel(t, q0t, z = L, Nsteps = Nsteps, gamma = gamma_norm, sigma2 = 0, betav = betav, f = f, B = B)

        # 3. DBP Equalizer (Dispersion Compensation & Nonlinearity)
        qzt_eq, qzf_eq = dbp_equalizer(t, qzt, z = L, Nsteps = Nsteps, gamma = gamma_norm, betav = betav, f = f, B = B)

        # 4.a) Demodulation & Detection
        s_hat_eq = np.array(demod(qzt_eq, dt, B, Ns, t))
        s_hat_unit_eq = s_hat_eq / curr_scale
        s_detected_eq, b_hat_eq = demap_decode(s_hat_unit_eq, const=M, normalize=True)

        # 4.b) Demodulation & Detection (before equalization)
        s_hat = np.array(demod(qzt, dt, B, Ns, t))
        s_hat_unit = s_hat / curr_scale
        s_detected, b_hat = demap_decode(s_hat_unit, const=M, normalize=True)

        # 5.a) BER/SER (after equalization)
        ser_eq = symbol_error(s_detected_eq, s_hat_unit_eq)
        ber_eq = np.sum(b != b_hat_eq) / len(b)
        print(f"  SER Accuracy (after equalization): {ser_eq:.4f}")
        print(f"  BER (after equalization): {ber_eq:.6e}")

        # 5.b) BER/SER (before equalization)
        ser = symbol_error(s_detected, s_hat_unit)
        ber = np.sum(b != b_hat) / len(b)
        print(f"  SER Accuracy (before equalization): {ser:.4f}")
        print(f"  BER (before equalization): {ber:.6e}")
        

        # 6.a) Plotting (after equalization)
        fig, axes_eq = plt.subplots(1, 2, figsize=(12, 5))
        
        # Spectrum
        Q0f_eq = np.fft.fft(q0t)
        f_plot_eq = np.fft.fftshift(f)
        axes_eq[0].semilogy(f_plot_eq, np.abs(np.fft.fftshift(Q0f_eq))**2, label='TX Spectrum')
        axes_eq[0].semilogy(f_plot_eq, np.abs(np.fft.fftshift(qzf_eq))**2, label='RX Spectrum', alpha=0.7)
        axes_eq[0].set_title(f'Spectral Broadening (Power={factor}x)')
        axes_eq[0].set_xlim([-4*B, 4*B])
        axes_eq[0].set_ylim([1e-6, None])
        axes_eq[0].legend()
        axes_eq[0].grid(True)

        # Constellation
        s_unit_eq = mapper(b, const=M, normalize=True)
        axes_eq[1].scatter(np.real(s_hat_unit_eq), np.imag(s_hat_unit_eq), s=10, alpha=0.5, label='Received')
        axes_eq[1].scatter(np.real(s_unit_eq), np.imag(s_unit_eq), marker='x', color='red', label='TX Ref')
        axes_eq[1].set_title(f'Constellation Distortion (Power={factor}x)')
        axes_eq[1].legend()
        axes_eq[1].grid(True)
        axes_eq[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'nl_results_power_eq_{factor}.png')
        print(f"  Saved plot: nl_results_power_eq_{factor}.png")
        plt.close()


        # 6.b) Plotting (before equalization)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Spectrum
        Q0f = np.fft.fft(q0t)
        f_plot = np.fft.fftshift(f)
        axes[0].semilogy(f_plot, np.abs(np.fft.fftshift(Q0f))**2, label='TX Spectrum')
        axes[0].semilogy(f_plot, np.abs(np.fft.fftshift(qzf))**2, label='RX Spectrum', alpha=0.7)
        axes[0].set_title(f'Spectral Broadening (Power={factor}x)')
        axes[0].set_xlim([-4*B, 4*B])
        axes[0].set_ylim([1e-6, None])
        axes[0].legend()
        axes[0].grid(True)

        # Constellation
        s_unit = mapper(b, const=M, normalize=True)
        axes[1].scatter(np.real(s_hat_unit), np.imag(s_hat_unit), s=10, alpha=0.5, label='Received')
        axes[1].scatter(np.real(s_unit), np.imag(s_unit), marker='x', color='red', label='TX Ref')
        axes[1].set_title(f'Constellation Distortion (Power={factor}x)')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'nl_results_power_{factor}.png')
        print(f"  Saved plot: nl_results_power_{factor}.png")
        plt.close()

if __name__ == "__main__":
    main_nl()
