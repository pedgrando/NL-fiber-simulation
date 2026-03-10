import numpy as np
import matplotlib.pyplot as plt
import math

from source    import source_bernoulli
from mapper    import mapper, demap_decode
from modulator import modul, demod
from channel   import nl_channel, dbp_equalizer
from wdm       import mux, demux
from miscellaneous import symbol_error
from parameters import (
    beta2_norm, gamma_norm, sigma_sq, Bn, P_norm, power_scale_norm, M
)

# -------------------------------------------------------------
# --------- Main: WDM Nonlinear Channel Pipeline --------------
# -------------------------------------------------------------

def main_wdm():

    # --------------- System parameters ------------------------
    Nu  = 5                         # number of WDM users
    Ns  = 200                        # symbols per user (per trial)
    B0  = Bn / Nu                   # per-user normalized bandwidth
    B   = Nu * B0                   # total WDM bandwidth  (= Bn)

    nb  = Ns * int(np.log2(M))      # bits per user

    # --------------- Simulation parameters --------------------
    sps    = 8                                      # oversampling factor
    T      = (Ns + 10) / B0                         # window fits all Ns symbols
    dt     = 1 / (B * sps)                          # sample step (total bandwidth)
    N      = int(2**np.ceil(np.log2(T / dt)))       # round up to power of 2 (FFT)
    T      = N * dt
    t      = np.linspace(-T / 2, T / 2, N, endpoint=False)

    # Frequency mesh (natural order)
    f = np.fft.fftfreq(N, d=dt)

    p      = 0.5                    # Bernoulli parameter
    z      = 1                      # normalized propagation distance
    Nsteps = 50                    # SSFM sub-steps
    sigma2 = sigma_sq               # noise at the physical operating point
    betav_norm = [0, 0, beta2_norm]

    # Constellation amplitude a (= symbol power scale).
    a = 0.25                        # tune this; a < power_scale_norm means P < P_target

    # User-of-interest indices
    # k=0 is the center user, stored at row k0 = floor(Nu/2) in the array
    # l=0 is the center symbol, at index l0 = floor(Ns/2) in the symbol vector
    k0 = Nu // 2
    l0 = Ns // 2

    # ---------------------- Transmitter -----------------------
    # Each user is modulated independently in the baseband interval W_0

    b = np.zeros((Nu, nb), dtype=int)
    s = np.zeros((Nu, Ns), dtype=complex)
    Q = np.zeros((Nu, N),  dtype=complex)

    for k in range(Nu):
        b[k] = source_bernoulli(nb, p)
        s[k] = mapper(b[k], const=M, normalize=True) * a
        Q[k] = modul(t, s[k], B0)

    # --------------- Plot user k=0 baseband signal -------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(Q[k0]), label='Re')
    axes[0].plot(t, np.imag(Q[k0]), label='Im', alpha=0.7)
    axes[0].set_title('Baseband signal  q₀(t, 0)  (user k=0)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].set_xlim([-Ns / (2 * B0) * 1.1, Ns / (2 * B0) * 1.1])
    axes[0].legend(); axes[0].grid(True)

    Qk0f = np.fft.fft(Q[k0])
    axes[1].plot(np.fft.fftshift(f), np.abs(np.fft.fftshift(Qk0f))**2)
    axes[1].set_title('|q̂₀(f, 0)|²  (user k=0, baseband)')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-3 * B0, 3 * B0]); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('wdm_user_baseband.png', dpi=150); # plt.show()

    # -------------------- Multiplexing ------------------------
    # q(t,0) = Σ_k  Q_k(t,0) · exp(j·2π·k·B0·t)

    q0t = mux(t, Q, B0)

    # --------------- Plot WDM TX signal -----------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(q0t), linewidth=0.6, label='Re')
    axes[0].plot(t, np.imag(q0t), linewidth=0.6, label='Im', alpha=0.7)
    axes[0].set_title('WDM composite signal  q(t, 0)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend(); axes[0].grid(True)

    Q0f = np.fft.fft(q0t)
    axes[1].plot(np.fft.fftshift(f), np.abs(np.fft.fftshift(Q0f))**2)
    axes[1].set_title('|q̂(f, 0)|²  (WDM TX spectrum, all users)')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-1.5 * B, 1.5 * B]); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('wdm_tx_signal.png', dpi=150); # plt.show()

    # --------------- Channel: nonlinear SSFM ------------------

    qzt, _ = nl_channel(t, q0t, z=z, Nsteps=Nsteps,
                        gamma=gamma_norm, sigma2=sigma2,
                        betav=betav_norm, f=f, B=B)

    # --------------- Plot WDM RX signal -----------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(qzt), linewidth=0.6, label='Re')
    axes[0].plot(t, np.imag(qzt), linewidth=0.6, label='Im', alpha=0.7)
    axes[0].set_title('WDM received signal  q(t, z)  (before demux/eq.)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend(); axes[0].grid(True)

    Qzf = np.fft.fft(qzt)
    axes[1].plot(np.fft.fftshift(f), np.abs(np.fft.fftshift(Qzf))**2)
    axes[1].set_title('|q̂(f, z)|²  (WDM RX spectrum, after NL channel)')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-1.5 * B, 1.5 * B]); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('wdm_rx_signal.png', dpi=150); # plt.show()

    # --------------- Demultiplexing ---------------------------
    # Q̃_k(t,z) = exp(-j·2π·k·B0·t) · F_{W_k}[ q(t,z) ]

    Q_rx = demux(t, qzt, Nu, B0)

    # --------------- Equalization (per user) ------------------

    Q_eq = np.zeros((Nu, N), dtype=complex)
    for k in range(Nu):
        Q_eq[k], _ = dbp_equalizer(t, Q_rx[k], z = z, Nsteps = Nsteps, 
                                  gamma = gamma_norm, betav=betav_norm, 
                                  f = f, B = B)

    # --------------- Plot equalized signal for user k=0 -------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.real(Q[k0]),    label='TX (Re)', alpha=0.5)
    axes[0].plot(t, np.real(Q_eq[k0]), '--', label='EQ (Re)')
    axes[0].set_title('User k=0: equalized vs transmitted (Re)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].set_xlim([-Ns / (2 * B0) * 1.1, Ns / (2 * B0) * 1.1])
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t, np.imag(Q[k0]),    label='TX (Im)', alpha=0.5)
    axes[1].plot(t, np.imag(Q_eq[k0]), '--', label='EQ (Im)')
    axes[1].set_title('User k=0: equalized vs transmitted (Im)')
    axes[1].set_xlabel('t (normalized)')
    axes[1].set_xlim([-Ns / (2 * B0) * 1.1, Ns / (2 * B0) * 1.1])
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('wdm_equalized_signal.png', dpi=150); # plt.show()

    # --------------------- Demodulator ------------------------

    s_hat = np.zeros((Nu, Ns), dtype=complex)
    for k in range(Nu):
        s_hat[k] = np.array(demod(Q_eq[k], dt, B0, Ns, t))

    # --------------------- Detector ---------------------------

    s_hat_unit = s_hat[k0] / a
    s_detected, b_hat = demap_decode(s_hat_unit, const=M, normalize=True)

    # --------------- Symbol of interest  (Q30) ----------------
    # Extract center user k=0, center symbol l=0

    s00    = s[k0, l0]
    shat00 = s_hat[k0, l0]
    print(f"\n=== Symbol of interest  (k=0, l=0) ===")
    print(f"  Transmitted  s00    = {s00 / a:.4f}")
    print(f"  Received     shat00 = {shat00 / a:.4f}")

    # --------------- BER / SER for single run -----------------

    ser = symbol_error(s_detected, s[k0] / a)
    ber = np.sum(b[k0] != b_hat) / nb
    print(f"\n=== Results (center user k=0, single run) ===")
    print(f"  a   = {a}")
    print(f"  σ²  = {sigma2:.4e}")
    print(f"  SER = {ser:.4f}")
    print(f"  BER = {ber:.6f}")
    print(f"  Bit errors: {np.sum(b[k0] != b_hat)} / {nb}")

    # --------------- Constellation plot -----------------------

    s_tx_unit = s[k0] / a
    fig, axes  = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(np.real(s_tx_unit), np.imag(s_tx_unit), marker='x', color='blue')
    axes[0].set_title(f'{M}-QAM TX constellation (user k=0)')
    axes[0].set_xlabel('In-phase'); axes[0].set_ylabel('Quadrature')
    axes[0].grid(True); axes[0].set_aspect('equal')

    axes[1].scatter(np.real(s_hat_unit), np.imag(s_hat_unit),
                    marker='o', color='red', alpha=0.6, label='RX')
    axes[1].scatter(np.real(s_tx_unit), np.imag(s_tx_unit),
                    marker='x', color='blue', label='TX')
    axes[1].set_title(f'Received constellation (user k=0, NL channel)')
    axes[1].set_xlabel('In-phase'); axes[1].set_ylabel('Quadrature')
    axes[1].legend(); axes[1].grid(True); axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig('wdm_constellation.png', dpi=150); # plt.show()

    # --------------------- BER vs SNR -------------------------
    # Run Nr independent trials per SNR point.
    # Track the center user k=0 across all its Ns_ber symbols.
    #
    # SNR = a² / sigma2  (= E[|s|²] / noise PSD) => sigma2 = a² / snr_lin
    #

    snr_range_dB = np.concatenate([np.arange(0, 25, 3), np.arange(35, 61, 3)])
    Nr    = 20                      # Trials per SNR point
    Ns_ber = 200                     # More symbols for smoother curves
    # Adjust SSFM steps for sweep
    Nsteps_sweep = 100

    # Rebuild mesh for Ns_ber (same dt, shorter T)
    T_ber = (Ns_ber + 20) / B0
    N_ber = int(2**np.ceil(np.log2(T_ber / dt)))
    T_ber = N_ber * dt
    t_ber = np.linspace(-T_ber / 2, T_ber / 2, N_ber, endpoint=False)
    f_ber = np.fft.fftfreq(N_ber, d=dt)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = {4: 'tab:blue', 16: 'tab:orange'}
    markers = {4: 'o',        16: 's'}

    for M_mod in [4, 16]:

        nb_ber = Ns_ber * int(np.log2(M_mod))
        print(f"\n--- BER vs SNR  ({M_mod}-QAM,  Nu={Nu} users,  sigma2={sigma_sq}) ---")

        bers_eq = []
        bers_no_eq = []

        for snr_dB in snr_range_dB:

            snr_lin   = 10 ** (snr_dB / 10)
            a = np.sqrt(sigma_sq * snr_lin * B)      # SNR = a² / (sigma2 * B) => a² = SNR * sigma2 * B

            total_bit_errors_eq = 0
            total_bit_errors_no_eq = 0
            total_bits_count = 0

            for trial in range(Nr):

                # --- Transmitter: all users ---
                b_all = np.zeros((Nu, nb_ber), dtype=int)
                s_all = np.zeros((Nu, Ns_ber), dtype=complex)
                Q_all = np.zeros((Nu, N_ber),  dtype=complex)

                for k in range(Nu):
                    b_all[k] = source_bernoulli(nb_ber, p)
                    s_all[k] = mapper(b_all[k], const=M_mod, normalize=True) * a
                    Q_all[k] = modul(t_ber, s_all[k], B0)

                # --- Multiplexing ---
                q0t_it = mux(t_ber, Q_all, B0)

                # --- Nonlinear channel ---
                qzt_it, _ = nl_channel(t_ber, q0t_it, z=z, Nsteps=Nsteps_sweep,
                                       gamma=gamma_norm, sigma2=sigma_sq,
                                       betav=betav_norm, f=f_ber, B=B)

                # --- Demultiplexing ---
                Q_rx_it = demux(t_ber, qzt_it, Nu, B0)

                # --- Case 1: DBP Equalizer ---
                q_eq_it, _ = dbp_equalizer(t_ber, Q_rx_it[k0], z = z, Nsteps = Nsteps_sweep, 
                                  gamma = gamma_norm, betav=betav_norm, 
                                  f = f_ber, B = B)
                s_hat_it       = np.array(demod(q_eq_it, dt, B0, Ns_ber, t_ber))
                s_hat_unit_it  = s_hat_it / a
                _, b_hat_it    = demap_decode(s_hat_unit_it, const=M_mod, normalize=True)
                total_bit_errors_eq += np.sum(b_all[k0] != b_hat_it)

                # --- Case 2: No Equalization ---
                s_hat_no_eq = np.array(demod(Q_rx_it[k0], dt, B0, Ns_ber, t_ber))
                s_hat_unit_no_eq = s_hat_no_eq / a
                _, b_hat_no_eq = demap_decode(s_hat_unit_no_eq, const=M_mod, normalize=True)
                total_bit_errors_no_eq += np.sum(b_all[k0] != b_hat_no_eq)

                total_bits_count += nb_ber

            ber_eq = total_bit_errors_eq / total_bits_count
            ber_no_eq = total_bit_errors_no_eq / total_bits_count
            bers_eq.append(ber_eq)
            bers_no_eq.append(ber_no_eq)
            print(f"  SNR = {snr_dB:5.1f} dB  BER_EQ = {ber_eq:.4e}  BER_NO_EQ = {ber_no_eq:.4e}")

        bers_eq  = np.array(bers_eq)
        bers_no_eq = np.array(bers_no_eq)
        
        valid_eq = bers_eq > 0
        valid_no_eq = bers_no_eq > 0
        
        ax.semilogy(snr_range_dB[valid_eq], bers_eq[valid_eq],
                    marker=markers[M_mod], color=colors[M_mod],
                    label=f'DBP {M_mod}-QAM')

    ax.set_xlabel('SNR = a² / σ²  [dB]')
    ax.set_ylabel('BER')
    ax.set_title(f'WDM BER vs SNR  –  {Nu} users, nonlinear channel  (a={a})')
    ax.legend(); ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_ylim([1e-4, 1])
    plt.tight_layout()
    plt.savefig('wdm_ber_vs_snr.png', dpi=150) # plt.show()


if __name__ == "__main__":
    main_wdm()
