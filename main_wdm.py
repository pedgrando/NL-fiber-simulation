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
    Ns  = 500                       # symbols per user (per trial)
    B0  = Bn / Nu                   # per-user normalized bandwidth
    B   = Nu * B0                   # total WDM bandwidth  (= Bn)

    nb  = Ns * int(np.log2(M))      # bits per user

    # --------------- Simulation parameters --------------------
    # Same derivation as main.py, but T is set by per-user symbol rate 1/B0.

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
    Nsteps = 1000                    # SSFM sub-steps
    sigma2 = sigma_sq               # noise at the physical operating point
    betav_norm = [0, 0, beta2_norm]

    # Constellation amplitude a (= symbol power scale).
    # power_scale_norm ≈ 1.96 enforces the P=1 mW physical power constraint,
    # but over L=10 000 km the NL impairment is catastrophic at that power.
    # Reduce a to operate in a mixed linear/NL regime  (see lab Remark 2).
    # For BER vs SNR plots, a also sets the x-axis intercept of the NL floor.
    a = 0.25                        # tune this; a < power_scale_norm means P < P_target

    # User-of-interest indices  (Q30: "middle symbol of the middle user")
    # Users are indexed k = k1..k2 (k1 = -floor(Nu/2), k2 = ceil(Nu/2)-1).
    # k=0 is the center user, stored at row k0 = floor(Nu/2) in the array.
    # l=0 is the center symbol, at index l0 = floor(Ns/2) in the symbol vector.
    k0 = Nu // 2
    l0 = Ns // 2

    # ---------------------- Transmitter -----------------------
    # Each user is modulated independently in the baseband interval W_0.

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
    plt.savefig('wdm_user_baseband.png', dpi=150); plt.show()

    # --------------- Multiplexing (eq. 20) --------------------
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
    plt.savefig('wdm_tx_signal.png', dpi=150); plt.show()

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
    plt.savefig('wdm_rx_signal.png', dpi=150); plt.show()

    # --------------- Demultiplexing (eq. 21) ------------------
    # Q̃_k(t,z) = exp(-j·2π·k·B0·t) · F_{W_k}[ q(t,z) ]

    Q_rx = demux(t, qzt, Nu, B0)

    # --------------- Equalization (per user) ------------------

    Q_eq = np.zeros((Nu, N), dtype=complex)
    for k in range(Nu):
        _, Q_eq[k]= dbp_equalizer(t, Q_rx[k], z = z, Nsteps = Nsteps, 
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
    plt.savefig('wdm_equalized_signal.png', dpi=150); plt.show()

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
    plt.savefig('wdm_constellation.png', dpi=150); plt.show()

    # --------------------- BER vs SNR -------------------------
    # Run Nr independent trials per SNR point.
    # Track the center user k=0 across all its Ns_ber symbols.
    #
    # SNR = a² / sigma2  (= E[|s|²] / noise PSD), consistent with run_snr_point.
    # => sigma2 = a² / snr_lin
    #
    # Final results: BER vs SNR for M = 4 and M = 16  (Section 3.3).

    snr_range_dB = np.arange(0, 33, 3)
    Nr    = 200                     # Monte-Carlo trials per SNR point
    Ns_ber = 100                     # symbols per user per trial (BER sweep)

    # Rebuild mesh for Ns_ber (same dt, shorter T)
    T_ber = (Ns_ber + 10) / B0
    N_ber = int(2**np.ceil(np.log2(T_ber / dt)))
    T_ber = N_ber * dt
    t_ber = np.linspace(-T_ber / 2, T_ber / 2, N_ber, endpoint=False)
    f_ber = np.fft.fftfreq(N_ber, d=dt)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = {4: 'tab:blue', 16: 'tab:orange'}
    markers = {4: 'o',        16: 's'}

    for M_mod in [4, 16]:

        nb_ber = Ns_ber * int(np.log2(M_mod))
        print(f"\n--- BER vs SNR  ({M_mod}-QAM,  Nu={Nu} users,  a={a}) ---")

        bers = []

        for snr_dB in snr_range_dB:

            snr_lin   = 10 ** (snr_dB / 10)
            sigma2_it = a**2 / snr_lin      # SNR = a² / sigma2

            total_bit_errors = 0
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
                qzt_it, _ = nl_channel(t_ber, q0t_it, z=z, Nsteps=Nsteps,
                                       gamma=gamma_norm, sigma2=sigma2_it,
                                       betav=betav_norm, f=f_ber, B=B)

                # --- Demultiplexing ---
                Q_rx_it = demux(t_ber, qzt_it, Nu, B0)

                # --- Equalize center user ---
                _, q_eq_it = dbp_equalizer(t_ber, Q_rx_it[k0], z = z, Nsteps = Nsteps, 
                                  gamma = gamma_norm, betav=betav_norm, 
                                  f = f_ber, B = B)

                # --- Demodulate center user ---
                s_hat_it       = np.array(demod(q_eq_it, dt, B0, Ns_ber, t_ber))
                s_hat_unit_it  = s_hat_it / a
                _, b_hat_it    = demap_decode(s_hat_unit_it, const=M_mod,
                                              normalize=True)

                total_bit_errors += np.sum(b_all[k0] != b_hat_it)
                total_bits_count += nb_ber

            ber_pt = total_bit_errors / total_bits_count
            bers.append(ber_pt)
            print(f"  SNR = {snr_dB:5.1f} dB  BER = {ber_pt:.4e}"
                  f"  ({total_bit_errors}/{total_bits_count})")

        bers  = np.array(bers)
        valid = bers > 0
        ax.semilogy(snr_range_dB[valid], bers[valid],
                    marker=markers[M_mod], color=colors[M_mod],
                    label=f'{M_mod}-QAM  (Nu={Nu})')

    ax.set_xlabel('SNR = a² / σ²  [dB]')
    ax.set_ylabel('BER')
    ax.set_title(f'WDM BER vs SNR  –  {Nu} users, nonlinear channel  (a={a})')
    ax.legend(); ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_ylim([1e-4, 1])
    plt.tight_layout()
    plt.savefig('wdm_ber_vs_snr.png', dpi=150); plt.show()


if __name__ == "__main__":
    main_wdm()
