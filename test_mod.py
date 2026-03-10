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

     # --------------- Simulation parameters -----------------

    T = 50
    N = 2**12
    dt = T/N
    t = np.linspace(-T/2, T/2, N)

    B = 3
    
    A1 = 1
    A2 = 2

    beta2 = -2
    gamma = 2
    L = 1
    Nsteps = 5000

    # frequency mesh (natural order)
    f = np.fft.fftfreq(N, d=dt)

    # --------------- System Parameters ---------------------

    Ns = 40

    D = 1
    t0 = 7

    # --------------- Simulation ----------------------------

    # Source
    #b = source_bernoulli(Ns*4, 0.5)

    # Mapper: bits → symbols (unit power, then scale to P_norm)
    #s = mapper(b, const=16, normalize=True)

    # Modulator: symbols → signal
    #q0t = modul(t, s, B)

    q0t = A2*np.exp(-((t-t0)**2)/2*(D**2)) + A1*np.exp(-((t+t0)**2)/2*(D**2))

    #x = 1*np.exp(-(t**2)/2*0.1)

    # Since they are normalized by the mapper, this should be close to Ns
    #symbol_energy = np.sum(np.abs(s)**2)

    # We multiply by dt to account for the discretization of the integral
    #signal_energy = np.sum(np.abs(q0t)**2) * dt

    #print(f"Total Symbol Energy: {symbol_energy:.4f}")
    #print(f"Total Signal Energy: {signal_energy:.4f}")
    #print(f"Ratio (Signal/Symbol): {signal_energy/symbol_energy:.4f}")

    # breather waves - A = 2.5/3.5

    # --------------- Plot transmitted signal -------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.abs(q0t)**2, label='|q(t,0)|²')
    #axes[0].plot(t, np.real(q0t), label='Re')
    #axes[0].plot(t, np.imag(q0t), label='Im', alpha=0.7)
    axes[0].set_title('Transmitted signal q(t, 0)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend()
    axes[0].grid(True)

    Q0f = np.fft.fft(q0t)
    axes[1].plot(np.fft.fftshift(f), np.abs(np.fft.fftshift(Q0f))**2)
    axes[1].set_title('|q̂(ω, 0)|²')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-B, B])
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('tx_signal.png', dpi=150)
    plt.show()

    # --------------- Channel: dispersive only -----------------

    qzt, qzf = lin_channel(q0t, f, beta2=beta2, z=L, sigma2=0.0, B=B)

    # --------------- Plot received signal ---------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.abs(q0t)**2,'--', label='|q(t,0)|²')
    axes[0].plot(t, np.abs(qzt)**2, label='|q(t,z)|²')
    #axes[0].plot(t, np.real(qzt), label='Re')
    #axes[0].plot(t, np.imag(qzt), label='Im', alpha=0.7)
    axes[0].set_title('Received signal q(t, z) (before eq.)')
    axes[0].set_xlabel('t (normalized)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(np.fft.fftshift(f), np.abs(np.fft.fftshift(qzf))**2)
    axes[1].set_title('|q̂(ω, z)|² (before eq.)')
    axes[1].set_xlabel('f (normalized)')
    axes[1].set_xlim([-B, B])
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('rx_signal.png', dpi=150)
    plt.show()

    # --------------- Equalizer: invert channel ----------------

    # For dispersive channel, equalization = inverse transfer function
    qzt_eq, qzfteq = lin_equalizer(qzt, t, f, beta2=beta2, L=L)

    # --------------- Plot equalized signal --------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.abs(q0t)**2, label='|q(t,0)|²')
    axes[0].plot(t, np.abs(qzt_eq)**2, '--', label='|q(t,L)|² (eq.)')
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


    

if __name__ == "__main__":
    main()
