import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm   # optional, for progress bar

from source import source_bernoulli, source_prbs
from mapper import mapper, demap_decode
from modulator import rrc, modul, demod
from channel import lin_channel, lin_equalizer
from miscellaneous import run_snr_point
from parameters import (
    beta2_norm, sigma_sq, Bn, P_norm, power_scale_norm, M, P
)


# --------------- System parameters ------------------------
B = Bn                          # normalized bandwidth
Ns = 500                        # number of symbols per trial
sps = 8                         # samples per symbol
p = 0.5
z = 1

# --------------- SNR range --------------------------------
snr_dB_list = np.arange(0, 18, 2)   # from 0 to 14 dB in steps of 2
n_trials = 250                       # number of trials per SNR
M_list = [2, 4, 8, 16] 

ber_list_mod = []
ser_list_mod = []

for M in M_list:

    ber_list = []
    ser_list = []

    print("Starting BER vs SNR simulation...")
    for snr_dB in tqdm(snr_dB_list, desc="SNR sweep"):
        ber, ser = run_snr_point(snr_dB, Ns, sps, B, p, beta2, z, power_scale_norm, M, n_trials=n_trials)
        ber_list.append(ber)
        ser_list.append(ser)
        print(f"SNR = {snr_dB} dB, BER = {ber:.8f}, SER = {ser:.8f}")

    ber_list_mod.append(ber_list)
    ser_list_mod.append(ser_list)


# --------------- Plot results -----------------------------
plt.figure(figsize=(8, 6))
for i, M in enumerate(M_list):
    plt.semilogy(snr_dB_list, ber_list_mod[i], 'o-', label=f'BER {M}-QAM')
plt.xlabel('SNR (dB)')
plt.ylabel('Error Rate')
plt.title(f'Multiple QAM over Dispersive Channel with Noise')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('ber_vs_snr.png', dpi=150)
plt.show()

print("\nResults summary:")
print(" SNR(dB)     BER        SER")
for snr, ber, ser in zip(snr_dB_list, ber_list, ser_list):
    print(f" {snr:4.1f}     {ber:.2e}   {ser:.2e}")
