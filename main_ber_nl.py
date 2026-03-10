import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from miscellaneous import run_nl_snr_point
from parameters import (
    beta2_norm, sigma_sq, gamma_norm, Bn, P_norm, power_scale_norm
)

def main_ber_nl():
    print("=== BER vs SNR in Nonlinear Channel (DBP Equalized) ===")
    
    # --------------- System parameters ------------------------
    B = Bn                          # normalized bandwidth
    Ns = 500                        # number of symbols per trial
    sps = 8                         # samples per symbol
    p = 0.5
    z = 1.0                         # normalized distance
    Nsteps = 100                    # SSFM steps
    betav = [0, 0, beta2_norm]
    
    # --------------- SNR range --------------------------------
    snr_dB_list = np.arange(0, 63, 3)
    n_trials = 20                 # Balanced for speed and visibility
    M_list = [4, 16] 

    plt.figure(figsize=(10, 7))

    for M in M_list:
        ber_list_eq = []
        ber_list_no_eq = []
        print(f"\nEvaluating {M}-QAM...")
        for snr_dB in tqdm(snr_dB_list, desc=f"SNR sweep {M}-QAM"):
            ber_eq, ber_no_eq = run_nl_snr_point(
                snr_dB, Ns, sps, B, p, betav, z, Nsteps, gamma_norm, 
                sigma_sq, M, n_trials=n_trials
            )
            ber_list_eq.append(ber_eq)
            ber_list_no_eq.append(ber_no_eq)
        
        plt.semilogy(snr_dB_list, ber_list_eq, 'o-', label=f'DBP {M}-QAM')
        plt.semilogy(snr_dB_list, ber_list_no_eq, '--', label=f'No EQ {M}-QAM')
        plt.semilogy(snr_dB_list, ber_list_no_eq, '--', label=f'No EQ {M}-QAM')

    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR for 4/16-QAM in Nonlinear Fiber (DBP Equalized)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ber_vs_snr_nl.png', dpi=150)
    print("\nSaved plot: ber_vs_snr_nl.png")
    # plt.show()

if __name__ == "__main__":
    main_ber_nl()
