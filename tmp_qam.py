import numpy as np
import matplotlib.pyplot as plt

from mapper import mapper, generate_constellation
from source import source_bernoulli

scale_power_factor = np.sqrt(6*10**-3)                   # total power in mW

for M in [2, 4, 8, 16, 32, 64]:
    c = generate_constellation(M)
    p = np.mean(np.abs(c)**2)

    b = source_bernoulli(1000*int(np.log2(M)), 0.5)

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    s_unit = mapper(b, const=M, normalize=True)
    s_unit = s_unit * scale_power_factor
    p = np.mean(np.abs(s_unit)**2)
    print(f"Power of {M}-QAM constellation: {p}")
    axes.scatter(np.real(s_unit), np.imag(s_unit), marker='x', color='blue')
    axes.set_title(f'{M}-QAM Constellation (Average Power = {np.round(p*10**3)} mW)')
    axes.set_xlabel('In-phase')
    axes.set_ylabel('Quadrature')
    axes.grid(True)
    axes.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'constellation_{M}.png', dpi=150)
    plt.show()

