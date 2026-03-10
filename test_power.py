import numpy as np
from modulator import modul, demod

def test_power_conservation():
    B = 2.5
    Ns = 1000
    s = np.random.randn(Ns) + 1j * np.random.randn(Ns)
    symbol_power = np.mean(np.abs(s)**2)
    
    T = (Ns + 20) / B
    sps = 8
    dt = 1 / (B * sps)
    N = int(2**np.ceil(np.log2(T / dt)))
    T = N * dt
    t = np.linspace(-T/2, T/2, N, endpoint=False)
    
    q = modul(t, s, B)
    
    signal_power = np.mean(np.abs(q)**2)
    
    print(f"Symbol Power (mean|s|^2): {symbol_power:.6f}")
    print(f"Signal Power (mean|q|^2): {signal_power:.6f}")
    print(f"Ratio: {signal_power / symbol_power:.6f}")
    
    # Check recovery
    s_hat = np.array(demod(q, dt, B, Ns, t))
    error = np.mean(np.abs(s - s_hat)**2)
    print(f"Demod Error: {error:.4e}")

test_power_conservation()
