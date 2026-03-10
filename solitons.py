import numpy as np
import matplotlib.pyplot as plt

from channel import nl_channel
from parameters import gamma_norm, betav_norm

# -------------------------------------------------------------
# ------------------- Soliton Simulations ---------------------
# -------------------------------------------------------------
#
# All simulations use the normalized NLSE (eq. 4, noiseless):
#
#   j dq/dz = d²q/dt² + 2|q|²q
#
# implemented via nl_channel with sigma2=0 and betav=betav_norm=[0,0,+2],
# gamma=gamma_norm=-2.

# Convenience wrapper: propagate noiseless over total distance z
def propagate(t, q0t, z, Nsteps=None, B=200.0):
    f = np.fft.fftfreq(len(t), d=t[1] - t[0])
    if Nsteps is None:
        Nsteps = max(200, int(z * 500))
    qzt, _ = nl_channel(t, q0t, z=z, Nsteps=Nsteps,
                        gamma=gamma_norm, sigma2=0.0,
                        betav=betav_norm, f=f, B=B)
    return qzt


# ---------------------------------------------------------------
# Fundamental soliton  q(t,0) = A·sech(A·t)
# ---------------------------------------------------------------
def fundamental_soliton():
    # --------------- Simulation parameters --------------------
    T      = 40.0
    N      = 4096
    dt     = T / N
    t      = np.linspace(-T / 2, T / 2, N, endpoint=False)

    z      = 1.0                    # propagation distance

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, A in enumerate([1, 2]):

        q0t = A / np.cosh(A * t)    # A·sech(A·t)  (real-valued)

        # Propagate
        qzt = propagate(t, q0t, z=z)

        # The fundamental soliton accumulates a phase exp(j A² z) but its
        # amplitude |q| is perfectly preserved.
        err = np.max(np.abs(np.abs(q0t) - np.abs(qzt)))
        print(f"\n  A = {A}")
        print(f"    max | |q(t,0)| - |q(t,z=1)| | = {err:.6f}")
        print(f"    => shape {'preserved ✓' if err < 0.01 * A else 'changes ✗'}")

        # --------------- Plots --------------------------------

        axes[row, 0].plot(t, np.abs(q0t), label='|q(t, 0)| = TX', color='tab:blue')
        axes[row, 0].plot(t, np.abs(qzt), '--', label=f'|q(t, z={z})|', color='tab:orange')
        axes[row, 0].set_title(f'A = {A}  –  soliton amplitude  (z = {z})')
        axes[row, 0].set_xlabel('t')
        axes[row, 0].set_ylabel('|q|')
        axes[row, 0].set_xlim([-6 / A, 6 / A])
        axes[row, 0].legend(); axes[row, 0].grid(True)

        axes[row, 1].plot(t, np.angle(qzt / (q0t + 1e-15)), color='tab:green')
        axes[row, 1].set_title(f'A = {A}  –  accumulated phase  (z = {z})')
        axes[row, 1].set_xlabel('t')
        axes[row, 1].set_ylabel('arg(q(z) / q(0))  [rad]')
        axes[row, 1].set_xlim([-6 / A, 6 / A])
        axes[row, 1].grid(True)

    plt.suptitle('Fundamental soliton  q(t,0) = A·sech(A·t)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fund_soliton.png', dpi=150)
    # plt.show()
    print("\n  Saved: fund_soliton.png")


# ---------------------------------------------------------------
# Breathing snapshots (A=2)
# ---------------------------------------------------------------
def sy_breathing_snapshots(A=2.0):
    # Dynamic snapshot selection for Satsuma-Yajima breathing solitons.
    # We find the recovery point (z_recov) by minimizing the L2 distance
    # to the initial shape, then find max compression within that period.

    T = 60.0
    N = 4096
    t = np.linspace(-T / 2, T / 2, N, endpoint=False)
    q_init = A / np.cosh(t)
    abs_init = np.abs(q_init)

    z_max = 2.5  # pi/2 is approx 1.57
    Nz = 300
    z_vals = np.linspace(0, z_max, Nz + 1)
    
    qs = [q_init.copy()]
    diff_norms = [0.0]
    peak_vals = [abs_init.max()]
    
    print(f"\n  Analyzing breather A={A} ...")
    q_current = q_init.copy()
    
    for iz in range(1, Nz + 1):
        dz = z_vals[iz] - z_vals[iz - 1]
        q_current = propagate(t, q_current, z=dz, Nsteps=40)
        qs.append(q_current.copy())
        
        abs_curr = np.abs(q_current)
        # L2 norm of the difference in magnitude
        diff = np.sqrt(np.sum((abs_curr - abs_init)**2) * (t[1]-t[0]))
        diff_norms.append(diff)
        peak_vals.append(abs_curr.max())

    diff_norms = np.array(diff_norms)
    peak_vals = np.array(peak_vals)

    # 1. Find Recovery (z_recov): 
    # For A=integer, period is pi/2. For others, we look for the best return
    # to initial state after a significant travel.
    if A > 1.2:
        # We look for the minimum in a window excluding the very beginning
        # and focusing on the range where a cycle should occur.
        search_idx = np.where(z_vals > 0.4)[0]
        if len(search_idx) > 0:
            idx_recov = search_idx[np.argmin(diff_norms[search_idx])]
        else:
            idx_recov = Nz
    else:
        idx_recov = 0

    # 2. Find Compression (z_comp):
    # Max peak amplitude in the interval [0, idx_recov]
    idx_comp = np.argmax(peak_vals[:idx_recov+1])

    z_snap = {
        'Initial': 0.0,
        'Compression': z_vals[idx_comp],
        'Recovery': z_vals[idx_recov]
    }
    
    snapshots = {
        'Initial': qs[0],
        'Compression': qs[idx_comp],
        'Recovery': qs[idx_recov]
    }

    # --------------- Plots ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    snap_order = ['Initial', 'Compression', 'Recovery']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    t_win = np.abs(t) <= 10

    for ax, label, col in zip(axes, snap_order, colors):
        z_v = z_snap[label]
        q_v = snapshots[label]
        ax.plot(t[t_win], np.abs(q_v[t_win]), color=col, linewidth=1.5, label='|q(z)|')
        if label != 'Initial':
            ax.plot(t[t_win], np.abs(snapshots['Initial'][t_win]), '--',
                    color='gray', alpha=0.5, label='z=0')
        ax.set_title(f'{label}\n(z = {z_v:.3f})')
        ax.set_xlabel('t'); ax.set_ylabel('|q|')
        ax.grid(True); ax.legend(fontsize=8)

    plt.suptitle(f'Dynamic Breather Snapshots (A = {A})\n'
                 'Detected via pulse-shape similarity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('soliton_breather_snapshots.png', dpi=150)
    print(f"  Saved: soliton_breather_snapshots.png (A={A})")
    print(f"    Compression at z = {z_snap['Compression']:.4f}, peak = {peak_vals[idx_comp]:.2f}")
    print(f"    Recovery at    z = {z_snap['Recovery']:.4f}, error = {diff_norms[idx_recov]:.4f}")


# ---------------------------------------------------------------
# Elastic soliton collision
# ---------------------------------------------------------------
def elastic_collision():

    # --------------- Simulation parameters --------------------
    # Two solitons with amplitudes A1=1, A2=2.
    # Carrier frequencies ±ω0 give opposite group velocities ∓2ω0
    # (for eq.4, group velocity = -2ω from the linear dispersion relation).
    # They travel towards each other and collide near z≈1.5.

    A1     = 1.0
    A2     = 2.0
    omega0 = 2.5                    # carrier frequency
    t0     = 8.0                    # initial separation

    z_end  = 3.0                    # total propagation distance (covers full collision)
    Nsteps = 1500

    T  = 60.0
    N  = 8192
    dt = T / N
    t  = np.linspace(-T / 2, T / 2, N, endpoint=False)

    # Initial condition:
    # soliton 1 at +t0, moving left  (carrier +omega0, v = -2*omega0)
    # soliton 2 at -t0, moving right (carrier -omega0, v = +2*omega0)
    q0t = (A1 * np.exp( 1j * omega0 * t) / np.cosh(A1 * (t - t0))
         + A2 * np.exp(-1j * omega0 * t) / np.cosh(A2 * (t + t0)))

    print(f"  A1={A1}, A2={A2}, ω0={omega0}, t0={t0}")
    print(f"  Group velocity soliton 1: v1 = -2ω0 = {-2*omega0}  (moves left)")
    print(f"  Group velocity soliton 2: v2 = +2ω0 = {+2*omega0}  (moves right)")
    print(f"  Expected collision around z ≈ t0/(2ω0) = {t0/(2*omega0):.2f}")

    # Propagate and save snapshots: before / during / after collision
    # Collision at roughly z_coll = t0 / (2*omega0)
    z_coll = t0 / (2 * omega0)
    z_snap = {
        'Before':  z_coll * 0.4,
        'During':  z_coll,
        'After':   z_coll * 1.6
    }

    # Propagate step by step to collect snapshots
    snapshots = {}
    q_current = q0t.copy()
    z_current = 0.0
    z_snap_sorted = sorted(z_snap.items(), key=lambda x: x[1])

    print("\n  Propagating …")
    for label, z_target in z_snap_sorted:
        dz = z_target - z_current
        q_current = propagate(t, q_current, z=dz,
                              Nsteps=max(200, int(dz * 500)))
        snapshots[label] = (z_target, q_current.copy())
        z_current = z_target
        print(f"    z = {z_target:.3f}  →  '{label}'  "
              f"peak = {np.abs(q_current).max():.4f}")

    # Also save initial condition
    snapshots['z=0'] = (0.0, q0t.copy())

    # --------------- Spatiotemporal map -----------------------
    # Full space-time |q(t,z)| over [0, z_end]

    print("\n  Building space-time map …")
    Nz_map    = 300
    z_map_arr = np.linspace(0, z_end, Nz_map + 1)
    qt_map    = np.zeros((Nz_map + 1, N))
    q_run     = q0t.copy()
    qt_map[0] = np.abs(q_run)

    for iz in range(1, Nz_map + 1):
        dz_step = z_map_arr[iz] - z_map_arr[iz - 1]
        q_run   = propagate(t, q_run, z=dz_step, Nsteps=100)
        qt_map[iz] = np.abs(q_run)

    # --------------- Plots ------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Amplitude profiles at three z values
    snap_order = ['Before', 'During', 'After']
    colors_snap = ['tab:blue', 'tab:orange', 'tab:green']
    t_win_p = np.abs(t) <= 18

    for ax, label, col in zip(axes[0:], snap_order, colors_snap):
        z_val, q_snap = snapshots[label]
        ax.plot(t[t_win_p], np.abs(q_snap[t_win_p]), color=col, linewidth=1.2)
        ax.plot(t[t_win_p], np.abs(q0t[t_win_p]), '--', color='gray',
                alpha=0.5, linewidth=0.8, label='Initial |q|')
        ax.set_title(f'{label}  (z = {z_val:.2f})')
        ax.set_xlabel('t'); ax.set_ylabel('|q|')
        ax.set_xlim([-18, 18])
        ax.legend(fontsize=8); ax.grid(True)

        peak = np.abs(q_snap).max()
        print(f"  z = {z_val:.3f} ({label}):  max|q| = {peak:.4f}")

    plt.suptitle('Elastic soliton collision\n'
                 f'q(t,0) = A₁·exp(+jω₀t)·sech[A₁(t−t₀)] + A₂·exp(−jω₀t)·sech[A₂(t+t₀)]\n'
                 f'A₁={A1}, A₂={A2}, ω₀={omega0}, t₀={t0}',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('collision_soliton.png', dpi=150)
    # plt.show()
    print("\n  Saved: collision_soliton.png")

    # --------------- Elasticity check -------------------------
    # After the collision, |q(t,z)| should consist of two sech pulses
    # with the SAME amplitudes A1, A2 as before – only their positions/phases change.
    print("\n  === Elasticity check (after collision) ===")
    z_after, q_after = snapshots['After']
    abs_after = np.abs(q_after)

    # Find the two peaks manually to avoid scipy dependency
    # A simple peak: value > neighbors
    peaks_mask = (abs_after[1:-1] > abs_after[:-2]) & (abs_after[1:-1] > abs_after[2:])
    peaks_idx = np.where(peaks_mask)[0] + 1
    
    # Filter by height and distance
    peaks_idx = [i for i in peaks_idx if abs_after[i] > 0.3]
    
    peaks_val = abs_after[peaks_idx]
    peaks_t   = t[peaks_idx]
    peaks_sorted = sorted(zip(peaks_val, peaks_t), key=lambda x: x[0], reverse=True)[:2]
    print(f"  Two largest peaks after collision:")
    for pv, pt in sorted(peaks_sorted, key=lambda x: x[1]):
        print(f"    t = {pt:.2f},  |q| = {pv:.4f}")
    print(f"  Expected: A1={A1} and A2={A2}  (amplitudes conserved)")


def main_solitons():

    #fundamental_soliton()
    sy_breathing_snapshots(A=3.5)
    #elastic_collision()


if __name__ == "__main__":
    main_solitons()
