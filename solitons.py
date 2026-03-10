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
# Satsuma–Yajima solitons  q(t,0) = A·sech(t)
# ---------------------------------------------------------------
def sy_solitons():
    # --------------- Simulation parameters --------------------
    T      = 40.0
    N      = 4096
    dt     = T / N
    t      = np.linspace(-T / 2, T / 2, N, endpoint=False)

    z_max  = 2 * np.pi              # propagate one full 'super-period'
    Nz     = 200                    # number of z snapshots for the space-time map
    z_vals = np.linspace(0, z_max, Nz + 1)

    # A values to explore
    A_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

    fig_st, axes_st = plt.subplots(2, 4, figsize=(16, 7))
    axes_st = axes_st.flatten()

    for idx, A in enumerate(A_values):

        q_current = A / np.cosh(t)  # A·sech(t)

        # Build space-time array  [Nz+1, N]
        qt_map = np.zeros((Nz + 1, N))
        qt_map[0] = np.abs(q_current)

        for iz in range(1, Nz + 1):
            dz_step = z_vals[iz] - z_vals[iz - 1]
            q_current = propagate(t, q_current, z=dz_step, Nsteps=50)
            qt_map[iz] = np.abs(q_current)

        # Find period by looking at the peak amplitude vs z
        peak_vs_z = qt_map.max(axis=1)

        # Display the space-time map
        t_idx = np.abs(t) <= 8
        ax = axes_st[idx]
        im = ax.imshow(qt_map[:, t_idx].T,
                       aspect='auto',
                       origin='lower',
                       extent=[0, z_max, t[t_idx][0], t[t_idx][-1]],
                       cmap='inferno')
        ax.set_title(f'A = {A}')
        ax.set_xlabel('z'); ax.set_ylabel('t')
        fig_st.colorbar(im, ax=ax, shrink=0.7)

        print(f"\n  A = {A}:")
        print(f"    Peak amplitude at z=0:      {peak_vs_z[0]:.4f}  (= A = {A})")
        print(f"    Peak amplitude at z=pi/2:   {peak_vs_z[Nz // 4]:.4f}")
        print(f"    Peak amplitude at z=pi:     {peak_vs_z[Nz // 2]:.4f}")
        print(f"    Peak amplitude at z=2pi:    {peak_vs_z[-1]:.4f}")

    plt.suptitle('Question 32: Satsuma–Yajima solitons  q(t,0) = A·sech(t)  –  |q(t,z)|',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('soliton_q32_map.png', dpi=150)
    # plt.show()
    print("\n  Saved: soliton_q32_map.png")

    # --------------- Peak-amplitude vs z for all A ------------

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    A_plot = [1, 2, 3, 4, 5, 6]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(A_plot)))

    for A, col in zip(A_plot, colors):
        q_current = A / np.cosh(t)
        peak = [np.abs(q_current).max()]
        for iz in range(1, Nz + 1):
            dz_step = z_vals[iz] - z_vals[iz - 1]
            q_current = propagate(t, q_current, z=dz_step, Nsteps=50)
            peak.append(np.abs(q_current).max())
        ax2.plot(z_vals, peak, color=col, label=f'A = {A}')

    ax2.axvline(np.pi / 2, color='gray', linestyle='--', linewidth=0.8, label='z = π/2')
    ax2.axvline(np.pi,     color='gray', linestyle=':',  linewidth=0.8, label='z = π')
    ax2.set_xlabel('z (normalized)')
    ax2.set_ylabel('max|q(t, z)|')
    ax2.set_title('Peak amplitude vs z  –  Satsuma–Yajima solitons  q(t,0) = A·sech(t)')
    ax2.legend(ncol=2); ax2.grid(True)
    plt.tight_layout()
    plt.savefig('soliton_q32_peak.png', dpi=150)
    # plt.show()
    print("  Saved: soliton_q32_peak.png")


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
    omega0 = 1.0                    # carrier frequency
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

    plt.suptitle('Question 33: Elastic soliton collision\n'
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

    fundamental_soliton()
    sy_solitons()
    elastic_collision()


if __name__ == "__main__":
    main_solitons()
