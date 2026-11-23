"""
Plotting and Visualization Utilities for Spacecraft Attitude Control.

Generates engineering-grade publication plots for:
- Attitude Angle theta(t) [deg]
- Spacecraft Angular Velocity theta_dot(t) [deg/s]
- Reaction Wheel Speed Omega(t) [RPM and rad/s]
- Control Torque u(t) and Disturbance Torque M_d(t) [N*m]
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_simulation_run(
    time: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    Omega: np.ndarray,
    torque: np.ndarray,
    disturbance: Optional[np.ndarray] = None,
    title: str = "Spacecraft Attitude Stabilization — PD Control",
    save_path: Optional[Path] = None,
    target_theta: float = 0.0,
    theoretical_theta_ss: Optional[float] = None,
    max_torque_limit: float = 1.0,
    max_speed_limit: float = 523.6,
    is_periodic: bool = False,
    pulse_window: Optional[tuple] = None,
    periodic_envelope_deg: Optional[float] = None,
) -> plt.Figure:
    """
    Generate a 4-panel diagnostic plot for an attitude simulation run.

    Panels:
    1. Attitude Angle theta(t) [deg] with setpoint and verified settling band / envelope.
    2. Spacecraft Angular Velocity theta_dot(t) [deg/s].
    3. Reaction Wheel Relative Speed Omega(t) [RPM and rad/s] with speed limits.
    4. Control Torque u(t) [N*m] and Disturbance Torque M_d(t) [N*m] with torque limits.
    """
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)

    t_end = time[-1]
    theta_deg = np.degrees(theta)
    theta_dot_deg = np.degrees(theta_dot)
    target_deg = np.degrees(target_theta)
    omega_rpm = Omega * 60.0 / (2.0 * np.pi)
    max_speed_rpm = max_speed_limit * 60.0 / (2.0 * np.pi)

    # 1. Attitude Angle Panel
    ax1 = axes[0]
    ax1.plot(time, theta_deg, "b-", linewidth=2.0, label=r"Attitude $\theta(t)$")
    ax1.axhline(target_deg, color="k", linestyle="--", alpha=0.7, label=r"Command Target $\theta_{\rm target} = 0^\circ$")

    if is_periodic:
        # Periodic disturbance scenario: Show periodic envelope rather than static settling band
        env_val = periodic_envelope_deg if periodic_envelope_deg is not None else 0.180
        ax1.axhline(env_val, color="m", linestyle=":", alpha=0.8, label=rf"Periodic Bounds $\pm {env_val:.3f}^\circ$")
        ax1.axhline(-env_val, color="m", linestyle=":", alpha=0.8)
        ax1.axhspan(-env_val, env_val, color="m", alpha=0.08, label="Steady-State Oscillation Band")
    elif theoretical_theta_ss is not None and abs(theoretical_theta_ss) > 1e-6:
        # Constant disturbance scenario: Show theoretical steady-state line & band around it
        ss_deg = np.degrees(theoretical_theta_ss)
        ax1.axhline(ss_deg, color="r", linestyle="-.", alpha=0.85, label=rf"Theoretical $\theta_{{ss}} = M_d/K_p = {ss_deg:.4f}^\circ$")
        # 2% band of initial excursion |theta_0 - theta_ss|
        band = 0.02 * abs(theta_deg[0] - ss_deg)
        ax1.axhspan(ss_deg - band, ss_deg + band, color="g", alpha=0.18, label=rf"$\pm 2\%$ Settling Band around $\theta_{{ss}}$ ($\pm{band:.3f}^\circ$)")
    else:
        # Nominal zero-disturbance regulation
        band = max(0.286, 0.02 * abs(theta_deg[0] - target_deg))
        ax1.axhspan(target_deg - band, target_deg + band, color="g", alpha=0.15, label=rf"$\pm 2\%$ Settling Band ($\pm{band:.2f}^\circ$)")

    ax1.set_ylabel(r"$\theta$ [deg]", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # 2. Angular Velocity Panel
    ax2 = axes[1]
    ax2.plot(time, theta_dot_deg, "r-", linewidth=1.8, label=r"Body Rate $\dot{\theta}(t)$")
    ax2.axhline(0.0, color="k", linestyle=":", alpha=0.5)
    ax2.set_ylabel(r"$\dot{\theta}$ [deg/s]", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # 3. Reaction Wheel Speed Panel
    ax3 = axes[2]
    ax3.plot(time, omega_rpm, "m-", linewidth=1.8, label=r"Wheel Speed $\Omega(t)$")
    ax3.axhline(max_speed_rpm, color="r", linestyle="--", alpha=0.7, label=rf"$\pm \Omega_{{\max}} = \pm {max_speed_rpm:.0f}$ RPM Limit")
    ax3.axhline(-max_speed_rpm, color="r", linestyle="--", alpha=0.7)
    ax3.set_ylabel(r"$\Omega$ [RPM]", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # 4. Control & Disturbance Torques Panel
    ax4 = axes[3]
    ax4.plot(time, torque, "c-", linewidth=1.8, label=r"Control Torque $u(t)$")
    if disturbance is not None and np.any(np.abs(disturbance) > 1e-12):
        ax4.plot(time, disturbance, "k--", linewidth=1.4, alpha=0.85, label=r"Disturbance $M_d(t)$")
    ax4.axhline(max_torque_limit, color="r", linestyle=":", alpha=0.6, label=rf"$\pm u_{{\max}} = \pm {max_torque_limit:.1f}$ N$\cdot$m Limit")
    ax4.axhline(-max_torque_limit, color="r", linestyle=":", alpha=0.6)
    ax4.set_ylabel(r"Torque [N$\cdot$m]", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Time [s]", fontsize=10, fontweight="bold")
    ax4.set_xlim(0, t_end)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Highlight pulse disturbance window if present
    if pulse_window is not None:
        t_p_start, t_p_end = pulse_window
        for ax in axes:
            ax.axvspan(t_p_start, t_p_end, color="orange", alpha=0.15)
        axes[0].text(
            0.5 * (t_p_start + t_p_end),
            0.85 * np.max(theta_deg),
            "Pulse Active\n(10s - 12s)",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6),
        )

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_comparative_scenarios(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate an overlaid comparative plot across multiple simulation scenarios.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Classical PD Attitude Control — Cross-Scenario Performance Comparison", fontsize=14, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (label, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        t = data["time"]
        theta_deg = np.degrees(data["theta"])
        theta_dot_deg = np.degrees(data["theta_dot"])
        omega_rpm = data["Omega"] * 60.0 / (2.0 * np.pi)
        torque = data["torque"]

        axes[0, 0].plot(t, theta_deg, label=label, color=color, linewidth=1.7)
        axes[0, 1].plot(t, theta_dot_deg, label=label, color=color, linewidth=1.7)
        axes[1, 0].plot(t, omega_rpm, label=label, color=color, linewidth=1.7)
        axes[1, 1].plot(t, torque, label=label, color=color, linewidth=1.7)

    # 1. Attitude
    axes[0, 0].set_title(r"Spacecraft Attitude $\theta(t)$", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel(r"$\theta$ [deg]", fontsize=10)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].legend(fontsize=8.5, loc="best", framealpha=0.9)

    # 2. Body Rate
    axes[0, 1].set_title(r"Body Angular Velocity $\dot{\theta}(t)$", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel(r"$\dot{\theta}$ [deg/s]", fontsize=10)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].legend(fontsize=8.5, loc="best", framealpha=0.9)

    # 3. Wheel Speed
    axes[1, 0].set_title(r"Reaction Wheel Speed $\Omega(t)$", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Time [s]", fontsize=10)
    axes[1, 0].set_ylabel(r"$\Omega$ [RPM]", fontsize=10)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(fontsize=8.5, loc="best", framealpha=0.9)

    # 4. Control Torque
    axes[1, 1].set_title(r"Motor Control Torque $u(t)$", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Time [s]", fontsize=10)
    axes[1, 1].set_ylabel(r"Torque [N$\cdot$m]", fontsize=10)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(fontsize=8.5, loc="best", framealpha=0.9)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
