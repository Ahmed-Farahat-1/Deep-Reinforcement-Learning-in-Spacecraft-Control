"""
GNC Performance Metrics for Spacecraft Attitude Stabilization and Control.

Provides rigorous, aerospace-standard metric calculations:
- Settling Time: Formally defined around the true asymptotic equilibrium (including steady-state offsets).
- Error Decay Time (90% to 10%): Replaces ambiguous "rise time" for attitude regulation maneuvers.
- Maximum Overshoot: Computed relative to the initial-to-steady-state excursion.
- Periodic Disturbance Metrics: Mean, RMS, peak amplitude, and peak-to-peak oscillation over complete cycles.
- Pulse Recovery Time: Measured both from simulation start and from pulse cessation.
- Separate Saturation Diagnostics: Distinct tracking for motor torque limits and wheel rotor speed limits.
- Squared Control Torque Integral: Mechanical control effort int_0^T u^2(t) dt (N^2*m^2*s).
- Momentum Accumulation Analysis: Empirical vs. theoretical spin-up rates and time-to-saturation projections.

All metrics strictly adhere to AIAA / IEEE guidance, navigation, and control conventions.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


def compute_settling_time(
    time: np.ndarray,
    theta: np.ndarray,
    target: float = 0.0,
    steady_state_val: Optional[float] = None,
    tolerance_pct: float = 0.02,
    tolerance_abs: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate settling time Ts to enter and remain within a defined error band.

    For regulation to zero with zero disturbance:
        Settles around target (default: 0.0).
    For systems with a non-zero steady-state offset (e.g., constant disturbance theta_ss = Md / Kp):
        Settles around steady_state_val, with band = tolerance_pct * |theta_0 - steady_state_val|.

    Parameters
    ----------
    time : np.ndarray
        Simulation time array (s).
    theta : np.ndarray
        Attitude angle trajectory (rad).
    target : float, optional
        Nominal commanded target setpoint, default: 0.0.
    steady_state_val : Optional[float], optional
        Theoretical or empirical steady-state value. If None, defaults to target.
    tolerance_pct : float, optional
        Percentage band based on total excursion |theta_0 - steady_state|, default: 0.02 (2%).
    tolerance_abs : Optional[float], optional
        Optional absolute minimum tolerance threshold (rad).

    Returns
    -------
    Optional[float]
        Settling time in seconds, or None if the response does not settle within the simulation window.
    """
    ref = target if steady_state_val is None else steady_state_val
    initial_excursion = abs(theta[0] - ref)

    if initial_excursion < 1e-12:
        return 0.0

    band = tolerance_pct * initial_excursion
    if tolerance_abs is not None:
        band = max(band, tolerance_abs)

    error = np.abs(theta - ref)
    outside_indices = np.where(error > band)[0]

    if len(outside_indices) == 0:
        return 0.0  # Started and remained within band

    last_outside = outside_indices[-1]
    if last_outside >= len(time) - 1:
        return None  # Never remained inside band before simulation end

    return float(time[last_outside + 1])


def compute_decay_time(
    time: np.ndarray,
    theta: np.ndarray,
    steady_state_val: float = 0.0,
) -> Optional[float]:
    """
    Calculate Error Decay Time (time required for the attitude deviation to decrease
    from 90% of initial excursion to 10% of initial excursion).

    Appropriate substitute for 'rise time' in initial-condition regulation problems.

    Parameters
    ----------
    time : np.ndarray
        Simulation time array (s).
    theta : np.ndarray
        Attitude trajectory (rad).
    steady_state_val : float, optional
        Asymptotic equilibrium attitude, default: 0.0.

    Returns
    -------
    Optional[float]
        Decay time in seconds (t_10pct - t_90pct), or None if thresholds are not crossed.
    """
    initial_dev = theta[0] - steady_state_val
    if abs(initial_dev) < 1e-12:
        return 0.0

    # Normalized error progress: 0.0 at t=0, approaching 1.0 as theta -> steady_state_val
    progress = 1.0 - (theta - steady_state_val) / initial_dev

    idx_10 = np.where(progress >= 0.10)[0]
    idx_90 = np.where(progress >= 0.90)[0]

    if len(idx_10) == 0 or len(idx_90) == 0:
        return None

    t_10 = time[idx_10[0]]
    t_90 = time[idx_90[0]]
    return float(t_90 - t_10)


def compute_overshoot(
    theta: np.ndarray,
    steady_state_val: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate maximum overshoot magnitude and percentage relative to asymptotic steady state.

    Overshoot is defined as the maximum transient excursion beyond steady_state_val:
        overshoot_pct = (overshoot_rad / |theta_0 - steady_state_val|) * 100%

    Parameters
    ----------
    theta : np.ndarray
        Attitude trajectory (rad).
    steady_state_val : float, optional
        Asymptotic reference attitude, default: 0.0.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'overshoot_pct', 'overshoot_rad', 'overshoot_deg'.
    """
    initial_excursion = theta[0] - steady_state_val
    if abs(initial_excursion) < 1e-12:
        return {"overshoot_pct": 0.0, "overshoot_rad": 0.0, "overshoot_deg": 0.0}

    if initial_excursion > 0:
        # Starting above steady state: overshoot occurs if theta drops below steady state
        min_theta = np.min(theta)
        overshoot_rad = max(0.0, steady_state_val - min_theta)
    else:
        # Starting below steady state: overshoot occurs if theta rises above steady state
        max_theta = np.max(theta)
        overshoot_rad = max(0.0, max_theta - steady_state_val)

    overshoot_pct = (overshoot_rad / abs(initial_excursion)) * 100.0
    overshoot_deg = float(np.degrees(overshoot_rad))

    return {
        "overshoot_pct": float(overshoot_pct),
        "overshoot_rad": float(overshoot_rad),
        "overshoot_deg": overshoot_deg,
    }


def compute_periodic_metrics(
    time: np.ndarray,
    theta: np.ndarray,
    period: float = 10.0,
    num_cycles: int = 1,
) -> Dict[str, float]:
    """
    Compute steady-state metrics over complete periodic disturbance cycles.

    Parameters
    ----------
    time : np.ndarray
        Time points (s).
    theta : np.ndarray
        Attitude angle trajectory (rad).
    period : float, optional
        Period of the disturbance T = 1/f in seconds, default: 10.0 s.
    num_cycles : int, optional
        Number of final full cycles to evaluate over, default: 1.

    Returns
    -------
    Dict[str, float]
        Periodic steady-state statistics: mean, rms, peak amplitude, peak-to-peak.
    """
    t_end = time[-1]
    t_start_eval = t_end - num_cycles * period

    mask = (time >= t_start_eval) & (time <= t_end)
    th_eval = theta[mask]

    mean_rad = float(np.mean(th_eval))
    rms_rad = float(np.sqrt(np.mean(th_eval**2)))
    peak_amp_rad = float(np.max(np.abs(th_eval - mean_rad)))
    p2p_rad = float(np.ptp(th_eval))

    return {
        "periodic_mean_rad": mean_rad,
        "periodic_mean_deg": float(np.degrees(mean_rad)),
        "periodic_rms_rad": rms_rad,
        "periodic_rms_deg": float(np.degrees(rms_rad)),
        "periodic_peak_amp_rad": peak_amp_rad,
        "periodic_peak_amp_deg": float(np.degrees(peak_amp_rad)),
        "periodic_p2p_rad": p2p_rad,
        "periodic_p2p_deg": float(np.degrees(p2p_rad)),
    }


def compute_pulse_recovery_metrics(
    time: np.ndarray,
    theta: np.ndarray,
    pulse_end_time: float = 12.0,
    target: float = 0.0,
    tolerance_pct: float = 0.02,
) -> Dict[str, Optional[float]]:
    """
    Compute recovery metrics for pulse disturbance scenarios.

    Reports:
    - Initial settling time (before pulse arrival).
    - Absolute settling time (from simulation t=0).
    - Recovery time (measured from pulse cessation t_end).
    - Maximum transient deflection during/after the pulse.

    Parameters
    ----------
    time : np.ndarray
        Time points (s).
    theta : np.ndarray
        Attitude angle trajectory (rad).
    pulse_end_time : float
        Time at which the pulse disturbance terminates (s).
    target : float
        Target attitude (rad).
    tolerance_pct : float
        Tolerance band percentage based on initial attitude displacement.

    Returns
    -------
    Dict[str, Optional[float]]
        Pulse response and recovery metrics.
    """
    initial_error = abs(theta[0] - target)
    band = tolerance_pct * initial_error

    # 1. Pre-pulse settling
    pre_mask = time < 10.0
    outside_pre = np.where(np.abs(theta[pre_mask] - target) > band)[0]
    initial_settling = float(time[outside_pre[-1] + 1]) if len(outside_pre) > 0 else 0.0

    # 2. Maximum deflection during pulse
    pulse_mask = time >= 10.0
    max_deflection_rad = float(np.max(np.abs(theta[pulse_mask] - target)))

    # 3. Post-pulse settling from t=0
    outside_all = np.where(np.abs(theta - target) > band)[0]
    if len(outside_all) == 0:
        absolute_settling = 0.0
        recovery_time = 0.0
    else:
        last_out = outside_all[-1]
        if last_out >= len(time) - 1:
            absolute_settling = None
            recovery_time = None
        else:
            absolute_settling = float(time[last_out + 1])
            recovery_time = float(max(0.0, absolute_settling - pulse_end_time))

    return {
        "initial_settling_time_s": initial_settling,
        "absolute_settling_time_s": absolute_settling,
        "recovery_time_s": recovery_time,
        "pulse_max_deflection_deg": float(np.degrees(max_deflection_rad)),
        "pulse_max_deflection_rad": max_deflection_rad,
    }


def analyze_saturation(
    time: np.ndarray,
    torque: np.ndarray,
    max_torque: float,
    Omega: np.ndarray,
    max_speed: float,
) -> Dict[str, Any]:
    """
    Perform rigorous, separate saturation analysis for motor torque and wheel rotor speed.

    Parameters
    ----------
    time : np.ndarray
        Time array (s).
    torque : np.ndarray
        Applied motor torque trajectory (N*m).
    max_torque : float
        Motor torque saturation limit |u_max| (N*m).
    Omega : np.ndarray
        Reaction wheel relative speed trajectory (rad/s).
    max_speed : float
        Rotor speed limit |Omega_max| (rad/s).

    Returns
    -------
    Dict[str, Any]
        Saturation diagnostics: max values, durations, event counts, capacity usage.
    """
    dt = time[1] - time[0] if len(time) > 1 else 0.01

    # Motor Torque Saturation Diagnostics
    torque_mag = np.abs(torque)
    max_torque_applied = float(np.max(torque_mag))
    is_torque_sat = torque_mag >= (max_torque - 1e-5)
    torque_sat_steps = int(np.sum(is_torque_sat))
    torque_sat_duration_s = float(torque_sat_steps * dt)

    # Count contiguous torque saturation events
    torque_events = 0
    in_torque_event = False
    for sat in is_torque_sat:
        if sat and not in_torque_event:
            torque_events += 1
            in_torque_event = True
        elif not sat:
            in_torque_event = False

    # Wheel Speed Saturation Diagnostics
    wheel_mag = np.abs(Omega)
    max_wheel_speed_rad_s = float(np.max(wheel_mag))
    max_wheel_speed_rpm = float(max_wheel_speed_rad_s * 60.0 / (2.0 * np.pi))
    is_wheel_sat = wheel_mag >= (max_speed - 1e-4)
    wheel_sat_steps = int(np.sum(is_wheel_sat))
    wheel_sat_duration_s = float(wheel_sat_steps * dt)

    wheel_events = 0
    in_wheel_event = False
    for sat in is_wheel_sat:
        if sat and not in_wheel_event:
            wheel_events += 1
            in_wheel_event = True
        elif not sat:
            in_wheel_event = False

    torque_capacity_pct = float((max_torque_applied / max_torque) * 100.0)
    wheel_capacity_pct = float((max_wheel_speed_rad_s / max_speed) * 100.0)

    return {
        "max_torque_applied_Nm": max_torque_applied,
        "torque_capacity_used_pct": torque_capacity_pct,
        "torque_sat_duration_s": torque_sat_duration_s,
        "torque_sat_events": torque_events,
        "max_wheel_speed_rad_s": max_wheel_speed_rad_s,
        "max_wheel_speed_rpm": max_wheel_speed_rpm,
        "wheel_capacity_used_pct": wheel_capacity_pct,
        "wheel_sat_duration_s": wheel_sat_duration_s,
        "wheel_sat_events": wheel_events,
    }


def analyze_constant_disturbance_momentum(
    time: np.ndarray,
    Omega: np.ndarray,
    Md: float = 0.02,
    I_wheel: float = 0.05,
    max_speed: float = 523.6,
) -> Dict[str, float]:
    """
    Analyze reaction wheel momentum accumulation and time-to-saturation under constant disturbance.

    Theory:
        u_steady = Md
        dOmega/dt = Md / I_wheel
        t_sat = (Omega_max - Omega_current) / (Md / I_wheel)

    Parameters
    ----------
    time : np.ndarray
        Time array (s).
    Omega : np.ndarray
        Wheel speed trajectory (rad/s).
    Md : float
        Constant disturbance torque (N*m).
    I_wheel : float
        Wheel moment of inertia (kg*m^2).
    max_speed : float
        Maximum wheel speed limit (rad/s).

    Returns
    -------
    Dict[str, float]
        Momentum accumulation metrics and extrapolated time to saturation.
    """
    theoretical_dOmega_dt = float(Md / I_wheel)  # rad/s^2
    theoretical_dOmega_rpm_s = float(theoretical_dOmega_dt * 60.0 / (2.0 * np.pi))

    # Calculate empirical slope over final 10 seconds of simulation
    mask = time >= (time[-1] - 10.0)
    t_sub = time[mask]
    omega_sub = Omega[mask]
    empirical_dOmega_dt = float((omega_sub[-1] - omega_sub[0]) / (t_sub[-1] - t_sub[0]))
    empirical_dOmega_rpm_s = float(empirical_dOmega_dt * 60.0 / (2.0 * np.pi))

    final_speed = Omega[-1]
    remaining_headroom = max_speed - final_speed
    time_to_sat_from_end = remaining_headroom / theoretical_dOmega_dt
    total_projected_time_to_sat_s = float(time[-1] + time_to_sat_from_end)
    total_projected_time_to_sat_min = float(total_projected_time_to_sat_s / 60.0)

    return {
        "theoretical_dOmega_dt_rad_s2": theoretical_dOmega_dt,
        "theoretical_dOmega_dt_rpm_s": theoretical_dOmega_rpm_s,
        "empirical_dOmega_dt_rad_s2": empirical_dOmega_dt,
        "empirical_dOmega_dt_rpm_s": empirical_dOmega_rpm_s,
        "time_to_saturation_s": total_projected_time_to_sat_s,
        "time_to_saturation_min": total_projected_time_to_sat_min,
    }


def compute_squared_torque_integral(time: np.ndarray, torque: np.ndarray) -> float:
    """
    Compute Squared Control Torque Integral: int_0^T u^2(t) dt.

    Units: N^2 * m^2 * s.
    Represents mechanical control effort (not electrical energy).
    """
    u_sq = torque**2
    trapz_func = getattr(np, "trapezoid", np.trapz)
    return float(trapz_func(u_sq, time))


def calculate_comprehensive_metrics(
    time: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    Omega: np.ndarray,
    torque: np.ndarray,
    scenario_id: str,
    target_theta: float = 0.0,
    theoretical_theta_ss: float = 0.0,
    max_torque_limit: float = 1.0,
    max_speed_limit: float = 523.6,
    disturbance_torque: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute unified, scenario-aware GNC performance metrics for attitude stabilization.
    """
    # 1. Settling Time handling per scenario type
    is_periodic = "sinusoidal" in scenario_id
    is_pulse = "pulse" in scenario_id
    has_constant_dist = "constant" in scenario_id

    # Reference attitude for settling and overshoot
    ref_steady_state = theoretical_theta_ss if has_constant_dist else target_theta

    if is_periodic:
        settling_time = None  # Persistent oscillation, no static settling time
        decay_time = None
    else:
        settling_time = compute_settling_time(
            time, theta, target=target_theta, steady_state_val=ref_steady_state, tolerance_pct=0.02
        )
        decay_time = compute_decay_time(time, theta, steady_state_val=ref_steady_state)

    # 2. Overshoot relative to actual steady state
    if is_periodic:
        overshoot_dict = {"overshoot_pct": None, "overshoot_deg": None, "overshoot_rad": None}
    else:
        overshoot_dict = compute_overshoot(theta, steady_state_val=ref_steady_state)

    # 3. Control Effort (Squared Control Torque Integral)
    effort = compute_squared_torque_integral(time, torque)

    # 4. Saturation Diagnostics
    sat_dict = analyze_saturation(time, torque, max_torque_limit, Omega, max_speed_limit)

    # 5. Final state values
    final_error_rad = abs(float(theta[-1] - target_theta))
    final_error_deg = float(np.degrees(final_error_rad))

    final_offset_from_ss_rad = abs(float(theta[-1] - ref_steady_state))
    final_offset_from_ss_deg = float(np.degrees(final_offset_from_ss_rad))

    # Base metrics dictionary
    metrics = {
        "scenario_id": scenario_id,
        "settling_time_s": settling_time,
        "decay_time_90_to_10_s": decay_time,
        "overshoot_pct": overshoot_dict["overshoot_pct"],
        "overshoot_deg": overshoot_dict["overshoot_deg"],
        "final_attitude_deg": float(np.degrees(theta[-1])),
        "final_error_from_target_deg": final_error_deg,
        "final_offset_from_ss_deg": final_offset_from_ss_deg,
        "theoretical_steady_state_deg": float(np.degrees(ref_steady_state)),
        "squared_control_torque_integral_N2m2s": effort,
        "max_torque_applied_Nm": sat_dict["max_torque_applied_Nm"],
        "torque_capacity_used_pct": sat_dict["torque_capacity_used_pct"],
        "torque_sat_duration_s": sat_dict["torque_sat_duration_s"],
        "torque_sat_events": sat_dict["torque_sat_events"],
        "max_wheel_speed_rpm": sat_dict["max_wheel_speed_rpm"],
        "max_wheel_speed_rad_s": sat_dict["max_wheel_speed_rad_s"],
        "wheel_capacity_used_pct": sat_dict["wheel_capacity_used_pct"],
        "wheel_sat_duration_s": sat_dict["wheel_sat_duration_s"],
        "wheel_sat_events": sat_dict["wheel_sat_events"],
    }

    # Add scenario-specific metrics
    if is_periodic:
        p_metrics = compute_periodic_metrics(time, theta, period=10.0, num_cycles=1)
        metrics.update(p_metrics)

    if is_pulse:
        pulse_metrics = compute_pulse_recovery_metrics(time, theta, pulse_end_time=12.0, target=target_theta)
        metrics.update(pulse_metrics)

    if has_constant_dist:
        mom_metrics = analyze_constant_disturbance_momentum(
            time, Omega, Md=0.02, I_wheel=0.05, max_speed=max_speed_limit
        )
        metrics.update(mom_metrics)

    return metrics


# Backward-compatible alias
calculate_performance_metrics = calculate_comprehensive_metrics

