"""
Comprehensive Comparative Evaluation Engine for Phase 4E (PPO vs Classical PD).

Evaluates the selected PPO controller against the audited Classical PD Controller
across all benchmark canonical scenarios and 5 held-out generalization scenarios.

Computes the 12 required aerospace control metrics:
1. Settling Time (Ts)
2. Error Decay Time
3. RMS Attitude Error
4. Maximum Overshoot (Mp)
5. Final Attitude Error
6. Squared Control Torque Integral (Effort)
7. Maximum Wheel Speed (Peak Omega)
8. Motor Torque Saturation Duration
9. Wheel Speed Saturation Duration
10. Pulse Recovery Time
11. Stabilization Success Rate
12. Physical Failure Rate
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.controllers.pd_controller import PDController
from src.simulation.spacecraft import Spacecraft1D
from src.simulation.disturbances import (
    ZeroDisturbance,
    ConstantDisturbance,
    SinusoidalDisturbance,
    PulseDisturbance,
)
from src.evaluation.metrics import (
    calculate_comprehensive_metrics,
    compute_squared_torque_integral,
    compute_decay_time,
    compute_pulse_recovery_metrics,
)
from src.agents.select_best_checkpoint import simulate_ppo_trajectory
from src.agents.evaluate_ppo import get_benchmark_scenarios, get_heldout_scenarios


def simulate_pd_trajectory(
    pd_controller: PDController,
    initial_state: List[float],
    disturbance: Any,
    duration: float = 30.0,
    dt: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Simulate deterministic closed-loop trajectory under Classical PD control."""
    sim = Spacecraft1D(dt=dt)
    sim.reset(initial_state=np.array(initial_state, dtype=np.float64))

    n_steps = int(round(duration / dt))
    t_list = []
    th_list = []
    th_dot_list = []
    om_list = []
    u_cmd_list = []
    u_applied_list = []
    tsat_list = []
    wsat_list = []

    for k in range(n_steps):
        t = k * dt
        st = sim.state
        th = st[0]
        th_dot = st[1]
        om = st[2]

        u_cmd, _ = pd_controller.compute_torque(theta=th, theta_dot=th_dot, Omega=om)
        _, info = sim.step(u_cmd=u_cmd, disturbance=disturbance)

        t_list.append(t)
        th_list.append(th)
        th_dot_list.append(th_dot)
        om_list.append(om)
        u_cmd_list.append(u_cmd)
        u_applied_list.append(info["u_applied"])
        tsat_list.append(info["is_torque_saturated"])
        wsat_list.append(info["is_wheel_saturated"])

    return {
        "time": np.array(t_list),
        "theta": np.array(th_list),
        "theta_dot": np.array(th_dot_list),
        "wheel_speed": np.array(om_list),
        "u_cmd": np.array(u_cmd_list),
        "u_applied": np.array(u_applied_list),
        "is_torque_saturated": np.array(tsat_list, dtype=bool),
        "is_wheel_saturated": np.array(wsat_list, dtype=bool),
        "terminated_early": False,
    }


def compute_controller_metrics(
    traj: Dict[str, np.ndarray],
    target_ss: float,
    is_periodic: bool = False,
    pulse_window: Optional[Tuple[float, float]] = None,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """Compute all 12 rigorous aerospace GNC metrics for a given trajectory."""
    t = traj["time"]
    th = traj["theta"]
    th_dot = traj["theta_dot"]
    om = traj["wheel_speed"]
    u_cmd = traj["u_cmd"]
    u_app = traj["u_applied"]
    is_tsat = traj["is_torque_saturated"]
    is_wsat = traj["is_wheel_saturated"]

    # 1. Physical Failure & Tumble Rate
    failed = traj.get("terminated_early", False) or bool(np.any(np.abs(th) > np.radians(90.0)))

    # 2. Final Attitude Error
    final_th = float(th[-1])
    final_err_deg = abs(np.degrees(final_th - target_ss))

    # 3. RMS Error
    rms_rad = float(np.sqrt(np.mean((th - target_ss) ** 2)))
    rms_deg = np.degrees(rms_rad)

    # 4. Settling Time (2% band)
    ts = None
    if not is_periodic and not failed:
        th0 = float(th[0])
        band = max(abs(th0 - target_ss) * 0.02, np.radians(0.01))
        # Find last time outside band
        outside = np.where(np.abs(th - target_ss) > band)[0]
        if len(outside) == 0:
            ts = 0.0
        elif outside[-1] < len(th) - 1:
            ts = float(t[outside[-1] + 1])
        else:
            ts = None  # Did not settle

    # 5. Decay Time (1/e)
    decay_time = None
    if not is_periodic and not failed:
        th0 = float(th[0])
        if abs(th0 - target_ss) > 1e-4:
            decay_time = compute_decay_time(t, th, steady_state_val=target_ss)

    # 6. Maximum Overshoot (%)
    mp_pct = 0.0
    if not is_periodic and not failed:
        th0 = float(th[0])
        delta0 = th0 - target_ss
        if abs(delta0) > 1e-4:
            if delta0 > 0:
                min_delta = np.min(th - target_ss)
                if min_delta < 0:
                    mp_pct = float(abs(min_delta) / delta0 * 100.0)
            else:
                max_delta = np.max(th - target_ss)
                if max_delta > 0:
                    mp_pct = float(max_delta / abs(delta0) * 100.0)

    # 7. Squared Torque Effort
    effort = float(np.sum(u_app ** 2) * dt)

    # 8. Peak Wheel Speed
    peak_om_rpm = float(np.max(np.abs(om)) * 60.0 / (2.0 * np.pi))

    # 9. Torque Saturation Duration
    tsat_duration = float(np.sum(is_tsat) * dt)

    # 10. Wheel Saturation Duration
    wsat_duration = float(np.sum(is_wsat) * dt)

    # 11. Pulse Recovery Time
    pulse_recov = None
    if pulse_window is not None and not failed:
        t_pulse_end = pulse_window[1]
        post_idx = np.where(t >= t_pulse_end)[0]
        if len(post_idx) > 0:
            th_post = th[post_idx]
            t_post = t[post_idx]
            # Settle within 0.05 deg post pulse
            out_post = np.where(np.abs(th_post - target_ss) > np.radians(0.05))[0]
            if len(out_post) == 0:
                pulse_recov = 0.0
            elif out_post[-1] < len(th_post) - 1:
                pulse_recov = float(t_post[out_post[-1] + 1] - t_pulse_end)

    # 12. Stabilization Success Rate
    stabilized = (final_err_deg < 0.20) and not failed

    return {
        "settling_time_s": ts,
        "decay_time_s": decay_time,
        "rms_attitude_error_deg": rms_deg,
        "max_overshoot_pct": mp_pct,
        "final_attitude_deg": np.degrees(final_th),
        "final_error_deg": final_err_deg,
        "control_effort": effort,
        "peak_wheel_speed_rpm": peak_om_rpm,
        "torque_saturation_duration_s": tsat_duration,
        "wheel_saturation_duration_s": wsat_duration,
        "pulse_recovery_time_s": pulse_recov,
        "is_stabilized": stabilized,
        "is_failed": failed,
    }


def run_comparative_evaluation(
    ppo_model_path: Path,
    output_dir: Path,
    include_heldout: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute head-to-head evaluation of PPO vs Classical PD on all scenarios."""
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    pd_controller = PDController(Kp=5.0, Kd=10.0)
    custom_objects = {"AntiSymmetricActorCriticPolicy": AntiSymmetricActorCriticPolicy}
    ppo_model = PPO.load(str(ppo_model_path), device="cpu", custom_objects=custom_objects)
    env = SpacecraftAttitudeEnv(include_integral=True)

    # Scenarios list
    scenarios = get_benchmark_scenarios(pd_controller)
    if include_heldout:
        scenarios += get_heldout_scenarios(pd_controller)

    records = []
    trajectories = {}

    print("\n" + "=" * 80)
    print("PHASE 4E: COMPARATIVE EVALUATION (PPO vs CLASSICAL PD)")
    print(f"Model: {ppo_model_path.name}")
    print(f"Total Scenarios: {len(scenarios)} (7 Benchmark + 5 Held-Out)")
    print("=" * 80)

    for sc in scenarios:
        sid = sc["id"]
        sname = sc["name"]
        init_st = sc["initial_state"]
        dist = sc["disturbance"]
        dur = sc["duration"]
        target_ss = sc.get("theoretical_theta_ss", 0.0)
        is_periodic = sc.get("is_periodic", False)
        pulse_win = sc.get("pulse_window", None)

        print(f"\nEvaluating: {sname}...")

        # 1. Simulate PD
        traj_pd = simulate_pd_trajectory(
            pd_controller=pd_controller,
            initial_state=init_st,
            disturbance=dist,
            duration=dur,
        )
        met_pd = compute_controller_metrics(
            traj=traj_pd,
            target_ss=target_ss,
            is_periodic=is_periodic,
            pulse_window=pulse_win,
        )

        # 2. Simulate PPO
        traj_ppo = simulate_ppo_trajectory(
            model=ppo_model,
            env=env,
            initial_state=init_st,
            disturbance=dist,
            duration=dur,
        )
        met_ppo = compute_controller_metrics(
            traj=traj_ppo,
            target_ss=target_ss,
            is_periodic=is_periodic,
            pulse_window=pulse_win,
        )

        trajectories[sid] = {"pd": traj_pd, "ppo": traj_ppo, "scenario": sc}

        # Save trajectory telemetry CSV (handling possible early termination cleanly)
        n_pd = len(traj_pd["time"])
        n_ppo = len(traj_ppo["time"])
        n_max = max(n_pd, n_ppo)

        def _pad(arr, length):
            if len(arr) < length:
                return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
            return arr[:length]

        t_common = _pad(traj_pd["time"], n_max) if n_pd >= n_ppo else _pad(traj_ppo["time"], n_max)
        df_traj = pd.DataFrame({
            "time_s": t_common,
            "pd_theta_deg": _pad(np.degrees(traj_pd["theta"]), n_max),
            "pd_theta_dot_deg_s": _pad(np.degrees(traj_pd["theta_dot"]), n_max),
            "pd_wheel_speed_rpm": _pad(traj_pd["wheel_speed"] * 60.0 / (2.0 * np.pi), n_max),
            "pd_u_applied_nm": _pad(traj_pd["u_applied"], n_max),
            "ppo_theta_deg": _pad(np.degrees(traj_ppo["theta"]), n_max),
            "ppo_theta_dot_deg_s": _pad(np.degrees(traj_ppo["theta_dot"]), n_max),
            "ppo_wheel_speed_rpm": _pad(traj_ppo["wheel_speed"] * 60.0 / (2.0 * np.pi), n_max),
            "ppo_u_applied_nm": _pad(traj_ppo["u_applied"], n_max),
        })
        df_traj.to_csv(telemetry_dir / f"{sid}_telemetry.csv", index=False)

        # Print summary line
        ts_pd_str = f"{met_pd['settling_time_s']:.2f}s" if met_pd['settling_time_s'] is not None else "N/A"
        ts_ppo_str = f"{met_ppo['settling_time_s']:.2f}s" if met_ppo['settling_time_s'] is not None else "N/A"
        print(f"  PD:  Ts={ts_pd_str:<8} | RMS={met_pd['rms_attitude_error_deg']:.3f}° | Effort={met_pd['control_effort']:.4f} | Peak RPM={met_pd['peak_wheel_speed_rpm']:.1f}")
        print(f"  PPO: Ts={ts_ppo_str:<8} | RMS={met_ppo['rms_attitude_error_deg']:.3f}° | Effort={met_ppo['control_effort']:.4f} | Peak RPM={met_ppo['peak_wheel_speed_rpm']:.1f}")

        # Store record for comparison table
        records.append({
            "scenario_id": sid,
            "scenario_name": sname,
            "is_heldout": "heldout" in sid,
            "target_ss_deg": np.degrees(target_ss),
            # PD metrics
            "pd_settling_time_s": met_pd["settling_time_s"],
            "pd_decay_time_s": met_pd["decay_time_s"],
            "pd_rms_deg": met_pd["rms_attitude_error_deg"],
            "pd_overshoot_pct": met_pd["max_overshoot_pct"],
            "pd_final_err_deg": met_pd["final_error_deg"],
            "pd_effort": met_pd["control_effort"],
            "pd_peak_wheel_rpm": met_pd["peak_wheel_speed_rpm"],
            "pd_torque_sat_s": met_pd["torque_saturation_duration_s"],
            "pd_wheel_sat_s": met_pd["wheel_saturation_duration_s"],
            "pd_pulse_recovery_s": met_pd["pulse_recovery_time_s"],
            "pd_stabilized": met_pd["is_stabilized"],
            "pd_failed": met_pd["is_failed"],
            # PPO metrics
            "ppo_settling_time_s": met_ppo["settling_time_s"],
            "ppo_decay_time_s": met_ppo["decay_time_s"],
            "ppo_rms_deg": met_ppo["rms_attitude_error_deg"],
            "ppo_overshoot_pct": met_ppo["max_overshoot_pct"],
            "ppo_final_err_deg": met_ppo["final_error_deg"],
            "ppo_effort": met_ppo["control_effort"],
            "ppo_peak_wheel_rpm": met_ppo["peak_wheel_speed_rpm"],
            "ppo_torque_sat_s": met_ppo["torque_saturation_duration_s"],
            "ppo_wheel_sat_s": met_ppo["wheel_saturation_duration_s"],
            "ppo_pulse_recovery_s": met_ppo["pulse_recovery_time_s"],
            "ppo_stabilized": met_ppo["is_stabilized"],
            "ppo_failed": met_ppo["is_failed"],
        })

    env.close()

    df_comp = pd.DataFrame(records)
    csv_path = output_dir / "comparative_metrics_ppo_vs_pd.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[EVALUATION COMPLETE] Comparative table saved to: {csv_path}")

    return df_comp, trajectories


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4E Comparative Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model .zip")
    parser.add_argument("--out", type=str, default="outputs/ppo_training/phase_4e/evaluations", help="Output directory")
    args = parser.parse_args()

    run_comparative_evaluation(Path(args.model), Path(args.out))
