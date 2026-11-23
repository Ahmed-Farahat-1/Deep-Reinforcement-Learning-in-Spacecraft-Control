"""
Phase 4G Master Comparative Evaluation Engine.
Evaluates 4 independent controllers:
1. Classical PD
2. Pure PPO (Phase 4E Champion)
3. Pure TD3 (Phase 4G Champion)
4. Pure SAC (Phase 4G Champion)
across all 12 scenarios (7 canonical + 5 held-out H1 to H5),
computing all 12 AIAA GNC performance metrics.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO, TD3, SAC
from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
from src.agents.td3.td3_policy import AntiSymmetricTD3Policy
from src.agents.sac.sac_policy import AntiSymmetricSACPolicy
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.controllers.pd_controller import PDController
from src.simulation.spacecraft import Spacecraft1D
from src.agents.select_best_checkpoint import simulate_ppo_trajectory
from src.agents.evaluate_phase_4e import (
    simulate_pd_trajectory,
    compute_controller_metrics,
)
from src.agents.evaluate_ppo import get_benchmark_scenarios, get_heldout_scenarios


def simulate_agent_trajectory(
    model: Any,
    env: SpacecraftAttitudeEnv,
    initial_state: List[float],
    disturbance: Any,
    duration: float = 30.0,
    dt: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Execute deterministic trajectory for any SB3 agent (PPO, TD3, SAC)."""
    return simulate_ppo_trajectory(
        model=model,
        env=env,
        initial_state=initial_state,
        disturbance=disturbance,
        duration=duration,
        dt=dt,
    )


def run_four_controller_evaluation(
    ppo_model_path: Path,
    td3_model_path: Path,
    sac_model_path: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Run deterministic closed-loop evaluation comparing PD, PPO, TD3, and SAC
    across 12 scenarios (7 Benchmark + 5 Held-Out).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Controllers
    physics_cfg = PROJECT_ROOT / "config" / "physics_config.yaml"
    pd_controller = PDController.from_config(physics_cfg)
    env = SpacecraftAttitudeEnv(include_integral=True)

    # Load RL Models
    custom_ppo = {"AntiSymmetricActorCriticPolicy": AntiSymmetricActorCriticPolicy}
    ppo_model = PPO.load(str(ppo_model_path), device="cpu", custom_objects=custom_ppo)

    custom_td3 = {"AntiSymmetricTD3Policy": AntiSymmetricTD3Policy}
    td3_model = TD3.load(str(td3_model_path), device="cpu", custom_objects=custom_td3)

    custom_sac = {"AntiSymmetricSACPolicy": AntiSymmetricSACPolicy}
    sac_model = SAC.load(str(sac_model_path), device="cpu", custom_objects=custom_sac)

    scenarios = get_benchmark_scenarios(pd_controller) + get_heldout_scenarios(pd_controller)
    trajectories = {}
    records = []

    print("\n" + "=" * 90)
    print("PHASE 4G: COMPARATIVE BENCHMARK (PD vs PPO vs TD3 vs SAC)")
    print(f"Total Scenarios: {len(scenarios)} (7 Benchmark + 5 Held-Out)")
    print("=" * 90)

    for sc in scenarios:
        sid = sc["id"]
        sname = sc["name"]
        init_st = sc["initial_state"]
        dist = sc["disturbance"]
        dur = sc["duration"]
        target_ss = sc.get("theoretical_theta_ss", 0.0)
        is_periodic = sc.get("is_periodic", False)
        pulse_win = sc.get("pulse_window", None)
        is_heldout = sc.get("is_heldout", False)

        print(f"\nEvaluating: {sname}...")

        # 1. PD
        traj_pd = simulate_pd_trajectory(pd_controller, init_st, dist, dur)
        met_pd = compute_controller_metrics(traj_pd, target_ss, is_periodic, pulse_win)

        # 2. PPO
        traj_ppo = simulate_agent_trajectory(ppo_model, env, init_st, dist, dur)
        met_ppo = compute_controller_metrics(traj_ppo, target_ss, is_periodic, pulse_win)

        # 3. TD3
        traj_td3 = simulate_agent_trajectory(td3_model, env, init_st, dist, dur)
        met_td3 = compute_controller_metrics(traj_td3, target_ss, is_periodic, pulse_win)

        # 4. SAC
        traj_sac = simulate_agent_trajectory(sac_model, env, init_st, dist, dur)
        met_sac = compute_controller_metrics(traj_sac, target_ss, is_periodic, pulse_win)

        trajectories[sid] = {
            "pd": traj_pd,
            "ppo": traj_ppo,
            "td3": traj_td3,
            "sac": traj_sac,
            "scenario": sc,
        }

        print(f"  PD:  Ts={met_pd['settling_time_s']} | RMS={met_pd['rms_attitude_error_deg']:.3f}° | Effort={met_pd['control_effort']:.4f} | Final={met_pd['final_error_deg']:.4f}°")
        print(f"  PPO: Ts={met_ppo['settling_time_s']} | RMS={met_ppo['rms_attitude_error_deg']:.3f}° | Effort={met_ppo['control_effort']:.4f} | Final={met_ppo['final_error_deg']:.4f}°")
        print(f"  TD3: Ts={met_td3['settling_time_s']} | RMS={met_td3['rms_attitude_error_deg']:.3f}° | Effort={met_td3['control_effort']:.4f} | Final={met_td3['final_error_deg']:.4f}°")
        print(f"  SAC: Ts={met_sac['settling_time_s']} | RMS={met_sac['rms_attitude_error_deg']:.3f}° | Effort={met_sac['control_effort']:.4f} | Final={met_sac['final_error_deg']:.4f}°")

        # Save trajectory CSV
        n_max = max(len(traj_pd["time"]), len(traj_ppo["time"]), len(traj_td3["time"]), len(traj_sac["time"]))
        def _pad(arr, length):
            if len(arr) < length:
                return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
            return arr[:length]

        t_common = _pad(traj_pd["time"], n_max)
        df_traj = pd.DataFrame({
            "time_s": t_common,
            "pd_theta_deg": _pad(np.degrees(traj_pd["theta"]), n_max),
            "pd_wheel_rpm": _pad(traj_pd["wheel_speed"] * 60.0 / (2.0 * np.pi), n_max),
            "pd_u_applied": _pad(traj_pd["u_applied"], n_max),
            "ppo_theta_deg": _pad(np.degrees(traj_ppo["theta"]), n_max),
            "ppo_wheel_rpm": _pad(traj_ppo["wheel_speed"] * 60.0 / (2.0 * np.pi), n_max),
            "ppo_u_applied": _pad(traj_ppo["u_applied"], n_max),
            "td3_theta_deg": _pad(np.degrees(traj_td3["theta"]), n_max),
            "td3_wheel_rpm": _pad(traj_td3["wheel_speed"] * 60.0 / (2.0 * np.pi), n_max),
            "td3_u_applied": _pad(traj_td3["u_applied"], n_max),
            "sac_theta_deg": _pad(np.degrees(traj_sac["theta"]), n_max),
            "sac_wheel_rpm": _pad(traj_sac["wheel_speed"] * 60.0 / (2.0 * np.pi), n_max),
            "sac_u_applied": _pad(traj_sac["u_applied"], n_max),
        })
        df_traj.to_csv(output_dir / f"trajectory_{sid}.csv", index=False)

        # Record metrics row
        records.append({
            "scenario_id": sid,
            "scenario_name": sname,
            "is_heldout": is_heldout,
            "target_ss_deg": np.degrees(target_ss),
            # PD
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
            # PPO
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
            # TD3
            "td3_settling_time_s": met_td3["settling_time_s"],
            "td3_decay_time_s": met_td3["decay_time_s"],
            "td3_rms_deg": met_td3["rms_attitude_error_deg"],
            "td3_overshoot_pct": met_td3["max_overshoot_pct"],
            "td3_final_err_deg": met_td3["final_error_deg"],
            "td3_effort": met_td3["control_effort"],
            "td3_peak_wheel_rpm": met_td3["peak_wheel_speed_rpm"],
            "td3_torque_sat_s": met_td3["torque_saturation_duration_s"],
            "td3_wheel_sat_s": met_td3["wheel_saturation_duration_s"],
            "td3_pulse_recovery_s": met_td3["pulse_recovery_time_s"],
            "td3_stabilized": met_td3["is_stabilized"],
            "td3_failed": met_td3["is_failed"],
            # SAC
            "sac_settling_time_s": met_sac["settling_time_s"],
            "sac_decay_time_s": met_sac["decay_time_s"],
            "sac_rms_deg": met_sac["rms_attitude_error_deg"],
            "sac_overshoot_pct": met_sac["max_overshoot_pct"],
            "sac_final_err_deg": met_sac["final_error_deg"],
            "sac_effort": met_sac["control_effort"],
            "sac_peak_wheel_rpm": met_sac["peak_wheel_speed_rpm"],
            "sac_torque_sat_s": met_sac["torque_saturation_duration_s"],
            "sac_wheel_sat_s": met_sac["wheel_saturation_duration_s"],
            "sac_pulse_recovery_s": met_sac["pulse_recovery_time_s"],
            "sac_stabilized": met_sac["is_stabilized"],
            "sac_failed": met_sac["is_failed"],
        })

    df_comp = pd.DataFrame(records)
    comp_csv = output_dir / "comparative_metrics_all_scenarios.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"\n[EVALUATION COMPLETE] Comparative table saved to: {comp_csv}")

    env.close()
    return df_comp, trajectories
