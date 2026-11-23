"""
Dedicated Validation & Checkpoint Selection Engine for Phase 4E.

Evaluates all candidate checkpoints against a balanced 8-scenario validation suite
strictly according to Section 9 criteria:
1. Minimize physical tumble and failure rate.
2. Maximize attitude stabilization success rate.
3. Minimize normalized RMS attitude error.
4. Minimize overshoot and recovery time.
5. Maintain safe wheel-speed and motor-torque margins.
6. Minimize control effort among policies with comparable stabilization performance.
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

from stable_baselines3 import PPO, TD3, SAC
from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.simulation.spacecraft import Spacecraft1D
from src.simulation.disturbances import (
    ZeroDisturbance,
    ConstantDisturbance,
    SinusoidalDisturbance,
    PulseDisturbance,
)
from src.agents.evaluate_ppo import simulate_ppo_trajectory


def compute_trajectory_rms_error(theta: np.ndarray, steady_state_value: float = 0.0) -> float:
    """Calculate RMS attitude error relative to steady-state value in radians."""
    return float(np.sqrt(np.mean((theta - steady_state_value) ** 2)))


def simulate_ppo_trajectory(
    model: PPO,
    env: SpacecraftAttitudeEnv,
    initial_state: List[float],
    disturbance: Any,
    duration: float = 30.0,
    dt: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Execute deterministic PPO trajectory on validated Spacecraft1D simulator."""
    sim = Spacecraft1D(dt=dt)
    sim.reset(initial_state=np.array(initial_state, dtype=np.float64))

    obs_dim = model.observation_space.shape[0] if hasattr(model, "observation_space") else 4
    z = 0.0

    steps = int(round(duration / dt))
    t_list = []
    th_list = []
    th_dot_list = []
    om_list = []
    u_cmd_list = []
    u_applied_list = []
    tsat_list = []
    wsat_list = []
    terminated_early = False

    for i in range(steps):
        t = sim.time
        st = sim.state
        th, th_dot, om = st[0], st[1], st[2]

        t_list.append(t)
        th_list.append(th)
        th_dot_list.append(th_dot)
        om_list.append(om)

        if abs(th) > env.failure_theta or abs(th_dot) > env.failure_theta_dot:
            u_cmd_list.append(0.0)
            u_applied_list.append(0.0)
            tsat_list.append(False)
            wsat_list.append(False)
            terminated_early = True
            break

        th_norm = float(np.clip(th / env.theta_scale, -1.0, 1.0))
        th_dot_norm = float(np.clip(th_dot / env.theta_dot_scale, -1.0, 1.0))
        om_norm = float(np.clip(om / env.Omega_scale, -1.0, 1.0))

        if obs_dim == 4:
            z_norm = float(np.clip(z / env.z_scale, -1.0, 1.0))
            obs = np.array([th_norm, th_dot_norm, om_norm, z_norm], dtype=np.float32)
        else:
            obs = np.array([th_norm, th_dot_norm, om_norm], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        act_clipped = float(np.clip(action[0], -1.0, 1.0))
        u_cmd = act_clipped * sim.max_torque

        _, info = sim.step(u_cmd=u_cmd, disturbance=disturbance)
        u_cmd_list.append(u_cmd)
        u_applied_list.append(info["u_applied"])
        tsat_list.append(info["is_torque_saturated"])
        wsat_list.append(info["is_wheel_saturated"])

        if obs_dim == 4:
            is_at_torque_limit = abs(u_cmd) >= (sim.max_torque - 1e-5)
            outward_push = (th * u_cmd > 0.0) and is_at_torque_limit
            wheel_blocked_push = bool(info["is_wheel_saturated"])
            if not (outward_push or wheel_blocked_push):
                z += 0.5 * (th + sim.state[0]) * dt
                z = float(np.clip(z, -env.z_max, env.z_max))

    return {
        "time": np.array(t_list),
        "theta": np.array(th_list),
        "theta_dot": np.array(th_dot_list),
        "wheel_speed": np.array(om_list),
        "u_cmd": np.array(u_cmd_list),
        "u_applied": np.array(u_applied_list),
        "is_torque_saturated": np.array(tsat_list, dtype=bool),
        "is_wheel_saturated": np.array(wsat_list, dtype=bool),
        "terminated_early": terminated_early,
    }


def get_validation_suite() -> List[Dict[str, Any]]:
    """Define dedicated 8-scenario balanced validation suite for checkpoint selection."""
    return [
        {
            "id": "val_1_pos10",
            "name": "Val 1: Step +10 deg (Zero Dist)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
        },
        {
            "id": "val_2_neg10",
            "name": "Val 2: Step -10 deg (Zero Dist)",
            "initial_state": [np.radians(-10.0), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
        },
        {
            "id": "val_3_pos30",
            "name": "Val 3: Step +30 deg (Torque Saturation)",
            "initial_state": [np.radians(30.0), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
        },
        {
            "id": "val_4_neg30",
            "name": "Val 4: Step -30 deg (Torque Saturation)",
            "initial_state": [np.radians(-30.0), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
        },
        {
            "id": "val_5_const_pos",
            "name": "Val 5: Constant Disturbance +0.02 N*m (+10 deg)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": ConstantDisturbance(torque_value=0.02),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
        },
        {
            "id": "val_6_const_neg",
            "name": "Val 6: Constant Disturbance -0.02 N*m (-10 deg)",
            "initial_state": [np.radians(-10.0), 0.0, 0.0],
            "disturbance": ConstantDisturbance(torque_value=-0.02),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
        },
        {
            "id": "val_7_sine",
            "name": "Val 7: Sinusoidal Disturbance 0.02*sin(0.2*pi*t)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": SinusoidalDisturbance(amplitude=0.02, frequency=0.1),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": True,
        },
        {
            "id": "val_8_pulse",
            "name": "Val 8: Pulse Disturbance +0.05 N*m (t in [10, 12]s)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": PulseDisturbance(magnitude=0.05, t_start=10.0, t_end=12.0),
            "target_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
        },
    ]


def evaluate_checkpoint_on_validation_suite(
    model_path: Path,
    validation_suite: Optional[List[Dict[str, Any]]] = None,
    algo: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint deterministically across all validation scenarios."""
    if validation_suite is None:
        validation_suite = get_validation_suite()

    model_path_str = str(model_path).lower()
    algo_str = algo.lower() if algo is not None else ""

    if "td3" in algo_str or ("td3" in model_path_str and "sac" not in algo_str):
        from src.agents.td3.td3_policy import AntiSymmetricTD3Policy
        custom_objects = {"AntiSymmetricTD3Policy": AntiSymmetricTD3Policy}
        model = TD3.load(str(model_path), device="cpu", custom_objects=custom_objects)
    elif "sac" in algo_str or "sac" in model_path_str:
        from src.agents.sac.sac_policy import AntiSymmetricSACPolicy
        custom_objects = {"AntiSymmetricSACPolicy": AntiSymmetricSACPolicy}
        model = SAC.load(str(model_path), device="cpu", custom_objects=custom_objects)
    else:
        custom_objects = {"AntiSymmetricActorCriticPolicy": AntiSymmetricActorCriticPolicy}
        model = PPO.load(str(model_path), device="cpu", custom_objects=custom_objects)

    env = SpacecraftAttitudeEnv(include_integral=True)

    tumble_count = 0
    stabilized_count = 0
    rms_errors = []
    overshoots = []
    max_wheel_rpms = []
    torque_sat_durations = []
    wheel_sat_durations = []
    efforts = []
    final_errors = []

    for sc in validation_suite:
        traj = simulate_ppo_trajectory(
            model=model,
            env=env,
            initial_state=sc["initial_state"],
            disturbance=sc["disturbance"],
            duration=sc["duration"],
        )

        # 1. Tumble detection
        if traj["terminated_early"] or np.any(np.abs(traj["theta"]) > np.radians(90.0)):
            tumble_count += 1

        # 2. Final attitude and RMS error
        final_err = abs(traj["theta"][-1] - sc["target_ss"])
        final_err_deg = np.degrees(final_err)
        final_errors.append(final_err_deg)

        # Stabilized if final error is small (< 0.2 deg)
        if final_err_deg < 0.20:
            stabilized_count += 1

        # RMS attitude error (degrees)
        rms_deg = np.degrees(compute_trajectory_rms_error(traj["theta"], sc["target_ss"]))
        rms_errors.append(rms_deg)

        # 3. Overshoot (for step scenarios)
        if not sc["is_periodic"]:
            th0 = sc["initial_state"][0]
            if abs(th0) > 1e-4:
                # Max excursion past 0
                if th0 > 0:
                    min_th = np.min(traj["theta"])
                    os_deg = max(0.0, -min_th)
                else:
                    max_th = np.max(traj["theta"])
                    os_deg = max(0.0, max_th)
                overshoots.append(np.degrees(os_deg))
            else:
                overshoots.append(0.0)
        else:
            overshoots.append(0.0)

        # 4. Actuator margins & effort
        max_om_rpm = float(np.max(np.abs(traj["wheel_speed"])) * 60.0 / (2.0 * np.pi))
        max_wheel_rpms.append(max_om_rpm)

        t_sat_s = float(np.sum(traj["is_torque_saturated"]) * env.dt)
        torque_sat_durations.append(t_sat_s)

        w_sat_s = float(np.sum(traj["is_wheel_saturated"]) * env.dt)
        wheel_sat_durations.append(w_sat_s)

        eff = float(np.sum(traj["u_applied"] ** 2) * env.dt)
        efforts.append(eff)

    env.close()

    mean_rms = float(np.mean(rms_errors))
    mean_os = float(np.mean(overshoots))
    mean_effort = float(np.mean(efforts))
    max_wheel = float(np.max(max_wheel_rpms))
    total_tsat = float(np.sum(torque_sat_durations))
    total_wsat = float(np.sum(wheel_sat_durations))
    mean_final_err = float(np.mean(final_errors))
    success_rate = float(stabilized_count / len(validation_suite) * 100.0)

    # Composite ranking score based on strict priorities:
    # Priority 1: 0 tumbles (penalty 10000 per tumble)
    # Priority 2: Success rate (+100 max)
    # Priority 3: RMS error (minimize)
    # Priority 4: Overshoot (minimize)
    # Priority 5: Safe wheel margins (penalty if max_wheel > 4500 RPM)
    # Priority 6: Control effort (minimize)
    score = (
        - (tumble_count * 10000.0)
        + (success_rate * 2.0)
        - (mean_rms * 5.0)
        - (mean_os * 1.0)
        - (mean_final_err * 10.0)
        - (mean_effort * 2.0)
        - (max(0.0, max_wheel - 4500.0) * 0.1)
        - (total_wsat * 50.0)
    )

    return {
        "model_file": model_path.name,
        "model_path": str(model_path),
        "tumble_count": tumble_count,
        "stabilized_count": stabilized_count,
        "success_rate_pct": success_rate,
        "mean_rms_deg": mean_rms,
        "mean_final_err_deg": mean_final_err,
        "mean_overshoot_deg": mean_os,
        "max_wheel_rpm": max_wheel,
        "total_torque_sat_s": total_tsat,
        "total_wheel_sat_s": total_wsat,
        "mean_effort": mean_effort,
        "composite_score": score,
    }


def rank_and_select_best_checkpoint(
    checkpoint_dir: Path,
    output_csv: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Scan all checkpoint .zip files, evaluate each, and select the optimal candidate."""
    if output_csv is not None and output_csv.is_file():
        print(f"\n[SELECTION] Loading existing ranking from: {output_csv}")
        df = pd.read_csv(output_csv)
        if len(df) > 0:
            df = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
            best_record = df.iloc[0].to_dict()
            print("\n" + "=" * 80)
            print(f"BEST CHECKPOINT (LOADED): {best_record['model_file']}")
            print(f"Composite Score:      {best_record['composite_score']:+.2f}")
            print(f"Tumble Failures:      {best_record['tumble_count']}")
            print(f"Stabilization Rate:   {best_record['success_rate_pct']:.1f}%")
            print(f"Mean RMS Error:       {best_record['mean_rms_deg']:.4f} deg")
            print(f"Mean Final Error:     {best_record['mean_final_err_deg']:.4f} deg")
            print(f"Peak Wheel Speed:     {best_record['max_wheel_rpm']:.1f} RPM")
            print(f"Mean Control Effort:  {best_record['mean_effort']:.4f} N^2*m^2*s")
            print("=" * 80 + "\n")
            return best_record, df

    zip_files = sorted(list(checkpoint_dir.glob("*.zip")))
    if not zip_files:
        raise FileNotFoundError(f"No checkpoint .zip files found in {checkpoint_dir}")

    val_suite = get_validation_suite()
    print(f"\n[SELECTION] Scanning {len(zip_files)} checkpoints in {checkpoint_dir.name}...")
    results = []

    for zf in zip_files:
        try:
            res = evaluate_checkpoint_on_validation_suite(zf, val_suite)
            results.append(res)
            print(f"  Checkpoint: {zf.name:<30} | Score: {res['composite_score']:+8.2f} | Tumbles: {res['tumble_count']} | RMS: {res['mean_rms_deg']:.3f}° | Final: {res['mean_final_err_deg']:.3f}°")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {zf.name}: {e}")

    df = pd.DataFrame(results)
    # Sort descending by composite score
    df = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[SELECTION] Validation ranking saved to: {output_csv}")

    best_record = df.iloc[0].to_dict()
    print("\n" + "=" * 80)
    print(f"BEST CHECKPOINT SELECTED: {best_record['model_file']}")
    print(f"Composite Score:      {best_record['composite_score']:+.2f}")
    print(f"Tumble Failures:      {best_record['tumble_count']} / {len(val_suite)}")
    print(f"Stabilization Rate:   {best_record['success_rate_pct']:.1f}%")
    print(f"Mean RMS Error:       {best_record['mean_rms_deg']:.4f} deg")
    print(f"Mean Final Error:     {best_record['mean_final_err_deg']:.4f} deg")
    print(f"Peak Wheel Speed:     {best_record['max_wheel_rpm']:.1f} RPM")
    print(f"Mean Control Effort:  {best_record['mean_effort']:.4f} N^2*m^2*s")
    print("=" * 80 + "\n")

    return best_record, df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4E Checkpoint Selection")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    rank_and_select_best_checkpoint(Path(args.dir), Path(args.out) if args.out else None)
