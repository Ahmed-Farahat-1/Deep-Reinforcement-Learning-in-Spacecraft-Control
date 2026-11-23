"""
Deterministic Evaluation and PD Comparative Benchmark for PPO Agent.

Phase 4A: Validates the trained pilot PPO model against the classical PD baseline
across 7 canonical and stress scenarios using identical physical models and AIAA GNC metrics.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from src.simulation.spacecraft import Spacecraft1D
from src.simulation.disturbances import (
    BaseDisturbance,
    ZeroDisturbance,
    ConstantDisturbance,
    SinusoidalDisturbance,
    PulseDisturbance,
)
from src.controllers.pd_controller import PDController
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.evaluation.metrics import calculate_comprehensive_metrics, compute_squared_torque_integral


def load_evaluation_model(model_path: Optional[Path] = None) -> PPO:
    """Load trained PPO policy for evaluation."""
    if model_path is None:
        p4b_best = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4b" / "best_model" / "best_model.zip"
        p4b_final = PROJECT_ROOT / "models" / "ppo" / "phase_4b" / "ppo_final_500k.zip"
        best_path = PROJECT_ROOT / "outputs" / "ppo_training" / "best_model" / "best_model.zip"
        pilot_path = PROJECT_ROOT / "models" / "ppo" / "ppo_pilot_100k.zip"
        final_path = PROJECT_ROOT / "outputs" / "ppo_training" / "final_pilot_model.zip"

        if p4b_best.is_file():
            model_path = p4b_best
            print(f"[EVAL] Loading Phase 4B best evaluation model from: {p4b_best}")
        elif p4b_final.is_file():
            model_path = p4b_final
            print(f"[EVAL] Loading Phase 4B final model from: {p4b_final}")
        elif best_path.is_file():
            model_path = best_path
            print(f"[EVAL] Loading Phase 4A best evaluation model from: {best_path}")
        elif pilot_path.is_file():
            model_path = pilot_path
            print(f"[EVAL] Loading Phase 4A pilot model from: {pilot_path}")
        elif final_path.is_file():
            model_path = final_path
            print(f"[EVAL] Loading Phase 4A final model from: {final_path}")
        else:
            raise FileNotFoundError("No trained PPO model found in models/ppo/ or outputs/ppo_training/")

    from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
    custom_objects = {
        "AntiSymmetricActorCriticPolicy": AntiSymmetricActorCriticPolicy,
    }
    return PPO.load(str(model_path), device="cpu", custom_objects=custom_objects)


def get_benchmark_scenarios(pd_controller: PDController) -> List[Dict[str, Any]]:
    """Define canonical benchmark scenarios identical to Phase 2B/2C plus near-saturation stress."""
    return [
        {
            "id": "scenario_1_pos10deg",
            "name": "Scenario 1: Initial Error +10 deg (Zero Disturbance)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
        {
            "id": "scenario_2_neg10deg",
            "name": "Scenario 2: Initial Error -10 deg (Zero Disturbance)",
            "initial_state": [np.radians(-10.0), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
        {
            "id": "scenario_3_zero_dist_large",
            "name": "Scenario 3: Initial Error +30 deg (Torque Saturation)",
            "initial_state": [np.radians(30.0), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
        {
            "id": "scenario_4_constant_dist",
            "name": "Scenario 4: Constant Disturbance (Md = 0.02 N*m)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": ConstantDisturbance(torque_value=0.02),
            "theoretical_theta_ss": 0.02 / pd_controller.Kp,  # 0.229183 deg
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
        {
            "id": "scenario_5_sinusoidal_dist",
            "name": "Scenario 5: Sinusoidal Disturbance (A = 0.02 N*m, f = 0.1 Hz)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": SinusoidalDisturbance(amplitude=0.02, frequency=0.1),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": True,
            "pulse_window": None,
        },
        {
            "id": "scenario_6_pulse_dist",
            "name": "Scenario 6: Pulse Disturbance (Md = 0.05 N*m, t in [10s, 12s])",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": PulseDisturbance(magnitude=0.05, t_start=10.0, t_end=12.0),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": (10.0, 12.0),
        },
        {
            "id": "scenario_7_near_saturation",
            "name": "Scenario 7: Near-Saturation Stress Test (Omega0 = 4200 RPM)",
            "initial_state": [np.radians(10.0), 0.0, float(4200.0 * 2.0 * np.pi / 60.0)],
            "disturbance": ZeroDisturbance(),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
    ]


def get_heldout_scenarios(pd_controller: PDController) -> List[Dict[str, Any]]:
    """Define 5 canonical held-out test scenarios for generalization audit (Phase 4D)."""
    return [
        {
            "id": "heldout_1_pos17_5deg",
            "name": "Held-Out 1: Unseen Step +17.5 deg (Zero Disturbance)",
            "initial_state": [np.radians(17.5), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
        {
            "id": "heldout_2_neg22_5deg",
            "name": "Held-Out 2: Unseen Step -22.5 deg (Zero Disturbance)",
            "initial_state": [np.radians(-22.5), 0.0, 0.0],
            "disturbance": ZeroDisturbance(),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
        {
            "id": "heldout_3_mixed_state",
            "name": "Held-Out 3: Mixed State (+15 deg, -1.5 deg/s, +500 RPM)",
            "initial_state": [np.radians(15.0), np.radians(-1.5), float(500.0 * 2.0 * np.pi / 60.0)],
            "disturbance": ZeroDisturbance(),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
        {
            "id": "heldout_4_high_freq_sine",
            "name": "Held-Out 4: High-Frequency Sine (A = 0.03 N*m, f = 0.25 Hz)",
            "initial_state": [np.radians(10.0), 0.0, 0.0],
            "disturbance": SinusoidalDisturbance(amplitude=0.03, frequency=0.25),
            "theoretical_theta_ss": 0.0,
            "duration": 30.0,
            "is_periodic": True,
            "pulse_window": None,
        },
        {
            "id": "heldout_5_strong_constant",
            "name": "Held-Out 5: Strong Disturbance (Md = -0.035 N*m, -10 deg)",
            "initial_state": [np.radians(-10.0), 0.0, 0.0],
            "disturbance": ConstantDisturbance(torque_value=-0.035),
            "theoretical_theta_ss": -0.035 / pd_controller.Kp,
            "duration": 30.0,
            "is_periodic": False,
            "pulse_window": None,
        },
    ]


def simulate_ppo_trajectory(
    model: PPO,
    sim: Spacecraft1D,
    env: SpacecraftAttitudeEnv,
    disturbance: BaseDisturbance,
    initial_state: List[float],
    duration: float = 30.0,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """Execute deterministic PPO trajectory on validated Spacecraft1D simulator with termination monitoring."""
    sim.dt = dt
    sim.reset(initial_state=initial_state)

    # Check whether model expects 3D or 4D observation
    obs_dim = model.observation_space.shape[0] if hasattr(model, "observation_space") else (4 if env.include_integral else 3)
    z = 0.0

    steps = int(round(duration / dt))
    t_list = []
    th_list = []
    th_dot_list = []
    om_list = []
    u_list = []
    dist_list = []
    terminated_early = False

    for i in range(steps):
        t = sim.time
        st = sim.state
        th, th_dot, om = st[0], st[1], st[2]

        t_list.append(t)
        th_list.append(th)
        th_dot_list.append(th_dot)
        om_list.append(om)
        dist_list.append(disturbance.torque(t))

        # Check termination condition (physical tumble or rate runaway)
        # Consistent with Gymnasium environment definition:
        # |theta| > failure_theta (90 deg) or |theta_dot| > failure_theta_dot (120 deg/s)
        is_tumbled = abs(th) > env.failure_theta
        is_rate_runaway = abs(th_dot) > env.failure_theta_dot
        if is_tumbled or is_rate_runaway:
            u_list.append(0.0)
            terminated_early = True
            break

        # Construct normalized observation vector
        th_norm = np.clip(th / env.theta_scale, -1.0, 1.0)
        th_dot_norm = np.clip(th_dot / env.theta_dot_scale, -1.0, 1.0)
        om_norm = np.clip(om / env.Omega_scale, -1.0, 1.0)

        if obs_dim == 4:
            z_norm = np.clip(z / env.z_scale, -1.0, 1.0)
            obs = np.array([th_norm, th_dot_norm, om_norm, z_norm], dtype=np.float32)
        else:
            obs = np.array([th_norm, th_dot_norm, om_norm], dtype=np.float32)

        # Deterministic action prediction
        action, _ = model.predict(obs, deterministic=True)
        act_clipped = float(np.clip(action[0], -1.0, 1.0))
        u_cmd = act_clipped * sim.max_torque

        # Step physics simulator
        _, info = sim.step(u_cmd=u_cmd, disturbance=disturbance)
        u_list.append(info["u_applied"])

        # Update integral state with anti-windup if in 4D mode
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
        "Omega": np.array(om_list),
        "torque": np.array(u_list),
        "disturbance": np.array(dist_list),
        "terminated_early": terminated_early,
    }


def simulate_pd_trajectory(
    controller: PDController,
    sim: Spacecraft1D,
    disturbance: BaseDisturbance,
    initial_state: List[float],
    duration: float = 30.0,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """Execute classical PD trajectory on validated Spacecraft1D simulator with termination monitoring."""
    sim.dt = dt
    sim.reset(initial_state=initial_state)

    steps = int(round(duration / dt))
    t_list = []
    th_list = []
    th_dot_list = []
    om_list = []
    u_list = []
    dist_list = []
    terminated_early = False

    for i in range(steps):
        t = sim.time
        st = sim.state
        th, th_dot, om = st[0], st[1], st[2]

        t_list.append(t)
        th_list.append(th)
        th_dot_list.append(th_dot)
        om_list.append(om)
        dist_list.append(disturbance.torque(t))

        if abs(th) > np.radians(90.0) or abs(th_dot) > np.radians(120.0):
            u_list.append(0.0)
            terminated_early = True
            break

        u_cmd, _ = controller.compute_torque(theta=th, theta_dot=th_dot, Omega=om)
        _, info = sim.step(u_cmd=u_cmd, disturbance=disturbance)
        u_list.append(info["u_applied"])

    return {
        "time": np.array(t_list),
        "theta": np.array(th_list),
        "theta_dot": np.array(th_dot_list),
        "Omega": np.array(om_list),
        "torque": np.array(u_list),
        "disturbance": np.array(dist_list),
        "terminated_early": terminated_early,
    }


def plot_scenario_comparison(
    ppo_res: Dict[str, np.ndarray],
    pd_res: Dict[str, np.ndarray],
    sc_info: Dict[str, Any],
    output_path: Path,
):
    """Generate high-resolution 3-panel comparative response figure (PPO vs. PD)."""
    t_ppo = ppo_res["time"]
    t_pd = pd_res["time"]
    th_ppo_deg = np.degrees(ppo_res["theta"])
    th_pd_deg = np.degrees(pd_res["theta"])

    om_ppo_rpm = ppo_res["Omega"] * 60.0 / (2.0 * np.pi)
    om_pd_rpm = pd_res["Omega"] * 60.0 / (2.0 * np.pi)

    u_ppo = ppo_res["torque"]
    u_pd = pd_res["torque"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{sc_info['name']}\nComparative Performance: PPO vs. Classical PD", fontsize=13, fontweight="bold")

    # 1. Attitude Angle theta
    ax = axes[0]
    ax.plot(t_pd, th_pd_deg, label="Classical PD", color="#1f77b4", linewidth=2.0, linestyle="--")
    ax.plot(t_ppo, th_ppo_deg, label="PPO Agent", color="#d62728", linewidth=2.0)
    ax.axhline(0.0, color="black", linestyle=":", alpha=0.6, label="Target (0°)")
    if sc_info.get("theoretical_theta_ss", 0.0) != 0.0:
        ss_deg = np.degrees(sc_info["theoretical_theta_ss"])
        ax.axhline(ss_deg, color="#2ca02c", linestyle="-.", alpha=0.7, label=f"Theory Offset ({ss_deg:.3f}°)")
    ax.set_ylabel("Attitude θ [deg]", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    # 2. Reaction Wheel Speed Omega
    ax = axes[1]
    ax.plot(t_pd, om_pd_rpm, label="Classical PD", color="#1f77b4", linewidth=2.0, linestyle="--")
    ax.plot(t_ppo, om_ppo_rpm, label="PPO Agent", color="#d62728", linewidth=2.0)
    ax.axhline(5000.0, color="gray", linestyle=":", alpha=0.5, label="±5000 RPM Limit")
    ax.axhline(-5000.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Wheel Speed Ω [RPM]", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)

    # 3. Control Torque u
    ax = axes[2]
    ax.plot(t_pd, u_pd, label="Classical PD", color="#1f77b4", linewidth=1.5, linestyle="--", alpha=0.85)
    ax.plot(t_ppo, u_ppo, label="PPO Agent", color="#d62728", linewidth=1.5, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="±1.0 N·m Limit")
    ax.axhline(-1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time [s]", fontweight="bold")
    ax.set_ylabel("Motor Torque u [N·m]", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def plot_learning_curve(eval_npz_path: Path, output_path: Path):
    """Plot PPO training evaluation reward curve."""
    if not eval_npz_path.is_file():
        return

    npz = np.load(str(eval_npz_path))
    timesteps = npz["timesteps"]
    results = npz["results"]  # shape (n_evals, n_episodes)

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, mean_rewards, marker="o", color="#2ca02c", linewidth=2.0, label="PPO Mean Eval Reward")
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color="#2ca02c", alpha=0.2)
    ax.set_title("PPO Pilot Training Convergence Curve (Phase 4A)", fontweight="bold")
    ax.set_xlabel("Timesteps", fontweight="bold")
    ax.set_ylabel("Mean Evaluation Episode Reward", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def run_full_evaluation(
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute full deterministic evaluation across all benchmark scenarios."""
    print("\n" + "=" * 85)
    print("PPO VS. CLASSICAL PD DETERMINISTIC EVALUATION BENCHMARK")
    print("=" * 85)

    physics_cfg_path = PROJECT_ROOT / "config" / "physics_config.yaml"
    sim = Spacecraft1D(config_path=physics_cfg_path)
    pd_controller = PDController.from_config(physics_cfg_path)
    env = SpacecraftAttitudeEnv()

    model = load_evaluation_model(model_path)
    scenarios = get_benchmark_scenarios(pd_controller)

    if output_dir is None:
        eval_dir = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4b" / "evaluations"
    else:
        eval_dir = output_dir

    if figures_dir is None:
        fig_dir = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4b" / "figures"
    else:
        fig_dir = figures_dir

    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ppo_records = []
    pd_records = []
    comparison_records = []

    for sc in scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        print(f"\n[EVAL] Simulating {sc_name}...")

        # 1. Simulate PPO
        ppo_res = simulate_ppo_trajectory(
            model=model,
            sim=sim,
            env=env,
            disturbance=sc["disturbance"],
            initial_state=sc["initial_state"],
            duration=sc["duration"],
        )

        # 2. Simulate Classical PD
        pd_res = simulate_pd_trajectory(
            controller=pd_controller,
            sim=sim,
            disturbance=sc["disturbance"],
            initial_state=sc["initial_state"],
            duration=sc["duration"],
        )

        # Export telemetry CSV files
        ppo_df = pd.DataFrame({k: v for k, v in ppo_res.items() if isinstance(v, np.ndarray)})
        pd_df = pd.DataFrame({k: v for k, v in pd_res.items() if isinstance(v, np.ndarray)})
        ppo_df.to_csv(eval_dir / f"telemetry_ppo_{sc_id}.csv", index=False)
        pd_df.to_csv(eval_dir / f"telemetry_pd_{sc_id}.csv", index=False)

        # 3. Compute Comprehensive AIAA Metrics
        ppo_m = calculate_comprehensive_metrics(
            time=ppo_res["time"],
            theta=ppo_res["theta"],
            theta_dot=ppo_res["theta_dot"],
            Omega=ppo_res["Omega"],
            torque=ppo_res["torque"],
            scenario_id=sc_id,
            target_theta=0.0,
            theoretical_theta_ss=sc["theoretical_theta_ss"],
            disturbance_torque=ppo_res["disturbance"],
        )

        pd_m = calculate_comprehensive_metrics(
            time=pd_res["time"],
            theta=pd_res["theta"],
            theta_dot=pd_res["theta_dot"],
            Omega=pd_res["Omega"],
            torque=pd_res["torque"],
            scenario_id=sc_id,
            target_theta=0.0,
            theoretical_theta_ss=sc["theoretical_theta_ss"],
            disturbance_torque=pd_res["disturbance"],
        )

        # 4. Generate Comparative Figure
        fig_path = fig_dir / f"comparison_{sc_id}.png"
        plot_scenario_comparison(ppo_res, pd_res, sc, fig_path)

        # Helper extractor
        def format_record(m: Dict[str, Any], theta_arr: np.ndarray, controller_type: str, term_early: bool = False) -> Dict[str, Any]:
            st = m.get("settling_time_s")
            st_val = float(st) if (st is not None and np.isfinite(st) and not term_early) else None
            dec = m.get("decay_time_90_to_10_s")
            dec_val = float(dec) if (dec is not None and np.isfinite(dec) and not term_early) else None
            ov = m.get("overshoot_pct")
            ov_val = float(ov) if (ov is not None and np.isfinite(ov)) else None
            rms_err = float(np.sqrt(np.mean(np.degrees(theta_arr)**2)))

            return {
                "scenario_id": sc_id,
                "controller": controller_type,
                "settling_time_s": st_val,
                "decay_time_s": dec_val,
                "rms_error_deg": rms_err,
                "overshoot_pct": ov_val,
                "final_error_deg": float(m.get("final_error_from_target_deg", 0.0)),
                "control_effort_N2m2s": float(m.get("squared_control_torque_integral_N2m2s", 0.0)),
                "max_wheel_speed_rpm": float(m.get("max_wheel_speed_rpm", 0.0)),
                "torque_sat_duration_s": float(m.get("torque_sat_duration_s", 0.0)),
                "wheel_sat_duration_s": float(m.get("wheel_sat_duration_s", 0.0)),
                "terminated_early": term_early,
            }

        rec_ppo = format_record(ppo_m, ppo_res["theta"], "PPO", ppo_res.get("terminated_early", False))
        rec_pd = format_record(pd_m, pd_res["theta"], "Classical_PD", pd_res.get("terminated_early", False))
        ppo_records.append(rec_ppo)
        pd_records.append(rec_pd)

        # Comparison summary record
        comp_rec = {
            "scenario": sc_id,
            "settling_time_ppo_s": rec_ppo["settling_time_s"],
            "settling_time_pd_s": rec_pd["settling_time_s"],
            "rms_error_ppo_deg": rec_ppo["rms_error_deg"],
            "rms_error_pd_deg": rec_pd["rms_error_deg"],
            "overshoot_ppo_pct": rec_ppo["overshoot_pct"],
            "overshoot_pd_pct": rec_pd["overshoot_pct"],
            "final_error_ppo_deg": rec_ppo["final_error_deg"],
            "final_error_pd_deg": rec_pd["final_error_deg"],
            "effort_ppo": rec_ppo["control_effort_N2m2s"],
            "effort_pd": rec_pd["control_effort_N2m2s"],
            "max_wheel_ppo_rpm": rec_ppo["max_wheel_speed_rpm"],
            "max_wheel_pd_rpm": rec_pd["max_wheel_speed_rpm"],
            "terminated_early_ppo": rec_ppo["terminated_early"],
        }
        comparison_records.append(comp_rec)

    # 5. Export Metrics
    comp_df = pd.DataFrame(comparison_records)
    comp_csv_path = eval_dir / "comparative_metrics_ppo_vs_pd.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\n[SAVE] Comparative metrics exported to: {comp_csv_path}")

    # Plot learning curve if evaluations exist
    eval_npz = eval_dir / "evaluations.npz"
    lc_path = fig_dir / "training_learning_curve.png"
    plot_learning_curve(eval_npz, lc_path)

    print("\n" + "=" * 85)
    print("COMPARATIVE EVALUATION SUMMARY (PPO VS. PD)")
    print("=" * 85)
    print(comp_df.to_string(index=False))
    print("=" * 85 + "\n")

    return comp_df, pd.DataFrame(ppo_records)

    env.close()
    return pd.DataFrame(ppo_records), comp_df


if __name__ == "__main__":
    run_full_evaluation()
