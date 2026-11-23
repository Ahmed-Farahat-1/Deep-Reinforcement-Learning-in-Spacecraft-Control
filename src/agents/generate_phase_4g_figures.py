"""
Phase 4G Scientific Visualization Engine (10 Publication-Grade Figures @ 300 DPI).

1. TD3 Training Learning Curve (td3_training_learning_curve.png)
2. SAC Training Learning Curve (sac_training_learning_curve.png)
3. TD3 vs SAC Validation Progression (td3_vs_sac_validation_progression.png)
4. Positive vs Negative Action Response (positive_vs_negative_action_response.png)
5. PPO vs TD3 vs SAC vs PD Attitude Response (ppo_vs_td3_vs_sac_vs_pd_attitude_response.png)
6. Body Angular Velocity Comparison (body_angular_velocity_comparison.png)
7. Reaction Wheel Speed Comparison (reaction_wheel_speed_comparison.png)
8. Applied Motor Torque Comparison (applied_motor_torque_comparison.png)
9. Held-Out Generalization Comparison (held_out_generalization_comparison.png)
10. Multi-Seed Statistical Summary (multi_seed_statistical_summary.png)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import TD3, SAC, PPO
from src.envs.spacecraft_env import SpacecraftAttitudeEnv

# Styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def plot_td3_learning_curves(telemetries: Dict[int, Dict[str, Any]], output_dir: Path):
    """Fig 1: TD3 Training Learning Curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {42: "#1f77b4", 123: "#2ca02c", 2026: "#d62728"}

    # 1. Attitude Error
    ax = axes[0]
    for s, data in telemetries.items():
        if "timesteps" in data and len(data["timesteps"]) > 0:
            steps = np.array(data["timesteps"]) / 1000.0
            errs = np.array(data.get("mean_theta_err_deg", []))
            ax.plot(steps, errs, label=f"Seed {s}", color=colors.get(s, "black"))
    ax.set_title("TD3: Mean Absolute Attitude Error Progression")
    ax.set_xlabel("Training Steps [k steps]")
    ax.set_ylabel("Mean |θ| Error [deg]")
    ax.grid(True)
    ax.legend()

    # 2. Max Wheel Speed
    ax = axes[1]
    for s, data in telemetries.items():
        if "timesteps" in data and len(data["timesteps"]) > 0:
            steps = np.array(data["timesteps"]) / 1000.0
            rpms = np.array(data.get("max_wheel_rpm", []))
            ax.plot(steps, rpms, label=f"Seed {s}", color=colors.get(s, "black"))
    ax.axhline(5000.0, color="red", linestyle=":", label="Speed Limit (5000 RPM)")
    ax.axhline(4500.0, color="orange", linestyle="--", label="Warning Threshold (4500 RPM)")
    ax.set_title("TD3: Peak Reaction Wheel Speed Progression")
    ax.set_xlabel("Training Steps [k steps]")
    ax.set_ylabel("Peak Wheel Speed [RPM]")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Fig 1: Pure TD3 Multi-Seed Training Telemetry & Learning Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "td3_training_learning_curve.png", dpi=300)
    plt.close()
    print("  -> Saved: td3_training_learning_curve.png")


def plot_sac_learning_curves(telemetries: Dict[int, Dict[str, Any]], output_dir: Path):
    """Fig 2: SAC Training Learning Curve."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {42: "#1f77b4", 123: "#2ca02c", 2026: "#d62728"}

    # 1. Attitude Error
    ax = axes[0]
    for s, data in telemetries.items():
        if "timesteps" in data and len(data["timesteps"]) > 0:
            steps = np.array(data["timesteps"]) / 1000.0
            errs = np.array(data.get("mean_theta_err_deg", []))
            ax.plot(steps, errs, label=f"Seed {s}", color=colors.get(s, "black"))
    ax.set_title("SAC: Mean Absolute Attitude Error")
    ax.set_xlabel("Training Steps [k steps]")
    ax.set_ylabel("Mean |θ| Error [deg]")
    ax.grid(True)
    ax.legend()

    # 2. Entropy Temperature (alpha)
    ax = axes[1]
    for s, data in telemetries.items():
        if "timesteps" in data and len(data["timesteps"]) > 0:
            steps = np.array(data["timesteps"]) / 1000.0
            alphas = np.array(data.get("ent_coefs", []))
            if len(alphas) == len(steps):
                ax.plot(steps, alphas, label=f"Seed {s}", color=colors.get(s, "black"))
    ax.set_title("SAC: Adaptive Entropy Temperature (α)")
    ax.set_xlabel("Training Steps [k steps]")
    ax.set_ylabel("Entropy Coefficient α")
    ax.grid(True)
    ax.legend()

    # 3. Max Wheel Speed
    ax = axes[2]
    for s, data in telemetries.items():
        if "timesteps" in data and len(data["timesteps"]) > 0:
            steps = np.array(data["timesteps"]) / 1000.0
            rpms = np.array(data.get("max_wheel_rpm", []))
            ax.plot(steps, rpms, label=f"Seed {s}", color=colors.get(s, "black"))
    ax.axhline(5000.0, color="red", linestyle=":", label="Speed Limit (5000 RPM)")
    ax.set_title("SAC: Peak Reaction Wheel Speed")
    ax.set_xlabel("Training Steps [k steps]")
    ax.set_ylabel("Peak Wheel Speed [RPM]")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Fig 2: Pure SAC Multi-Seed Training Telemetry & Temperature Adaptation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sac_training_learning_curve.png", dpi=300)
    plt.close()
    print("  -> Saved: sac_training_learning_curve.png")


def plot_td3_vs_sac_validation(df_td3_rank: pd.DataFrame, df_sac_rank: pd.DataFrame, output_dir: Path):
    """Fig 3: TD3 vs SAC Validation Progression across Checkpoints."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract step from model_file
    def _get_step(fn):
        parts = str(fn).replace(".zip", "").split("_")
        for p in parts:
            if p.isdigit():
                return int(p)
        return 0

    df_t = df_td3_rank.copy()
    df_s = df_sac_rank.copy()
    df_t["step"] = df_t["model_file"].apply(_get_step)
    df_s["step"] = df_s["model_file"].apply(_get_step)
    df_t = df_t.sort_values(by="step")
    df_s = df_s.sort_values(by="step")

    # 1. Composite Score
    ax = axes[0]
    ax.plot(df_t["step"] / 1000.0, df_t["composite_score"], marker="o", label="TD3 Checkpoints", color="navy")
    ax.plot(df_s["step"] / 1000.0, df_s["composite_score"], marker="s", label="SAC Checkpoints", color="darkgreen")
    ax.set_title("Validation Composite Score Progression")
    ax.set_xlabel("Checkpoint Timestep [k steps]")
    ax.set_ylabel("Composite Ranking Score")
    ax.grid(True)
    ax.legend()

    # 2. Final Attitude Error
    ax = axes[1]
    ax.plot(df_t["step"] / 1000.0, df_t["mean_final_err_deg"], marker="o", label="TD3 Final Error", color="navy")
    ax.plot(df_s["step"] / 1000.0, df_s["mean_final_err_deg"], marker="s", label="SAC Final Error", color="darkgreen")
    ax.set_title("Validation Mean Final Attitude Error [deg]")
    ax.set_xlabel("Checkpoint Timestep [k steps]")
    ax.set_ylabel("Final Error [deg]")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Fig 3: TD3 vs SAC Dedicated Validation Suite Progression", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "td3_vs_sac_validation_progression.png", dpi=300)
    plt.close()
    print("  -> Saved: td3_vs_sac_validation_progression.png")


def plot_positive_vs_negative_response(td3_model: Any, sac_model: Any, env: SpacecraftAttitudeEnv, output_dir: Path):
    """Fig 4: Positive vs Negative Action Response & Symmetry Verification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    angles_deg = np.linspace(-30.0, 30.0, 201)

    td3_acts, sac_acts = [], []
    for a in angles_deg:
        obs = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        obs[0] = np.radians(a) / env.theta_scale
        u_td3 = float(td3_model.predict(obs, deterministic=True)[0][0] * env.max_torque)
        u_sac = float(sac_model.predict(obs, deterministic=True)[0][0] * env.max_torque)
        td3_acts.append(u_td3)
        sac_acts.append(u_sac)

    td3_acts = np.array(td3_acts)
    sac_acts = np.array(sac_acts)

    # 1. Action vs Angle
    ax = axes[0]
    ax.plot(angles_deg, td3_acts, label="Pure TD3 u(θ)", color="navy", linewidth=2.0)
    ax.plot(angles_deg, sac_acts, label="Pure SAC u(θ)", color="darkgreen", linewidth=2.0)
    ax.axhline(0.0, color="black", linestyle=":", alpha=0.6)
    ax.axvline(0.0, color="black", linestyle=":", alpha=0.6)
    ax.set_title("Commanded Motor Torque u_cmd(θ) vs Attitude Error")
    ax.set_xlabel("Attitude Error θ [deg]")
    ax.set_ylabel("Commanded Torque [N·m]")
    ax.grid(True)
    ax.legend()

    # 2. Symmetry Error Residual |u(θ) + u(-θ)|
    ax = axes[1]
    res_td3 = np.abs(td3_acts + td3_acts[::-1])
    res_sac = np.abs(sac_acts + sac_acts[::-1])
    ax.plot(angles_deg, res_td3, label="TD3 Residual |u(θ) + u(-θ)|", color="navy")
    ax.plot(angles_deg, res_sac, label="SAC Residual |u(θ) + u(-θ)|", color="darkgreen", linestyle="--")
    ax.set_title("Odd-Symmetry Discrepancy Residual")
    ax.set_xlabel("Attitude Error θ [deg]")
    ax.set_ylabel("Symmetry Discrepancy [N·m]")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Fig 4: Anti-Symmetric Deterministic Response: TD3 vs SAC", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "positive_vs_negative_action_response.png", dpi=300)
    plt.close()
    print("  -> Saved: positive_vs_negative_action_response.png")


def plot_comparative_responses(trajectories: Dict[str, Dict[str, Any]], output_dir: Path):
    """Plot Figs 5, 6, 7, 8 comparing PPO vs TD3 vs SAC vs Classical PD."""
    scenarios_to_plot = [
        ("scenario_1_pos10deg", "Scenario 1 (+10° Step)"),
        ("scenario_2_neg10deg", "Scenario 2 (-10° Step)"),
        ("scenario_3_zero_dist_large", "Scenario 3 (+30° Large Step)"),
        ("scenario_4_constant_dist", "Scenario 4 (Constant Dist 0.02 N·m)"),
        ("scenario_5_sinusoidal_dist", "Scenario 5 (Sinusoidal Dist 0.02 N·m)"),
        ("scenario_6_pulse_dist", "Scenario 6 (Pulse Dist 0.05 N·m)"),
    ]

    # -------------------------------------------------------------------------
    # Fig 5: Attitude Response Comparison θ(t)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            ax.plot(data["pd"]["time"], np.degrees(data["pd"]["theta"]), label="Classical PD", color="crimson", linestyle="--", linewidth=2.0)
            ax.plot(data["ppo"]["time"], np.degrees(data["ppo"]["theta"]), label="PPO (Phase 4E)", color="purple", linewidth=1.8)
            ax.plot(data["td3"]["time"], np.degrees(data["td3"]["theta"]), label="Pure TD3", color="navy", linewidth=2.0)
            ax.plot(data["sac"]["time"], np.degrees(data["sac"]["theta"]), label="Pure SAC", color="darkgreen", linewidth=2.0)
            ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)
            ax.axhspan(-0.2, 0.2, color="green", alpha=0.1)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Attitude θ [deg]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 5: Spacecraft Attitude Stabilization Response θ(t): PD vs PPO vs TD3 vs SAC", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_vs_td3_vs_sac_vs_pd_attitude_response.png", dpi=300)
    plt.close()
    print("  -> Saved: ppo_vs_td3_vs_sac_vs_pd_attitude_response.png")

    # -------------------------------------------------------------------------
    # Fig 6: Body Angular Velocity Comparison θ_dot(t)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            ax.plot(data["pd"]["time"], np.degrees(data["pd"]["theta_dot"]), label="Classical PD", color="crimson", linestyle="--", linewidth=1.8)
            ax.plot(data["ppo"]["time"], np.degrees(data["ppo"]["theta_dot"]), label="PPO (Phase 4E)", color="purple", linewidth=1.6)
            ax.plot(data["td3"]["time"], np.degrees(data["td3"]["theta_dot"]), label="Pure TD3", color="navy", linewidth=1.8)
            ax.plot(data["sac"]["time"], np.degrees(data["sac"]["theta_dot"]), label="Pure SAC", color="darkgreen", linewidth=1.8)
            ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Body Rate [deg/s]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 6: Body Angular Velocity Damping θ_dot(t): PD vs PPO vs TD3 vs SAC", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "body_angular_velocity_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: body_angular_velocity_comparison.png")

    # -------------------------------------------------------------------------
    # Fig 7: Reaction Wheel Speed Comparison Ω(t)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            ax.plot(data["pd"]["time"], data["pd"]["wheel_speed"] * 60.0 / (2.0 * np.pi), label="Classical PD", color="crimson", linestyle="--", linewidth=1.8)
            ax.plot(data["ppo"]["time"], data["ppo"]["wheel_speed"] * 60.0 / (2.0 * np.pi), label="PPO", color="purple", linewidth=1.6)
            ax.plot(data["td3"]["time"], data["td3"]["wheel_speed"] * 60.0 / (2.0 * np.pi), label="Pure TD3", color="navy", linewidth=1.8)
            ax.plot(data["sac"]["time"], data["sac"]["wheel_speed"] * 60.0 / (2.0 * np.pi), label="Pure SAC", color="darkgreen", linewidth=1.8)
            ax.axhline(5000.0, color="red", linestyle=":", alpha=0.6)
            ax.axhline(-5000.0, color="red", linestyle=":", alpha=0.6)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Wheel Speed [RPM]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 7: Reaction Wheel Speed Ω(t): PD vs PPO vs TD3 vs SAC", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "reaction_wheel_speed_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: reaction_wheel_speed_comparison.png")

    # -------------------------------------------------------------------------
    # Fig 8: Applied Motor Torque Comparison u(t)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            ax.plot(data["pd"]["time"], data["pd"]["u_applied"], label="Classical PD", color="crimson", linestyle="--", linewidth=1.6)
            ax.plot(data["ppo"]["time"], data["ppo"]["u_applied"], label="PPO", color="purple", linewidth=1.4)
            ax.plot(data["td3"]["time"], data["td3"]["u_applied"], label="Pure TD3", color="navy", linewidth=1.6)
            ax.plot(data["sac"]["time"], data["sac"]["u_applied"], label="Pure SAC", color="darkgreen", linewidth=1.6)
            ax.axhline(1.0, color="red", linestyle=":", alpha=0.6)
            ax.axhline(-1.0, color="red", linestyle=":", alpha=0.6)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Applied Torque [N·m]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 8: Applied Motor Control Torque u(t): PD vs PPO vs TD3 vs SAC", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "applied_motor_torque_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: applied_motor_torque_comparison.png")


def plot_heldout_generalization(trajectories: Dict[str, Dict[str, Any]], output_dir: Path):
    """Fig 9: Held-Out Scenarios (H1 to H5) Generalization Comparison."""
    heldout_keys = [
        ("heldout_1_pos17_5deg", "H1: Unseen Step +17.5°"),
        ("heldout_2_neg22_5deg", "H2: Unseen Step -22.5°"),
        ("heldout_3_mixed_state", "H3: Mixed (+15°, -1.5°/s, +500 RPM)"),
        ("heldout_4_high_freq_sine", "H4: High-Freq Sine (0.25 Hz, 0.03 N·m)"),
        ("heldout_5_strong_constant", "H5: Strong Disturbance (-0.035 N·m)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (hid, title) in enumerate(heldout_keys):
        ax = axes[idx]
        if hid in trajectories:
            data = trajectories[hid]
            ax.plot(data["pd"]["time"], np.degrees(data["pd"]["theta"]), label="Classical PD", color="crimson", linestyle="--", linewidth=2.0)
            ax.plot(data["ppo"]["time"], np.degrees(data["ppo"]["theta"]), label="PPO (Phase 4E)", color="purple", linewidth=1.8)
            ax.plot(data["td3"]["time"], np.degrees(data["td3"]["theta"]), label="Pure TD3", color="navy", linewidth=2.0)
            ax.plot(data["sac"]["time"], np.degrees(data["sac"]["theta"]), label="Pure SAC", color="darkgreen", linewidth=2.0)
            ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)
            ax.axhspan(-0.2, 0.2, color="green", alpha=0.1)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Attitude θ [deg]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    axes[5].axis("off")
    axes[5].text(0.1, 0.5, "Held-Out Test Scenarios (H1–H5)\n\nStrictly held out from training.\nEvaluates generalization to unseen\nmaneuvers, composite initial states,\nand external disturbance regimes.", fontsize=13, va="center")

    fig.suptitle("Fig 9: Held-Out Generalization Evaluation (H1–H5): PD vs PPO vs TD3 vs SAC", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "held_out_generalization_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: held_out_generalization_comparison.png")


def plot_multi_seed_summary(df_multi: pd.DataFrame, output_dir: Path):
    """Fig 10: Multi-Seed Statistical Summary across PPO, TD3, and SAC."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    algos = df_multi["algorithm"].unique()
    colors = {"PPO": "purple", "TD3": "navy", "SAC": "darkgreen"}

    # 1. Mean Final Error by Algorithm
    ax = axes[0]
    for algo in algos:
        sub = df_multi[df_multi["algorithm"] == algo]
        ax.bar(
            [f"{algo}\nS{s}" for s in sub["seed"]],
            sub["mean_final_err_deg"],
            color=colors.get(algo, "gray"),
            alpha=0.85,
        )
    ax.set_title("Mean Final Attitude Error [deg]")
    ax.set_ylabel("Final Error [deg] (Lower is Better)")
    ax.grid(True, axis="y")

    # 2. Control Effort
    ax = axes[1]
    for algo in algos:
        sub = df_multi[df_multi["algorithm"] == algo]
        ax.bar(
            [f"{algo}\nS{s}" for s in sub["seed"]],
            sub["mean_effort"],
            color=colors.get(algo, "gray"),
            alpha=0.85,
        )
    ax.set_title("Mechanical Control Effort (∫ u² dt)")
    ax.set_ylabel("Effort [N²·m²·s]")
    ax.grid(True, axis="y")

    # 3. Peak Reaction Wheel Speed
    ax = axes[2]
    for algo in algos:
        sub = df_multi[df_multi["algorithm"] == algo]
        ax.bar(
            [f"{algo}\nS{s}" for s in sub["seed"]],
            sub["max_wheel_rpm"],
            color=colors.get(algo, "gray"),
            alpha=0.85,
        )
    ax.axhline(5000.0, color="red", linestyle=":", label="Limit (5000 RPM)")
    ax.set_title("Peak Reaction Wheel Speed [RPM]")
    ax.set_ylabel("Wheel Speed [RPM]")
    ax.grid(True, axis="y")
    ax.legend()

    fig.suptitle("Fig 10: Multi-Seed Statistical Comparison: PPO vs TD3 vs SAC", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "multi_seed_statistical_summary.png", dpi=300)
    plt.close()
    print("  -> Saved: multi_seed_statistical_summary.png")
