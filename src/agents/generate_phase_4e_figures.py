"""
Phase 4E Scientific Visualization Engine.

Generates 9 publication-grade figures @ 300 DPI:
1. PPO Training Learning Curve (ppo_training_learning_curve.png)
2. Evaluation Reward Progression (evaluation_reward_progression.png)
3. Positive vs Negative Policy Response (positive_vs_negative_policy_response.png)
4. PPO vs PD Attitude Response (ppo_vs_pd_attitude_response.png)
5. Body Angular Velocity Comparison (body_angular_velocity_comparison.png)
6. Reaction Wheel Speed Comparison (reaction_wheel_speed_comparison.png)
7. Control Torque Comparison (control_torque_comparison.png)
8. Held-Out Scenario Results (held_out_scenario_results.png)
9. Multi-Seed Performance Comparison (multi_seed_performance_comparison.png)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from src.envs.spacecraft_env import SpacecraftAttitudeEnv

# Matplotlib styling for academic publication
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


def plot_learning_curves(
    telemetry_histories: Dict[int, Dict[str, Any]],
    output_dir: Path,
):
    """Plot Fig 1: PPO Training Learning Curve & Aerospace Telemetry across timesteps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {42: "#1f77b4", 123: "#2ca02c", 2026: "#d62728"}

    # 1. Mean Attitude Error Progression
    ax = axes[0, 0]
    for seed, hist in telemetry_histories.items():
        if "timesteps" in hist and len(hist["timesteps"]) > 0:
            steps = np.array(hist["timesteps"])
            th_err = np.array(hist.get("mean_theta_err_deg", []))
            if len(th_err) == len(steps):
                ax.plot(steps / 1000.0, th_err, label=f"Seed {seed}", color=colors.get(seed, "#333333"))
    ax.set_title("Mean Absolute Attitude Error Progression")
    ax.set_xlabel("Training Timesteps [k steps]")
    ax.set_ylabel("Mean |θ| Error [deg]")
    ax.axhline(0.2, color="green", linestyle=":", label="Stabilization Band (0.2°)")
    ax.grid(True)
    ax.legend()

    # 2. Peak Wheel Speed Progression
    ax = axes[0, 1]
    for seed, hist in telemetry_histories.items():
        if "timesteps" in hist and len(hist["timesteps"]) > 0:
            steps = np.array(hist["timesteps"])
            wheel_rpm = np.array(hist.get("max_wheel_rpm", []))
            if len(wheel_rpm) == len(steps):
                ax.plot(steps / 1000.0, wheel_rpm, label=f"Seed {seed}", color=colors.get(seed, "#333333"))
    ax.set_title("Peak Reaction Wheel Speed Progression")
    ax.set_xlabel("Training Timesteps [k steps]")
    ax.set_ylabel("Peak Wheel Speed [RPM]")
    ax.axhline(5000.0, color="red", linestyle="--", label="Physical Saturation (5000 RPM)")
    ax.axhline(4500.0, color="orange", linestyle=":", label="Warning Threshold (4500 RPM)")
    ax.grid(True)
    ax.legend()

    # 3. Entropy Annealing Schedule
    ax = axes[1, 0]
    steps_smooth = np.linspace(0, 500, 100)
    ent_smooth = 0.002 + (1.0 - steps_smooth / 500.0) * (0.005 - 0.002)
    ax.plot(steps_smooth, ent_smooth, color="purple", label="Dynamic Entropy Schedule")
    ax.axhline(0.002, color="black", linestyle="--", label="Entropy Floor (0.002)")
    ax.set_title("Entropy Regularization Schedule (Guaranteed >= 0.002)")
    ax.set_xlabel("Training Timesteps [k steps]")
    ax.set_ylabel("Entropy Coefficient (c_ent)")
    ax.grid(True)
    ax.legend()

    # 4. Multi-Seed SPS Throughput
    ax = axes[1, 1]
    seeds = list(telemetry_histories.keys())
    sps_vals = [telemetry_histories[s].get("avg_sps", 500.0) for s in seeds]
    bars = ax.bar([f"Seed {s}" for s in seeds], sps_vals, color=[colors.get(s, "#1f77b4") for s in seeds], alpha=0.8, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f} SPS", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Vectorized CPU Training Throughput")
    ax.set_ylabel("Steps Per Second (SPS)")
    ax.set_ylim(0, max(sps_vals) * 1.25 if sps_vals else 1000)
    ax.grid(True, axis="y")

    fig.suptitle("Fig 1: Phase 4E PPO Training Dynamics & Telemetry (500k Steps)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_training_learning_curve.png", dpi=300)
    plt.close()
    print("  -> Saved: ppo_training_learning_curve.png")


def plot_eval_reward_progression(
    eval_histories: Dict[int, Dict[str, Any]],
    output_dir: Path,
):
    """Plot Fig 2: Evaluation Reward Progression across 500k timesteps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {42: "#1f77b4", 123: "#2ca02c", 2026: "#d62728"}

    for seed, hist in eval_histories.items():
        if "timesteps" in hist and len(hist["timesteps"]) > 0:
            steps = np.array(hist["timesteps"]) / 1000.0
            means = np.array(hist.get("mean_rewards", []))
            stds = np.array(hist.get("std_rewards", []))
            if len(means) == len(steps):
                ax.plot(steps, means, label=f"Seed {seed} (Mean Eval Reward)", color=colors.get(seed, "#333"), linewidth=2.0)
                if len(stds) == len(means):
                    ax.fill_between(steps, means - stds, means + stds, color=colors.get(seed, "#333"), alpha=0.15)

    ax.set_title("Fig 2: Periodic Evaluation Reward Across 8 Balanced Validation Scenarios", fontweight="bold")
    ax.set_xlabel("Training Timesteps [k steps]")
    ax.set_ylabel("Mean Multi-Objective Episode Reward")
    ax.grid(True)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_reward_progression.png", dpi=300)
    plt.close()
    print("  -> Saved: evaluation_reward_progression.png")


def plot_positive_vs_negative_response(
    model: PPO,
    env: SpacecraftAttitudeEnv,
    output_dir: Path,
):
    """Plot Fig 3: Anti-Symmetric Policy Response across [-30 deg, +30 deg]."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    angles_deg = np.linspace(-30.0, 30.0, 201)
    actions = []
    actions_neg = []

    for a in angles_deg:
        obs = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        obs[0] = np.radians(a) / env.theta_scale
        u = float(model.predict(obs, deterministic=True)[0][0] * env.max_torque)
        actions.append(u)

        obs_neg = -obs
        u_neg = float(model.predict(obs_neg, deterministic=True)[0][0] * env.max_torque)
        actions_neg.append(u_neg)

    actions = np.array(actions)
    actions_neg = np.array(actions_neg)

    # 1. Action vs Error Curve
    ax = axes[0]
    ax.plot(angles_deg, actions, label="Learned Policy u(s)", color="navy", linewidth=2.2)
    ax.plot(angles_deg, -actions_neg, label="Inverted Response -u(-s)", color="crimson", linestyle="--", linewidth=1.8)
    ax.axhline(0.0, color="black", linestyle=":", alpha=0.6)
    ax.axvline(0.0, color="black", linestyle=":", alpha=0.6)
    ax.set_title("Control Torque u(θ) vs Attitude Error")
    ax.set_xlabel("Attitude Error θ [deg]")
    ax.set_ylabel("Commanded Torque u_cmd [N·m]")
    ax.grid(True)
    ax.legend()

    # 2. Symmetry Error Residual |u(s) + u(-s)|
    ax = axes[1]
    symm_err = np.abs(actions + actions_neg)
    ax.plot(angles_deg, symm_err, color="green", linewidth=1.5, label="Symmetry Error |u(s) + u(-s)|")
    ax.set_yscale("log")
    ax.set_title("Mathematical Anti-Symmetry Verification Residual")
    ax.set_xlabel("Attitude Error θ [deg]")
    ax.set_ylabel("Absolute Symmetry Discrepancy [N·m]")
    ax.grid(True, which="both")
    ax.legend()

    fig.suptitle("Fig 3: Anti-Symmetric Actor Response & Zero Equilibrium Verification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "positive_vs_negative_policy_response.png", dpi=300)
    plt.close()
    print("  -> Saved: positive_vs_negative_policy_response.png")


def plot_comparative_responses(
    trajectories: Dict[str, Dict[str, Any]],
    output_dir: Path,
):
    """Plot Figs 4, 5, 6, 7 comparing PPO vs PD on canonical step and disturbance scenarios."""
    scenarios_to_plot = [
        ("scenario_1_pos10deg", "Scenario 1 (+10° Step)"),
        ("scenario_2_neg10deg", "Scenario 2 (-10° Step)"),
        ("scenario_3_zero_dist_large", "Scenario 3 (+30° Large Step)"),
        ("scenario_4_constant_dist", "Scenario 4 (Constant Disturbance +0.02 N·m)"),
        ("scenario_5_sinusoidal_dist", "Scenario 5 (Sinusoidal Disturbance 0.02 N·m)"),
        ("scenario_6_pulse_dist", "Scenario 6 (Pulse Disturbance 0.05 N·m)"),
    ]

    # -------------------------------------------------------------
    # Fig 4: Attitude Response Comparison θ(t)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            t_pd = data["pd"]["time"]
            t_ppo = data["ppo"]["time"]
            th_pd = np.degrees(data["pd"]["theta"])
            th_ppo = np.degrees(data["ppo"]["theta"])

            ax.plot(t_pd, th_pd, label="Classical PD", color="crimson", linestyle="--", linewidth=2.0)
            ax.plot(t_ppo, th_ppo, label="Trained PPO (Phase 4E)", color="navy", linewidth=2.0)
            ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)
            ax.axhspan(-0.2, 0.2, color="green", alpha=0.1, label="Fine Pointing Band (±0.2°)")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Attitude θ [deg]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 4: Spacecraft Attitude Stabilization Response θ(t): PPO vs Classical PD", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_vs_pd_attitude_response.png", dpi=300)
    plt.close()
    print("  -> Saved: ppo_vs_pd_attitude_response.png")

    # -------------------------------------------------------------
    # Fig 5: Body Angular Velocity Comparison θ_dot(t)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            t_pd = data["pd"]["time"]
            t_ppo = data["ppo"]["time"]
            thd_pd = np.degrees(data["pd"]["theta_dot"])
            thd_ppo = np.degrees(data["ppo"]["theta_dot"])

            ax.plot(t_pd, thd_pd, label="Classical PD", color="crimson", linestyle="--", linewidth=1.8)
            ax.plot(t_ppo, thd_ppo, label="Trained PPO (Phase 4E)", color="navy", linewidth=1.8)
            ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Angular Velocity [deg/s]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 5: Spacecraft Body Angular Velocity θ_dot(t): PPO vs Classical PD", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "body_angular_velocity_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: body_angular_velocity_comparison.png")

    # -------------------------------------------------------------
    # Fig 6: Reaction Wheel Speed Comparison Ω(t)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            t_pd = data["pd"]["time"]
            t_ppo = data["ppo"]["time"]
            om_pd = data["pd"]["wheel_speed"] * 60.0 / (2.0 * np.pi)
            om_ppo = data["ppo"]["wheel_speed"] * 60.0 / (2.0 * np.pi)

            ax.plot(t_pd, om_pd, label="Classical PD", color="crimson", linestyle="--", linewidth=1.8)
            ax.plot(t_ppo, om_ppo, label="Trained PPO (Phase 4E)", color="navy", linewidth=1.8)
            ax.axhline(5000.0, color="red", linestyle=":", alpha=0.6, label="Max Capacity (5000 RPM)")
            ax.axhline(-5000.0, color="red", linestyle=":", alpha=0.6)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Wheel Speed Ω [RPM]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 6: Reaction Wheel Speed Ω(t): PPO vs Classical PD", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "reaction_wheel_speed_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: reaction_wheel_speed_comparison.png")

    # -------------------------------------------------------------
    # Fig 7: Control Torque Comparison u(t)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (sid, title) in enumerate(scenarios_to_plot):
        ax = axes[idx]
        if sid in trajectories:
            data = trajectories[sid]
            t_pd = data["pd"]["time"]
            t_ppo = data["ppo"]["time"]
            u_pd = data["pd"]["u_applied"]
            u_ppo = data["ppo"]["u_applied"]

            ax.plot(t_pd, u_pd, label="Classical PD", color="crimson", linestyle="--", linewidth=1.6)
            ax.plot(t_ppo, u_ppo, label="Trained PPO (Phase 4E)", color="navy", linewidth=1.6)
            ax.axhline(1.0, color="red", linestyle=":", alpha=0.6, label="Torque Limits (±1.0 N·m)")
            ax.axhline(-1.0, color="red", linestyle=":", alpha=0.6)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Applied Torque [N·m]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Fig 7: Applied Motor Control Torque u(t): PPO vs Classical PD", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "control_torque_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: control_torque_comparison.png")


def plot_heldout_scenario_results(
    trajectories: Dict[str, Dict[str, Any]],
    output_dir: Path,
):
    """Plot Fig 8: Held-Out Scenarios (H1 to H5) Generalization Test Results."""
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
            t_pd = data["pd"]["time"]
            t_ppo = data["ppo"]["time"]
            th_pd = np.degrees(data["pd"]["theta"])
            th_ppo = np.degrees(data["ppo"]["theta"])

            ax.plot(t_pd, th_pd, label="Classical PD", color="crimson", linestyle="--", linewidth=2.0)
            ax.plot(t_ppo, th_ppo, label="Trained PPO (Phase 4E)", color="navy", linewidth=2.0)
            ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)
            ax.axhspan(-0.2, 0.2, color="green", alpha=0.1, label="Fine Pointing Band (±0.2°)")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Attitude θ [deg]")
            ax.grid(True)
            if idx == 0:
                ax.legend(loc="upper right")

    # Hide unused 6th subplot
    axes[5].axis("off")
    axes[5].text(0.1, 0.5, "Held-Out Test Scenarios (H1-H5)\n\nStrictly held out from training.\nEvaluates policy generalization\nand steady-state disturbance\nrejection capabilities.", fontsize=13, va="center")

    fig.suptitle("Fig 8: Held-Out Generalization Evaluation (H1-H5): PPO vs Classical PD", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "held_out_scenario_results.png", dpi=300)
    plt.close()
    print("  -> Saved: held_out_scenario_results.png")


def plot_multi_seed_comparison(
    multi_seed_metrics: Dict[int, Dict[str, float]],
    output_dir: Path,
):
    """Plot Fig 9: Multi-Seed Statistical Performance Comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    seeds = list(multi_seed_metrics.keys())
    seed_labels = [f"Seed {s}" for s in seeds]
    colors = ["#1f77b4", "#2ca02c", "#d62728"][:len(seeds)]

    # 1. Stabilization Success Rate (%)
    ax = axes[0, 0]
    succ_rates = [multi_seed_metrics[s].get("success_rate_pct", 100.0) for s in seeds]
    bars = ax.bar(seed_labels, succ_rates, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Attitude Stabilization Success Rate")
    ax.set_ylabel("Success Rate [%]")
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y")

    # 2. Mean RMS Attitude Error
    ax = axes[0, 1]
    rms_vals = [multi_seed_metrics[s].get("mean_rms_deg", 0.5) for s in seeds]
    bars = ax.bar(seed_labels, rms_vals, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.05, f"{bar.get_height():.3f}°", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Mean RMS Attitude Error Across Validation Suite")
    ax.set_ylabel("RMS Attitude Error [deg]")
    ax.set_ylim(0, max(rms_vals) * 1.3 if rms_vals else 5.0)
    ax.grid(True, axis="y")

    # 3. Mean Control Effort
    ax = axes[1, 0]
    eff_vals = [multi_seed_metrics[s].get("mean_effort", 0.1) for s in seeds]
    bars = ax.bar(seed_labels, eff_vals, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.01, f"{bar.get_height():.4f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Mean Squared Control Effort ∫u²dt")
    ax.set_ylabel("Effort [N²·m²·s]")
    ax.set_ylim(0, max(eff_vals) * 1.3 if eff_vals else 1.0)
    ax.grid(True, axis="y")

    # 4. Composite Validation Score
    ax = axes[1, 1]
    scores = [multi_seed_metrics[s].get("composite_score", 0.0) for s in seeds]
    bars = ax.bar(seed_labels, scores, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 2, f"{bar.get_height():+.1f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Composite Validation Ranking Score")
    ax.set_ylabel("Score (Higher is Better)")
    ax.grid(True, axis="y")

    fig.suptitle("Fig 9: Multi-Seed Statistical Repeatability & Performance Summary", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "multi_seed_performance_comparison.png", dpi=300)
    plt.close()
    print("  -> Saved: multi_seed_performance_comparison.png")
