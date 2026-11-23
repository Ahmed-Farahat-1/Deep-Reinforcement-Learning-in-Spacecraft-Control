"""
Phase 4E: Final PPO Training Pipeline for Spacecraft Attitude Stabilization.

Trains from scratch using AntiSymmetricActorCriticPolicy, 4D state space,
redesigned reward, safe momentum initialization (|H0| <= 10.0 N*m*s),
and linear entropy annealing from 0.005 to 0.002.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml
import numpy as np
import torch as th
import gymnasium as gym

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
from src.agents.train_ppo import (
    TrainingScenarioWrapper,
    EvaluationScenarioWrapper,
    SpacecraftTelemetryCallback,
    load_training_config,
    evaluate_policy_health,
)


def make_phase4e_vectorized_env(
    n_envs: int = 4,
    scenario_distribution: Optional[Dict[str, float]] = None,
    seed: int = 42,
    include_integral: bool = True,
) -> DummyVecEnv:
    """Create vectorized Gymnasium training environments for Phase 4E."""
    def _make(idx: int):
        def _thunk():
            base_env = SpacecraftAttitudeEnv(include_integral=include_integral)
            wrapped_env = TrainingScenarioWrapper(base_env, scenario_distribution)
            monitored_env = Monitor(wrapped_env)
            monitored_env.reset(seed=seed + idx)
            return monitored_env
        return _thunk

    return DummyVecEnv([_make(i) for i in range(n_envs)])


def train_single_seed(
    seed: int = 42,
    total_timesteps: int = 500000,
    checkpoint_step: int = 100000,
    config_path: Optional[Path] = None,
) -> Tuple[PPO, Dict[str, Any]]:
    """
    Train a single PPO agent from scratch for a given seed up to total_timesteps.
    Preserves global optimizer and entropy schedule across the entire run.
    """
    cfg = load_training_config(config_path)
    ppo_cfg = cfg.get("ppo", {})
    scenario_dist = cfg.get("training_scenarios", {}).get("distribution", None)

    # Resolve hyperparameters
    lr = float(ppo_cfg.get("learning_rate", 3e-4))
    n_steps = int(ppo_cfg.get("n_steps", 2048))
    batch_size = int(ppo_cfg.get("batch_size", 256))
    n_epochs = int(ppo_cfg.get("n_epochs", 10))
    gamma = float(ppo_cfg.get("gamma", 0.998))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.2))
    ent_start = float(ppo_cfg.get("ent_coef", 0.005))
    ent_end = float(ppo_cfg.get("ent_coef_end", 0.002))  # Guaranteed >= 0.002
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
    n_envs = int(ppo_cfg.get("n_envs", 4))
    device = str(ppo_cfg.get("device", "cpu"))
    include_integral = bool(ppo_cfg.get("include_integral", True))

    # Directories for this specific seed
    base_out = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4e" / f"seed_{seed}"
    checkpoint_dir = base_out / "checkpoints"
    eval_log_dir = base_out / "evaluations"
    best_model_dir = base_out / "best_model"
    model_dir = PROJECT_ROOT / "models" / "ppo" / "phase_4e" / f"seed_{seed}"
    tensorboard_dir = PROJECT_ROOT / "logs" / "ppo" / "phase_4e" / f"seed_{seed}"

    for d in [base_out, checkpoint_dir, eval_log_dir, best_model_dir, model_dir, tensorboard_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"PHASE 4E: STARTING PPO TRAINING RUN — SEED {seed}")
    print("=" * 80)
    print(f"Algorithm:           PPO (Clean init from scratch)")
    print(f"Policy:              AntiSymmetricActorCriticPolicy (mu(s) = 0.5[f(s) - f(-s)])")
    print(f"Observation:         4D [theta, theta_dot, Omega, z] (anti-windup clamping & freezing)")
    print(f"Total Timesteps:     {total_timesteps:,}")
    print(f"Checkpoint Interval: ~{checkpoint_step:,} steps")
    print(f"Entropy Schedule:    {ent_start} -> {ent_end} (Linear Annealing, Floor=0.002)")
    print(f"Vectorized Workers:  {n_envs} CPU environments")
    print(f"Batch / Minibatch:   Rollout={n_steps * n_envs} | Batch={batch_size} | Epochs={n_epochs}")
    print(f"Discount (gamma):    {gamma} (Horizon ~5.0s, retains 18.4% weight at Ts=8.44s)")
    print(f"Outputs Directory:   {base_out}")
    print("=" * 80 + "\n")

    # Save configuration snapshot
    snapshot_path = base_out / "config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    # Create Environments
    train_env = make_phase4e_vectorized_env(
        n_envs=n_envs,
        scenario_distribution=scenario_dist,
        seed=seed,
        include_integral=include_integral,
    )

    eval_base_env = SpacecraftAttitudeEnv(include_integral=include_integral)
    eval_wrapped = EvaluationScenarioWrapper(eval_base_env)
    eval_env = DummyVecEnv([lambda: Monitor(eval_wrapped)])

    # Callbacks
    # Checkpoint every 25,000 steps so we have fine-grained checkpoints at 25k, 50k, 75k, 100k, ..., 500k
    chk_step_freq = max(1, 25000 // n_envs)
    eval_step_freq = max(1, 10000 // n_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=chk_step_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_step_freq,
        n_eval_episodes=8,  # Evaluates all 8 balanced scenarios
        deterministic=True,
        render=False,
        verbose=0,
    )

    telemetry_callback = SpacecraftTelemetryCallback(
        log_freq=2048,
        ent_start=ent_start,
        ent_end=ent_end,
        total_timesteps=total_timesteps,
        verbose=0,
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback, telemetry_callback])

    # Model Architecture
    net_arch = ppo_cfg.get("net_arch", {"pi": [64, 64], "vf": [64, 64]})
    policy_kwargs = {
        "net_arch": {
            "pi": net_arch.get("pi", [64, 64]),
            "vf": net_arch.get("vf", [64, 64]),
        }
    }

    model = PPO(
        policy=AntiSymmetricActorCriticPolicy,
        env=train_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_start,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        tensorboard_log=str(tensorboard_dir),
        verbose=0,
    )

    # Staged loop preserving continuous learning
    num_stages = max(1, total_timesteps // checkpoint_step)
    t_start = time.perf_counter()
    stage_records = []

    for stage_idx in range(1, num_stages + 1):
        target_steps = stage_idx * checkpoint_step
        print(f"\n--- [SEED {seed}] STAGE {stage_idx}/{num_stages} (Target: {target_steps:,} timesteps) ---")
        t_stage_start = time.perf_counter()

        model.learn(
            total_timesteps=checkpoint_step,
            callback=callback_list,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        t_stage_end = time.perf_counter()
        dur = t_stage_end - t_stage_start
        sps = checkpoint_step / max(dur, 1e-4)
        cum_steps = model.num_timesteps

        # Save milestone checkpoint
        milestone_path = model_dir / f"ppo_seed_{seed}_{cum_steps // 1000}k.zip"
        model.save(str(milestone_path))

        # Health diagnostics
        health = evaluate_policy_health(model, eval_base_env)
        print(f"  Stage {stage_idx} Completed: {cum_steps:,} steps | Duration: {dur:.1f}s | Speed: {sps:.1f} SPS")
        print(f"  Policy Std: {health['std']:.4f} | Current Ent Coef: {getattr(model, 'ent_coef', ent_end):.5f}")
        print(f"  Actions: u(+10°)={health['u_pos10']:+.4f} Nm, u(-10°)={health['u_neg10']:+.4f} Nm")
        print(f"  Direction Correct: {health['direction_correct']} | Symmetry Score: {health['symmetry_score']:.4f}")

        # Check for catastrophic collapse
        if health["near_zero_collapse"] and stage_idx >= 3:
            print(f"[WARNING] Near-zero action collapse detected at stage {stage_idx}!")

        stage_records.append({
            "stage": stage_idx,
            "cumulative_timesteps": cum_steps,
            "duration_s": dur,
            "sps": sps,
            "milestone_model": str(milestone_path),
            "health": health,
        })

    t_end = time.perf_counter()
    total_duration = t_end - t_start
    overall_sps = total_timesteps / max(total_duration, 1e-4)

    # Save final model for this seed
    final_seed_path = model_dir / f"ppo_seed_{seed}_final_500k.zip"
    model.save(str(final_seed_path))
    print(f"\n[SEED {seed} COMPLETE] Final model saved to: {final_seed_path}")
    print(f"Total Training Time: {total_duration:.1f}s ({total_duration/60.0:.2f} min) | Avg SPS: {overall_sps:.1f}")

    # Read evaluation data from EvalCallback
    eval_npz = eval_log_dir / "evaluations.npz"
    eval_history = {}
    if eval_npz.is_file():
        data = np.load(str(eval_npz))
        eval_history = {
            "timesteps": data["timesteps"].tolist(),
            "mean_rewards": [float(np.mean(r)) for r in data["results"]],
            "std_rewards": [float(np.std(r)) for r in data["results"]],
        }

    summary = {
        "seed": seed,
        "total_timesteps": int(model.num_timesteps),
        "total_duration_s": float(total_duration),
        "avg_sps": float(overall_sps),
        "final_model": str(final_seed_path),
        "best_eval_model": str(best_model_dir / "best_model.zip"),
        "stages": stage_records,
        "eval_history": eval_history,
        "telemetry": {
            "timesteps": telemetry_callback.logged_timesteps,
            "mean_theta_err_deg": telemetry_callback.logged_attitude_errors,
            "max_wheel_rpm": telemetry_callback.logged_wheel_speeds,
        },
    }

    summary_file = base_out / f"seed_{seed}_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    train_env.close()
    eval_env.close()
    return model, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4E PPO Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (42, 123, 2026)")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--checkpoint-step", type=int, default=100000, help="Milestone checkpoint interval")
    args = parser.parse_args()

    train_single_seed(seed=args.seed, total_timesteps=args.timesteps, checkpoint_step=args.checkpoint_step)
