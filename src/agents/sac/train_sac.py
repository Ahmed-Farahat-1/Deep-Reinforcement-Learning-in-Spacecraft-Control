"""
Phase 4G: SAC Training Pipeline for Spacecraft Attitude Stabilization.

Implements Soft Actor-Critic using AntiSymmetricSACPolicy, 4D observation space,
automatic entropy temperature tuning (ent_coef="auto_0.1", target_entropy=-1.0),
off-policy experience replay, and resumable checkpoints (saving both model and replay buffer).
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.agents.train_ppo import TrainingScenarioWrapper, EvaluationScenarioWrapper
from src.agents.sac.sac_policy import AntiSymmetricSACPolicy, verify_sac_anti_symmetry
from src.agents.td3.train_td3 import OffPolicyCheckpointCallback


def load_sac_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load SAC configuration YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "sac_config.yaml"
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SACTelemetryCallback(BaseCallback):
    """
    Tracks and logs aerospace GNC metrics and entropy temperature during SAC training.
    """

    def __init__(self, log_freq: int = 2500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.logged_timesteps: List[int] = []
        self.logged_rewards: List[float] = []
        self.logged_attitude_errors: List[float] = []
        self.logged_wheel_speeds: List[float] = []
        self.logged_ent_coefs: List[float] = []
        self.recent_theta_errors: List[float] = []
        self.recent_wheel_speeds: List[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            st = info.get("state", None)
            if st is not None:
                self.recent_theta_errors.append(abs(float(st[0])))
                self.recent_wheel_speeds.append(abs(float(st[2])) * 60.0 / (2.0 * np.pi))

        if self.n_calls % self.log_freq == 0:
            step = self.num_timesteps
            mean_th = float(np.mean(self.recent_theta_errors)) if self.recent_theta_errors else 0.0
            max_om = float(np.max(self.recent_wheel_speeds)) if self.recent_wheel_speeds else 0.0
            ent_coef = float(self.model.log_ent_coef.exp().item()) if hasattr(self.model, "log_ent_coef") else 0.0
            self.logged_timesteps.append(step)
            self.logged_attitude_errors.append(float(np.degrees(mean_th)))
            self.logged_wheel_speeds.append(max_om)
            self.logged_ent_coefs.append(ent_coef)
            self.recent_theta_errors.clear()
            self.recent_wheel_speeds.clear()
        return True


def evaluate_sac_health(model: SAC, env: SpacecraftAttitudeEnv) -> Dict[str, Any]:
    """Inspect SAC policy torque actions, anti-symmetry, direction correctness, and entropy coefficient."""
    obs_dim = env.observation_space.shape[0]

    obs_pos = np.zeros(obs_dim, dtype=np.float32)
    obs_pos[0] = np.radians(10.0) / env.theta_scale
    u_pos_norm = float(model.predict(obs_pos, deterministic=True)[0][0])
    u_pos = u_pos_norm * env.max_torque

    obs_neg = np.zeros(obs_dim, dtype=np.float32)
    obs_neg[0] = np.radians(-10.0) / env.theta_scale
    u_neg_norm = float(model.predict(obs_neg, deterministic=True)[0][0])
    u_neg = u_neg_norm * env.max_torque

    is_dir_correct = (u_pos > 0.0) and (u_neg < 0.0)
    is_symm, max_disc, zero_val = verify_sac_anti_symmetry(model, obs_dim=obs_dim, n_samples=50)
    ent_coef = float(model.log_ent_coef.exp().item()) if hasattr(model, "log_ent_coef") else 0.0

    return {
        "u_pos_10deg_nm": u_pos,
        "u_neg_10deg_nm": u_neg,
        "action_magnitude": abs(u_pos),
        "direction_correct": is_dir_correct,
        "is_anti_symmetric": is_symm,
        "max_symmetry_discrepancy": max_disc,
        "zero_equilibrium_value": zero_val,
        "ent_coef": ent_coef,
    }


def make_vectorized_sac_env(
    n_envs: int = 4,
    scenario_distribution: Optional[Dict[str, float]] = None,
    seed: int = 42,
    include_integral: bool = True,
) -> DummyVecEnv:
    """Create vectorized Gymnasium training environments for SAC."""
    def _make(idx: int):
        def _thunk():
            base_env = SpacecraftAttitudeEnv(include_integral=include_integral)
            wrapped_env = TrainingScenarioWrapper(base_env, scenario_distribution)
            monitored_env = Monitor(wrapped_env)
            monitored_env.reset(seed=seed + idx)
            return monitored_env
        return _thunk

    return DummyVecEnv([_make(i) for i in range(n_envs)])


def train_single_seed_sac(
    seed: int = 42,
    total_timesteps: int = 500000,
    checkpoint_step: int = 50000,
    config_path: Optional[Path] = None,
    output_tag: Optional[str] = None,
) -> Tuple[SAC, Dict[str, Any]]:
    """
    Train a single SAC agent from scratch for a given seed up to total_timesteps.
    """
    cfg = load_sac_config(config_path)
    sac_cfg = cfg.get("sac", {})
    scenario_dist = cfg.get("training_scenarios", {}).get("distribution", None)

    lr = float(sac_cfg.get("learning_rate", 3e-4))
    buffer_size = int(sac_cfg.get("buffer_size", 500000))
    learning_starts = int(sac_cfg.get("learning_starts", 10000))
    if total_timesteps <= learning_starts:
        learning_starts = min(1000, max(100, total_timesteps // 5))
    batch_size = int(sac_cfg.get("batch_size", 256))
    tau = float(sac_cfg.get("tau", 0.005))
    gamma = float(sac_cfg.get("gamma", 0.998))
    train_freq = int(sac_cfg.get("train_freq", 1))
    gradient_steps = int(sac_cfg.get("gradient_steps", 4))
    ent_coef = sac_cfg.get("ent_coef", "auto_0.1")
    target_entropy = float(sac_cfg.get("target_entropy", -1.0))
    n_envs = int(sac_cfg.get("n_envs", 4))
    device = str(sac_cfg.get("device", "cpu"))
    include_integral = bool(sac_cfg.get("include_integral", True))

    tag = output_tag if output_tag is not None else f"seed_{seed}"
    base_out = PROJECT_ROOT / "outputs" / "sac" / "phase_4g" / tag
    checkpoint_dir = base_out / "checkpoints"
    eval_log_dir = base_out / "evaluations"
    model_dir = PROJECT_ROOT / "models" / "sac" / "phase_4g" / tag
    tensorboard_dir = PROJECT_ROOT / "logs" / "sac" / "phase_4g" / tag

    for d in [base_out, checkpoint_dir, eval_log_dir, model_dir, tensorboard_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"PHASE 4G: STARTING SAC TRAINING RUN — SEED {seed}")
    print("=" * 80)
    print(f"Algorithm:           SAC (Soft Actor-Critic from scratch)")
    print(f"Policy:              AntiSymmetricSACPolicy (mu(s) = 0.5[f(s) - f(-s)])")
    print(f"Observation:         4D [theta, theta_dot, Omega, z] (anti-windup clamping & freezing)")
    print(f"Total Timesteps:     {total_timesteps:,}")
    print(f"Checkpoint Interval: ~{checkpoint_step:,} steps")
    print(f"Entropy Temp Tuning: Auto (ent_coef={ent_coef}, target_entropy={target_entropy})")
    print(f"Buffer / Batch:      Buffer={buffer_size:,} | Batch={batch_size} | Starts={learning_starts:,}")
    print(f"Discount (gamma):    {gamma} | Tau={tau}")
    print(f"Vectorized Workers:  {n_envs} CPU environments (train_freq={train_freq}, grad_steps={gradient_steps})")
    print(f"Outputs Directory:   {base_out}")
    print("=" * 80 + "\n")

    # Save config snapshot
    snapshot_path = base_out / "config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    train_env = make_vectorized_sac_env(
        n_envs=n_envs,
        scenario_distribution=scenario_dist,
        seed=seed,
        include_integral=include_integral,
    )

    policy_kwargs = {
        "net_arch": [64, 64],
        "activation_fn": th.nn.Tanh,
    }

    # Check for existing valid checkpoints to resume
    resumed_step = 0
    latest_valid_chk = None
    latest_valid_rb = None
    if checkpoint_dir.is_dir():
        cand_chks = sorted(list(checkpoint_dir.glob("sac_checkpoint_*_steps.zip")), key=lambda p: int(p.stem.split("_")[-2]) if p.stem.split("_")[-2].isdigit() else 0)
        for chk in reversed(cand_chks):
            parts = chk.stem.split("_")
            if len(parts) >= 3 and parts[-2].isdigit():
                s_num = int(parts[-2])
                rb = checkpoint_dir / f"sac_checkpoint_replay_buffer_{s_num}_steps.pkl"
                if rb.is_file() and rb.stat().st_size > 1000 and chk.stat().st_size > 1000:
                    latest_valid_chk = chk
                    latest_valid_rb = rb
                    resumed_step = s_num
                    break

    if latest_valid_chk is not None and 0 < resumed_step < total_timesteps:
        print(f"\n  [RECOVERY] Resuming SAC training from checkpoint: {latest_valid_chk.name} (step {resumed_step:,})")
        print(f"  [RECOVERY] Restoring replay buffer from: {latest_valid_rb.name}")
        custom_objects = {"AntiSymmetricSACPolicy": AntiSymmetricSACPolicy}
        model = SAC.load(str(latest_valid_chk), env=train_env, device=device, custom_objects=custom_objects)
        model.load_replay_buffer(str(latest_valid_rb))
        model.num_timesteps = resumed_step
        remaining_timesteps = total_timesteps - resumed_step
        reset_num_timesteps = False
        try:
            from src.agents.manifest_tracker import update_manifest
            update_manifest("SAC", seed, resumed_step, "RECOVERING", latest_valid_chk.name, latest_valid_rb.name, f"Resumed from {latest_valid_chk.name}")
        except Exception:
            pass
    else:
        model = SAC(
            policy=AntiSymmetricSACPolicy,
            env=train_env,
            learning_rate=lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=(train_freq, "step"),
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            seed=seed,
            device=device,
            verbose=0,
        )
        remaining_timesteps = total_timesteps
        reset_num_timesteps = True
        try:
            from src.agents.manifest_tracker import update_manifest
            update_manifest("SAC", seed, 0, "TRAINING", "None", "None", "Starting from scratch")
        except Exception:
            pass

    # Callbacks
    chk_step_freq = max(1, checkpoint_step // n_envs)
    checkpoint_callback = OffPolicyCheckpointCallback(
        save_freq=chk_step_freq,
        save_path=checkpoint_dir,
        name_prefix="sac_checkpoint",
        save_replay_buffer=True,
        algo="SAC",
        seed=seed,
    )
    telemetry_callback = SACTelemetryCallback(log_freq=max(1, 2000 // n_envs))

    t_start = time.time()
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[checkpoint_callback, telemetry_callback],
        reset_num_timesteps=reset_num_timesteps,
    )
    total_duration = time.time() - t_start
    avg_sps = model.num_timesteps / max(total_duration, 1e-4)

    # Save final model and replay buffer
    final_model_path = model_dir / f"sac_seed_{seed}_final_500k.zip"
    model.save(str(final_model_path))
    model.save_replay_buffer(str(model_dir / f"sac_seed_{seed}_final_replay_buffer.pkl"))

    # Health check
    eval_base_env = SpacecraftAttitudeEnv(include_integral=include_integral)
    health = evaluate_sac_health(model, eval_base_env)
    eval_base_env.close()

    print(f"\n[SAC SEED {seed} COMPLETE] Steps: {model.num_timesteps:,} in {total_duration:.1f}s ({avg_sps:.1f} SPS)")
    print(f"Health: u(+10°)={health['u_pos_10deg_nm']:+.4f} Nm | Direction Correct: {health['direction_correct']} | Symm Residual: {health['max_symmetry_discrepancy']:.1e} Nm | Alpha: {health['ent_coef']:.4f}")

    try:
        from src.agents.manifest_tracker import update_manifest
        update_manifest("SAC", seed, int(model.num_timesteps), "COMPLETED", final_model_path.name, f"sac_seed_{seed}_final_replay_buffer.pkl", "Completed full training")
    except Exception:
        pass

    summary = {
        "seed": seed,
        "algorithm": "SAC",
        "total_timesteps": int(model.num_timesteps),
        "total_duration_s": float(total_duration),
        "avg_sps": float(avg_sps),
        "final_model": str(final_model_path),
        "health": health,
        "telemetry": {
            "timesteps": telemetry_callback.logged_timesteps,
            "mean_theta_err_deg": telemetry_callback.logged_attitude_errors,
            "max_wheel_rpm": telemetry_callback.logged_wheel_speeds,
            "ent_coefs": telemetry_callback.logged_ent_coefs,
        },
    }

    summary_file = base_out / f"seed_{seed}_sac_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    train_env.close()
    return model, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4G SAC Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (42, 123, 2026)")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--checkpoint-step", type=int, default=50000, help="Checkpoint interval")
    args = parser.parse_args()

    train_single_seed_sac(seed=args.seed, total_timesteps=args.timesteps, checkpoint_step=args.checkpoint_step)
