"""
Proximal Policy Optimization (PPO) Training Pipeline for Spacecraft Attitude Stabilization.

Phase 4A: Implementation and 100,000-timestep pilot training.
Validates the RL training pipeline, environment coupling, TensorBoard telemetry,
checkpointing, and policy convergence behavior.
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml
import numpy as np
import gymnasium as gym

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

from src.envs.spacecraft_env import SpacecraftAttitudeEnv


class TrainingScenarioWrapper(gym.Wrapper):
    """
    Gymnasium environment wrapper for training that samples initial states and disturbances
    according to a specified probability distribution across canonical and randomized regimes.
    
    Ensures the PPO policy is trained symmetrically against:
    - Nominal attitude stabilization
    - Symmetrical positive and negative step errors (+10 deg, -10 deg, +30 deg, -30 deg)
    - Randomized initial states (|H0| < 3.14 N*m*s)
    - Symmetrical environmental disturbances (Constant +/- Md, Sinusoidal, Pulse +/- Md)
    - Near-saturation actuator boundary conditions (within physical momentum constraints)
    """

    def __init__(self, env: gym.Env, scenario_distribution: Optional[Dict[str, float]] = None):
        super().__init__(env)
        self.distribution = scenario_distribution or {
            "random_nominal": 0.40,
            "pos10deg": 0.075,
            "neg10deg": 0.075,
            "pos30deg": 0.05,
            "neg30deg": 0.05,
            "constant_pos": 0.05,
            "constant_neg": 0.05,
            "sinusoidal_dist": 0.10,
            "pulse_pos": 0.05,
            "pulse_neg": 0.05,
            "near_saturation": 0.05,
        }
        self.scenario_names = list(self.distribution.keys())
        self.scenario_probs = np.array(list(self.distribution.values()), dtype=np.float64)
        self.scenario_probs /= np.sum(self.scenario_probs)  # Normalize to sum to 1.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sample training scenario if options are not explicitly provided."""
        if options is not None:
            return self.env.reset(seed=seed, options=options)

        # Sample scenario according to configured distribution
        chosen_scenario = str(self.env.np_random.choice(self.scenario_names, p=self.scenario_probs))

        if chosen_scenario == "random_nominal":
            opts = {"randomize": True}
        elif chosen_scenario in ["pos10deg", "pos10"]:
            opts = {"scenario": "pos10deg"}
        elif chosen_scenario in ["neg10deg", "neg10"]:
            opts = {"scenario": "neg10deg"}
        elif chosen_scenario in ["pos30deg", "large30deg", "large30", "pos30"]:
            opts = {"scenario": "pos30deg"}
        elif chosen_scenario in ["neg30deg", "neg30"]:
            opts = {"scenario": "neg30deg"}
        elif chosen_scenario in ["constant_pos", "constant_dist", "constant"]:
            opts = {"scenario": "constant"}
        elif chosen_scenario in ["constant_neg"]:
            opts = {"scenario": "constant_neg"}
        elif chosen_scenario in ["sinusoidal_dist", "sinusoidal"]:
            opts = {"scenario": "sinusoidal"}
        elif chosen_scenario in ["pulse_pos", "pulse_dist", "pulse"]:
            opts = {"scenario": "pulse"}
        elif chosen_scenario in ["pulse_neg"]:
            opts = {"scenario": "pulse_neg"}
        elif chosen_scenario == "near_saturation":
            # Physically feasible near-saturation test with |H0| <= 10.0 N*m*s
            # Initial wheel speed between 1500 and 1850 RPM (H_wheel ~ 7.85 to 9.68 N*m*s)
            th_0 = float(self.env.np_random.uniform(-np.radians(15.0), np.radians(15.0)))
            th_dot_0 = float(self.env.np_random.uniform(-np.radians(1.0), np.radians(1.0)))
            om_sign = -1.0 if th_0 >= 0 else 1.0
            om_val = float(self.env.np_random.uniform(1500.0, 1850.0) * 2.0 * np.pi / 60.0) * om_sign
            H_tot = 10.05 * th_dot_0 + 0.05 * om_val
            if abs(H_tot) > 9.95:
                max_h_wheel = max(0.0, 9.95 - abs(10.05 * th_dot_0))
                om_val = float(np.sign(om_val) * (max_h_wheel / 0.05))
            opts = {"initial_state": [th_0, th_dot_0, om_val]}
        else:
            opts = {"randomize": True}

        return self.env.reset(seed=seed, options=opts)


class EvaluationScenarioWrapper(gym.Wrapper):
    """
    Gymnasium environment wrapper for periodic evaluation that cycles deterministically
    through balanced canonical and disturbance scenarios.
    
    Ensures EvalCallback measures bidirectional symmetry, disturbance resistance,
    and saturation respect, avoiding bias toward a single initial condition.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.scenarios = [
            {"scenario": "pos10deg"},
            {"scenario": "neg10deg"},
            {"scenario": "pos30deg"},
            {"scenario": "neg30deg"},
            {"scenario": "constant"},
            {"scenario": "constant_neg"},
            {"scenario": "sinusoidal"},
            {"scenario": "pulse"},
        ]
        self._eval_idx = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Cycle deterministically through balanced evaluation scenarios."""
        if options is not None:
            return self.env.reset(seed=seed, options=options)
        opts = self.scenarios[self._eval_idx % len(self.scenarios)]
        self._eval_idx += 1
        return self.env.reset(seed=seed, options=opts)


class SpacecraftTelemetryCallback(BaseCallback):
    """
    Custom Stable-Baselines3 Callback for logging aerospace GNC diagnostics to TensorBoard
    and dynamically annealing entropy regularization.
    """

    def __init__(
        self,
        log_freq: int = 1000,
        ent_start: float = 0.005,
        ent_end: float = 0.0005,
        total_timesteps: int = 500000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.total_timesteps = total_timesteps
        self.step_count = 0

        # Ring buffers for rolling diagnostic calculation
        self.buf_theta_err: List[float] = []
        self.buf_theta_dot: List[float] = []
        self.buf_wheel_speed: List[float] = []
        self.buf_torque_sat: List[float] = []
        self.buf_wheel_sat: List[float] = []
        self.buf_success: List[float] = []

        # History tracking for evaluation export
        self.logged_timesteps: List[int] = []
        self.logged_rewards: List[float] = []
        self.logged_attitude_errors: List[float] = []
        self.logged_wheel_speeds: List[float] = []

    def _on_rollout_end(self) -> None:
        """Anneal entropy coefficient after each rollout."""
        if self.total_timesteps > 0:
            progress = max(0.0, 1.0 - (self.num_timesteps / self.total_timesteps))
            new_ent = float(self.ent_end + progress * (self.ent_start - self.ent_end))
            self.model.ent_coef = new_ent
            self.logger.record("train/ent_coef_annealed", new_ent)

    def _on_step(self) -> bool:
        self.step_count += 1
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        # Check for NaN / Inf values in rewards
        for r in rewards:
            if not np.isfinite(r):
                print(f"[ERROR] Non-finite reward detected at timestep {self.num_timesteps}: {r}!")
                return False  # Abort training safely

        # Accumulate telemetry
        for info in infos:
            th_err = abs(info.get("theta_deg", 0.0))
            th_dot = abs(info.get("theta_dot_deg_s", 0.0))
            om_rpm = abs(info.get("wheel_speed_rpm", 0.0))
            t_sat = 1.0 if info.get("is_torque_saturated", False) else 0.0
            w_sat = 1.0 if info.get("is_wheel_saturated", False) else 0.0
            succ = 1.0 if info.get("is_success", False) else 0.0

            self.buf_theta_err.append(th_err)
            self.buf_theta_dot.append(th_dot)
            self.buf_wheel_speed.append(om_rpm)
            self.buf_torque_sat.append(t_sat)
            self.buf_wheel_sat.append(w_sat)
            self.buf_success.append(succ)

        # Periodic logging to TensorBoard
        if self.num_timesteps % self.log_freq == 0 and len(self.buf_theta_err) > 0:
            mean_theta = float(np.mean(self.buf_theta_err))
            mean_theta_dot = float(np.mean(self.buf_theta_dot))
            max_wheel = float(np.max(self.buf_wheel_speed))
            torque_sat_freq = float(np.mean(self.buf_torque_sat))
            wheel_sat_freq = float(np.mean(self.buf_wheel_sat))
            success_rate = float(np.mean(self.buf_success))

            # Record custom metrics to TensorBoard logger
            self.logger.record("spacecraft/mean_abs_attitude_error_deg", mean_theta)
            self.logger.record("spacecraft/mean_abs_angular_velocity_deg_s", mean_theta_dot)
            self.logger.record("spacecraft/max_wheel_speed_rpm", max_wheel)
            self.logger.record("spacecraft/torque_saturation_freq", torque_sat_freq)
            self.logger.record("spacecraft/wheel_saturation_freq", wheel_sat_freq)
            self.logger.record("spacecraft/fine_pointing_success_rate", success_rate)

            # Keep track for summary plotting
            self.logged_timesteps.append(int(self.num_timesteps))
            self.logged_attitude_errors.append(mean_theta)
            self.logged_wheel_speeds.append(max_wheel)

            # Clear buffers
            self.buf_theta_err.clear()
            self.buf_theta_dot.clear()
            self.buf_wheel_speed.clear()
            self.buf_torque_sat.clear()
            self.buf_wheel_sat.clear()
            self.buf_success.clear()

        return True


def load_training_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "training_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def make_vectorized_env(
    n_envs: int = 4,
    scenario_distribution: Optional[Dict[str, float]] = None,
    seed: int = 42,
    include_integral: bool = True,
) -> DummyVecEnv:
    """Create vectorized Gymnasium training environments with scenario sampling wrapper."""
    def _make(idx: int):
        def _thunk():
            base_env = SpacecraftAttitudeEnv(include_integral=include_integral)
            wrapped_env = TrainingScenarioWrapper(base_env, scenario_distribution)
            monitored_env = Monitor(wrapped_env)
            monitored_env.reset(seed=seed + idx)
            return monitored_env
        return _thunk

    return DummyVecEnv([_make(i) for i in range(n_envs)])


def linear_entropy_schedule(initial_value: float = 0.005, final_value: float = 0.0005):
    """Linearly decays entropy coefficient from initial_value to final_value based on progress_remaining."""
    def _schedule(progress_remaining: float) -> float:
        return float(final_value + progress_remaining * (initial_value - final_value))
    return _schedule


def evaluate_policy_health(model: PPO, env: SpacecraftAttitudeEnv) -> Dict[str, Any]:
    """Diagnostic health check evaluating policy actions and directional symmetry."""
    dim = env.observation_space.shape[0]
    obs_p10 = np.zeros(dim, dtype=np.float32)
    obs_p10[0] = np.radians(10.0) / env.theta_scale

    obs_n10 = np.zeros(dim, dtype=np.float32)
    obs_n10[0] = np.radians(-10.0) / env.theta_scale

    obs_p30 = np.zeros(dim, dtype=np.float32)
    obs_p30[0] = np.radians(30.0) / env.theta_scale

    obs_n30 = np.zeros(dim, dtype=np.float32)
    obs_n30[0] = np.radians(-30.0) / env.theta_scale

    obs_0 = np.zeros(dim, dtype=np.float32)

    u_p10 = float(model.predict(obs_p10, deterministic=True)[0][0] * env.max_torque)
    u_n10 = float(model.predict(obs_n10, deterministic=True)[0][0] * env.max_torque)
    u_p30 = float(model.predict(obs_p30, deterministic=True)[0][0] * env.max_torque)
    u_n30 = float(model.predict(obs_n30, deterministic=True)[0][0] * env.max_torque)
    u_0 = float(model.predict(obs_0, deterministic=True)[0][0] * env.max_torque)

    # Extract policy std
    import torch
    dist = model.policy.get_distribution(torch.as_tensor(obs_0).unsqueeze(0).to(model.device))
    std = float(dist.distribution.stddev.item())

    # Symmetry score: ratio of magnitude difference to sum
    mag_diff = abs(abs(u_p10) - abs(u_n10))
    mag_sum = abs(u_p10) + abs(u_n10) + 1e-6
    symmetry_score = 1.0 - (mag_diff / mag_sum)

    # Direction correctness
    dir_correct = (u_p10 > 0) and (u_n10 < 0)
    near_zero = (abs(u_p10) < 0.001) and (abs(u_n10) < 0.001)

    return {
        "u_pos10": u_p10,
        "u_neg10": u_n10,
        "u_pos30": u_p30,
        "u_neg30": u_n30,
        "u_zero": u_0,
        "std": std,
        "symmetry_score": symmetry_score,
        "direction_correct": dir_correct,
        "near_zero_collapse": near_zero,
    }


def run_training_pipeline(
    config_path: Optional[Path] = None,
    total_timesteps: Optional[int] = None,
    stage_steps: int = 100000,
) -> Tuple[PPO, Dict[str, Any]]:
    """
    Execute full PPO training pipeline with staged checkpoints and evaluations.

    Parameters
    ----------
    config_path : Optional[Path]
        Path to training configuration YAML.
    total_timesteps : Optional[int]
        Override total timesteps (default: 500,000 for Phase 4B).
    stage_steps : int
        Timesteps per training stage (default: 100,000).

    Returns
    -------
    Tuple[PPO, Dict[str, Any]]
        Trained model instance and training summary dictionary.
    """
    cfg = load_training_config(config_path)
    ppo_cfg = cfg.get("ppo", {})
    logging_cfg = cfg.get("logging", {})
    scenario_dist = cfg.get("training_scenarios", {}).get("distribution", None)

    # 1. Hyperparameters resolution
    lr = float(ppo_cfg.get("learning_rate", 3e-4))
    n_steps = int(ppo_cfg.get("n_steps", 2048))
    batch_size = int(ppo_cfg.get("batch_size", 256))
    n_epochs = int(ppo_cfg.get("n_epochs", 10))
    gamma = float(ppo_cfg.get("gamma", 0.998))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.2))
    ent_start = float(ppo_cfg.get("ent_coef", 0.005))
    ent_end = float(ppo_cfg.get("ent_coef_end", 0.0005))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
    seed = int(ppo_cfg.get("seed", 42))
    n_envs = int(ppo_cfg.get("n_envs", 4))
    device = str(ppo_cfg.get("device", "cpu"))

    if total_timesteps is None:
        total_timesteps = int(ppo_cfg.get("total_timesteps", 500000))

    # Continuous entropy decay schedule
    ent_schedule = linear_entropy_schedule(ent_start, ent_end)

    # Architecture definition
    net_arch = ppo_cfg.get("net_arch", {"pi": [64, 64], "vf": [64, 64]})
    policy_kwargs = {
        "net_arch": {
            "pi": net_arch.get("pi", [64, 64]),
            "vf": net_arch.get("vf", [64, 64]),
        }
    }

    print("\n" + "=" * 80)
    print("SPACECRAFT ATTITUDE CONTROL: PPO FULL TRAINING (PHASE 4B)")
    print("=" * 80)
    print(f"Total Target Timesteps: {total_timesteps:,} (Staged in {total_timesteps // stage_steps} stages of {stage_steps:,})")
    print(f"Parallel Envs:          {n_envs}")
    print(f"Rollout Steps (n):      {n_steps} per env ({n_steps * n_envs} per rollout)")
    print(f"Minibatch Size:         {batch_size}")
    print(f"Optimization Epochs:    {n_epochs}")
    print(f"Learning Rate:          {lr}")
    print(f"Discount (gamma):       {gamma} (Horizon ~5.0s, retains 18.4% weight at Ts=8.44s)")
    print(f"Entropy Schedule:       {ent_start} -> {ent_end} (Linear Annealing)")
    print(f"Reward Effort Weight:   {cfg.get('environment', {}).get('reward_weights', {}).get('w_effort', 0.001)}")
    print(f"Policy Net Arch:        Actor {policy_kwargs['net_arch']['pi']} | Critic {policy_kwargs['net_arch']['vf']}")
    print(f"Hardware Device:        {device}")
    print(f"Random Seed:            {seed}")
    print("=" * 80 + "\n")

    # 2. Setup Directories (Phase 4B isolated from Phase 4A)
    tensorboard_dir = PROJECT_ROOT / logging_cfg.get("tensorboard_log", "logs/ppo/phase_4b")
    checkpoint_dir = PROJECT_ROOT / logging_cfg.get("save_path", "outputs/ppo_training/phase_4b/checkpoints")
    best_model_dir = PROJECT_ROOT / logging_cfg.get("best_model_path", "outputs/ppo_training/phase_4b/best_model")
    eval_log_dir = PROJECT_ROOT / logging_cfg.get("eval_log_path", "outputs/ppo_training/phase_4b/evaluations")
    final_model_dir = PROJECT_ROOT / "models" / "ppo" / "phase_4b"
    summary_dir = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4b"
    figures_dir = summary_dir / "figures"

    for d in [tensorboard_dir, checkpoint_dir, best_model_dir, eval_log_dir, final_model_dir, summary_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save training configuration snapshot
    snapshot_path = summary_dir / "training_config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    print(f"[SETUP] Training configuration snapshot saved to: {snapshot_path}")

    # 3. Create Vectorized Environments
    policy_name = ppo_cfg.get("policy", "AntiSymmetricActorCriticPolicy")
    include_integral = bool(ppo_cfg.get("include_integral", True))

    if policy_name == "AntiSymmetricActorCriticPolicy":
        from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
        policy_cls = AntiSymmetricActorCriticPolicy
        print("[SETUP] Using AntiSymmetricActorCriticPolicy (enforces odd symmetry by construction)")
    else:
        policy_cls = "MlpPolicy"
        print("[SETUP] Using standard MlpPolicy")

    print(f"[SETUP] Creating vectorized training environments (include_integral={include_integral})...")
    train_env = make_vectorized_env(
        n_envs=n_envs,
        scenario_distribution=scenario_dist,
        seed=seed,
        include_integral=include_integral,
    )

    print("[SETUP] Creating deterministic symmetric evaluation environment...")
    eval_base_env = SpacecraftAttitudeEnv(include_integral=include_integral)
    eval_wrapped = EvaluationScenarioWrapper(eval_base_env)
    eval_env = DummyVecEnv([lambda: Monitor(eval_wrapped)])

    # 4. Instantiate Callbacks
    checkpoint_freq = int(logging_cfg.get("checkpoint_freq", 25000))
    eval_freq = int(logging_cfg.get("eval_freq", 10000))

    step_checkpoint_freq = max(1, checkpoint_freq // n_envs)
    step_eval_freq = max(1, eval_freq // n_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=step_checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_spacecraft",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=step_eval_freq,
        n_eval_episodes=8,  # Evaluates all 8 symmetric scenarios
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

    # 5. Initialize PPO Agent
    print(f"[SETUP] Initializing PPO agent from scratch ({policy_cls})...")
    model = PPO(
        policy=policy_cls,
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

    # 6. Execute Staged Training
    num_stages = max(1, total_timesteps // stage_steps)
    print(f"\n[TRAINING] Commencing Phase 4B Staged Full Training ({num_stages} stages)...")
    t_start_total = time.perf_counter()
    stage_logs = []

    for stage_idx in range(1, num_stages + 1):
        target_cum = stage_idx * stage_steps
        print("\n" + "-" * 75)
        print(f"STAGE {stage_idx}/{num_stages} — Steps {model.num_timesteps:,} -> {target_cum:,}")
        print("-" * 75)
        t_stage_start = time.perf_counter()

        try:
            model.learn(
                total_timesteps=stage_steps,
                callback=callback_list,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Training aborted at Stage {stage_idx} due to exception: {e}")
            train_env.close()
            eval_env.close()
            raise e

        t_stage_end = time.perf_counter()
        stage_dur = t_stage_end - t_stage_start
        cum_steps = model.num_timesteps
        stage_sps = stage_steps / max(stage_dur, 1e-4)

        # Save stage checkpoint model
        stage_model_file = final_model_dir / f"ppo_stage_{stage_idx}_{cum_steps // 1000}k.zip"
        model.save(str(stage_model_file))
        print(f"[STAGE {stage_idx} SAVED] Checkpoint: {stage_model_file.name}")

        # Diagnostic policy evaluation
        health = evaluate_policy_health(model, eval_base_env)
        print(f"[STAGE {stage_idx} DIAGNOSTIC] Elapsed: {stage_dur:.1f}s ({stage_sps:.1f} SPS) | Std: {health['std']:.4f}")
        print(f"  Torque u(+10°): {health['u_pos10']:+.6f} N·m (Sign correct: {health['u_pos10'] > 0})")
        print(f"  Torque u(-10°): {health['u_neg10']:+.6f} N·m (Sign correct: {health['u_neg10'] < 0})")
        print(f"  Torque u(+30°): {health['u_pos30']:+.6f} N·m | u(-30°): {health['u_neg30']:+.6f} N·m")
        print(f"  Direction Correct: {health['direction_correct']} | Symmetry Score: {health['symmetry_score']:.4f}")

        stage_logs.append({
            "stage": stage_idx,
            "cumulative_timesteps": cum_steps,
            "stage_duration_seconds": stage_dur,
            "stage_steps_per_second": stage_sps,
            "health": health,
        })

    t_end_total = time.perf_counter()
    total_duration = t_end_total - t_start_total
    avg_sps = total_timesteps / max(total_duration, 1e-4)

    print("\n" + "=" * 80)
    print("PHASE 4B FULL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Elapsed Time: {total_duration:.2f} s ({total_duration / 60.0:.2f} min)")
    print(f"Average Throughput: {avg_sps:.1f} steps/second (SPS)")
    print(f"Completed Steps:    {model.num_timesteps:,}")

    # 7. Save Final Models and Metadata
    final_500k_path = final_model_dir / "ppo_final_500k.zip"
    summary_final_path = summary_dir / "final_phase4b_model.zip"

    model.save(str(final_500k_path))
    model.save(str(summary_final_path))
    print(f"[SAVE] Final 500k model saved to: {final_500k_path}")

    # Read evaluation results if available
    eval_npz_path = eval_log_dir / "evaluations.npz"
    eval_data = {}
    if eval_npz_path.is_file():
        npz = np.load(str(eval_npz_path))
        eval_data = {
            "eval_timesteps": npz["timesteps"].tolist(),
            "eval_mean_rewards": [float(np.mean(r)) for r in npz["results"]],
            "eval_std_rewards": [float(np.std(r)) for r in npz["results"]],
        }

    # Summary record
    summary = {
        "status": "COMPLETED",
        "phase": "4B",
        "total_timesteps": int(total_timesteps),
        "total_elapsed_seconds": float(total_duration),
        "steps_per_second": float(avg_sps),
        "device": device,
        "n_envs": n_envs,
        "final_model_path": str(final_500k_path),
        "best_model_path": str(best_model_dir / "best_model.zip"),
        "stage_logs": stage_logs,
        "evaluations": eval_data,
        "telemetry_history": {
            "timesteps": telemetry_callback.logged_timesteps,
            "mean_attitude_errors_deg": telemetry_callback.logged_attitude_errors,
            "max_wheel_speeds_rpm": telemetry_callback.logged_wheel_speeds,
        },
    }

    summary_file = summary_dir / "phase_4b_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Phase 4B training summary exported to: {summary_file}")

    # Close environments
    train_env.close()
    eval_env.close()

    # 8. Trigger Full Deterministic Evaluation
    print("\n[EVALUATION] Commencing full deterministic evaluation of best Phase 4B model...")
    from src.agents.evaluate_ppo import run_full_evaluation
    best_eval_model_path = best_model_dir / "best_model.zip"
    target_eval_model = best_eval_model_path if best_eval_model_path.is_file() else final_500k_path

    run_full_evaluation(
        model_path=target_eval_model,
        output_dir=eval_log_dir,
        figures_dir=figures_dir,
    )

    return model, summary


if __name__ == "__main__":
    run_training_pipeline()
