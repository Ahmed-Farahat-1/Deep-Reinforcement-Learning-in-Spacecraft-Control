"""
Custom Gymnasium Reinforcement Learning Environment for Spacecraft Attitude Control.

Wraps the validated single-axis spacecraft attitude and reaction-wheel simulator
(Spacecraft1D) and configurable environmental disturbance models (BaseDisturbance)
into a standard Gymnasium environment compatible with Stable-Baselines3.

State & Observation:
    State: [theta, theta_dot, Omega] (+ optional integral error z)
    Observation: Normalized float32 vector in [-1.0, 1.0]^N (N=3 or N=4):
        theta_norm     = clip(theta / pi, -1.0, 1.0)
        theta_dot_norm = clip(theta_dot / 2.10, -1.0, 1.0)
        Omega_norm     = clip(Omega / Omega_max, -1.0, 1.0)
        z_norm         = clip(z / z_scale, -1.0, 1.0)  (when include_integral=True)

Action:
    action in [-1.0, 1.0] (1D Box, continuous)
    u_cmd = action[0] * u_max (N*m)

Redesigned Multi-Objective Reward Function (Phase 4D):
    R = r_theta + r_theta_dot + r_effort + r_omega + r_sat + r_bonus (+ r_z)
    - r_theta: Huber attitude penalty with linear non-vanishing gradient for large errors
    - r_theta_dot: Quadratic body rate damping penalty
    - r_effort: Command effort penalty (u_cmd^2)
    - r_omega: Momentum-aware penalty (only above 90% wheel speed; nominal speeds unpenalized)
    - r_sat: Actuator limit penalties and blocked outward torque penalty
    - r_bonus: Dense Gaussian precision pointing bonus with wide 1.0 deg basin
    - r_z: Integral error penalty to eliminate steady-state offset

All physical quantities strictly use SI units (rad, rad/s, N*m, kg*m^2, s).
"""

from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.simulation.spacecraft import Spacecraft1D
from src.simulation.disturbances import (
    BaseDisturbance,
    ZeroDisturbance,
    ConstantDisturbance,
    SinusoidalDisturbance,
    PulseDisturbance,
)


class SpacecraftAttitudeEnv(gym.Env):
    """
    Gymnasium environment for spacecraft single-axis attitude stabilization.

    Parameters
    ----------
    training_config_path : Optional[Union[str, Path]], optional
        Path to training configuration YAML. Defaults to config/training_config.yaml.
    physics_config_path : Optional[Union[str, Path]], optional
        Path to physics configuration YAML. Defaults to config/physics_config.yaml.
    include_integral : Optional[bool], optional
        Whether to include integrated attitude error as 4th observation dimension.
        Defaults to None (loads from training_config.yaml observation.include_integral).
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        training_config_path: Optional[Union[str, Path]] = None,
        physics_config_path: Optional[Union[str, Path]] = None,
        include_integral: Optional[bool] = None,
    ):
        super().__init__()

        # Resolve paths
        project_root = Path(__file__).resolve().parent.parent.parent
        self.training_config_path = (
            Path(training_config_path)
            if training_config_path is not None
            else project_root / "config" / "training_config.yaml"
        )
        self.physics_config_path = (
            Path(physics_config_path)
            if physics_config_path is not None
            else project_root / "config" / "physics_config.yaml"
        )

        # 1. Instantiate physics simulator
        self.simulator = Spacecraft1D(config_path=self.physics_config_path)

        # 2. Load environment & reward configurations
        self._load_env_config(include_integral=include_integral)

        # 3. Define Action Space: continuous 1D Box [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )

        # 4. Define Observation Space: normalized Box [-1.0, 1.0] (3D or 4D)
        obs_dim = 4 if self.include_integral else 3
        self.observation_space = spaces.Box(
            low=np.full((obs_dim,), -1.0, dtype=np.float32),
            high=np.full((obs_dim,), 1.0, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Disturbance model currently active in the episode
        self._current_disturbance: BaseDisturbance = ZeroDisturbance()
        self._step_count: int = 0
        self._current_scenario_id: str = "custom"
        self._integral_error: float = 0.0

    def _load_env_config(self, include_integral: Optional[bool] = None) -> None:
        """Load and parse environment configuration parameters."""
        if self.training_config_path.is_file():
            with open(self.training_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

        env_cfg = cfg.get("environment", {})

        # Episode limits
        self.time_step = float(env_cfg.get("time_step", self.simulator.dt))
        self.dt = self.time_step
        self.max_episode_steps = int(env_cfg.get("max_episode_steps", 3000))
        self.simulator.dt = self.time_step

        # Observation scaling
        obs_cfg = env_cfg.get("observation", {})
        self.theta_scale = float(obs_cfg.get("theta_scale", np.pi))
        self.theta_dot_scale = float(obs_cfg.get("theta_dot_scale", 2.10))
        self.Omega_scale = float(obs_cfg.get("Omega_scale", self.simulator.max_speed))
        self.clip_obs = bool(obs_cfg.get("clip_obs", True))

        # 4D Observation (Integral error state)
        if include_integral is not None:
            self.include_integral = bool(include_integral)
        else:
            self.include_integral = bool(obs_cfg.get("include_integral", False))
        self.z_scale = float(obs_cfg.get("z_scale", 1.0))
        self.z_max = float(obs_cfg.get("z_max", 1.0))
        self.anti_windup = bool(obs_cfg.get("anti_windup", True))

        # Action mapping
        act_cfg = env_cfg.get("action", {})
        self.max_torque = float(act_cfg.get("max_torque", self.simulator.max_torque))

        # Reward weights
        rw_cfg = env_cfg.get("reward_weights", {})
        self.reward_type = str(rw_cfg.get("reward_type", "huber"))
        self.w_theta = float(rw_cfg.get("w_theta", 1.0))
        self.w_theta_dot = float(rw_cfg.get("w_theta_dot", 0.1))
        self.w_effort = float(rw_cfg.get("w_effort", 0.001))
        self.w_omega = float(rw_cfg.get("w_omega", 0.005))
        self.w_torque_sat = float(rw_cfg.get("w_torque_sat", 0.05))
        self.w_wheel_sat = float(rw_cfg.get("w_wheel_sat", 1.00))
        self.w_blocked_torque = float(rw_cfg.get("w_blocked_torque", 0.20))
        self.w_bonus = float(rw_cfg.get("w_bonus", 0.50))
        self.w_z = float(rw_cfg.get("w_z", 0.05))
        self.huber_delta = float(rw_cfg.get("huber_delta", 0.05))
        self.omega_warn = float(rw_cfg.get("omega_warn", 0.90 * self.simulator.max_speed))

        # Reward normalization scaling
        rs_cfg = env_cfg.get("reward_scaling", {})
        self.theta_ref = float(rs_cfg.get("theta_ref", np.radians(30.0)))
        self.theta_dot_ref = float(rs_cfg.get("theta_dot_ref", 0.10))
        self.u_ref = float(rs_cfg.get("u_ref", self.max_torque))
        self.omega_ref = float(rs_cfg.get("omega_ref", self.simulator.max_speed))
        self.tolerance_theta = float(rs_cfg.get("tolerance_theta", np.radians(0.10)))
        self.tolerance_theta_dot = float(rs_cfg.get("tolerance_theta_dot", np.radians(0.01)))

        # Physical feasibility bound on initial angular momentum
        self.max_initial_momentum = float(env_cfg.get("max_initial_momentum", 10.0))

        # PPO parameters
        ppo_cfg = cfg.get("ppo", {})
        self.gamma = float(ppo_cfg.get("gamma", 0.998))

        # Initial condition configuration
        ic_cfg = env_cfg.get("initial_conditions", {})
        self.randomize_init = bool(ic_cfg.get("randomize", False))
        self.theta_range_rad = np.radians(ic_cfg.get("theta_range_deg", [-30.0, 30.0]))
        self.theta_dot_range_rad = np.radians(ic_cfg.get("theta_dot_range_deg", [-3.0, 3.0]))
        self.omega_range_rad = (
            np.array(ic_cfg.get("Omega_range_rpm", [-500.0, 500.0])) * 2.0 * np.pi / 60.0
        )
        self.near_sat_omega_range_rad = (
            np.array(ic_cfg.get("near_saturation_range_rpm", [3800.0, 4600.0])) * 2.0 * np.pi / 60.0
        )
        self.default_initial_state = np.array(
            ic_cfg.get("default_state", [np.radians(10.0), 0.0, 0.0]), dtype=np.float64
        )

        # Failure / termination limits
        term_cfg = env_cfg.get("termination", {})
        self.failure_theta = np.radians(float(term_cfg.get("failure_theta_deg", 90.0)))
        self.failure_theta_dot = np.radians(float(term_cfg.get("failure_theta_dot_deg", 120.0)))
        self.terminate_on_wheel_sat = bool(term_cfg.get("terminate_on_wheel_sat", False))

    def _get_obs(self) -> np.ndarray:
        """
        Construct normalized observation vector:
        - 3D: [theta_norm, theta_dot_norm, Omega_norm]
        - 4D: [theta_norm, theta_dot_norm, Omega_norm, z_norm] (if include_integral)

        Returns
        -------
        np.ndarray
            Normalized observation vector in [-1.0, 1.0]^N and dtype float32.
        """
        st = self.simulator.state
        theta, theta_dot, Omega = st[0], st[1], st[2]

        theta_norm = theta / self.theta_scale
        theta_dot_norm = theta_dot / self.theta_dot_scale
        Omega_norm = Omega / self.Omega_scale

        if self.clip_obs:
            theta_norm = np.clip(theta_norm, -1.0, 1.0)
            theta_dot_norm = np.clip(theta_dot_norm, -1.0, 1.0)
            Omega_norm = np.clip(Omega_norm, -1.0, 1.0)

        if self.include_integral:
            z_norm = self._integral_error / self.z_scale
            if self.clip_obs:
                z_norm = np.clip(z_norm, -1.0, 1.0)
            return np.array([theta_norm, theta_dot_norm, Omega_norm, z_norm], dtype=np.float32)

        return np.array([theta_norm, theta_dot_norm, Omega_norm], dtype=np.float32)

    def _compute_reward(
        self,
        theta: float,
        theta_dot: float,
        Omega: float,
        u_cmd: float,
        u_applied: float,
        is_torque_sat: bool,
        is_wheel_sat: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute multi-objective reward function (Phase 4D Huber & Momentum-Aware).

        Terms:
        1. r_theta: Huber penalty on attitude error (linear gradient for large errors)
        2. r_theta_dot: Quadratic body rate damping penalty
        3. r_effort: Command effort penalty (u_cmd^2)
        4. r_omega: Momentum-aware penalty (only above 90% wheel speed; nominal speeds unpenalized)
        5. r_sat: Actuator limit penalties and blocked outward torque penalty
        6. r_bonus: Smooth Gaussian precision pointing bonus with wide 1.0 deg basin
        7. r_z: Integral error penalty (if include_integral)
        """
        norm_theta = theta / self.theta_ref
        norm_theta_dot = theta_dot / self.theta_dot_ref
        norm_u = u_cmd / self.u_ref

        # 1. Attitude Error Penalty: Huber Formulation
        abs_norm_theta = abs(norm_theta)
        if abs_norm_theta <= self.huber_delta:
            huber_val = (norm_theta ** 2) / (2.0 * self.huber_delta)
        else:
            huber_val = abs_norm_theta - 0.5 * self.huber_delta
        r_theta = -self.w_theta * float(huber_val)

        # 2. Angular Rate Damping Penalty
        r_theta_dot = -self.w_theta_dot * float(norm_theta_dot ** 2)

        # 3. Control Effort Penalty
        r_effort = -self.w_effort * float(norm_u ** 2)

        # 4. Reaction Wheel Momentum Management
        # Nominal speeds (<= 90% Omega_max) receive zero penalty to allow slewing
        abs_omega = abs(Omega)
        if abs_omega > self.omega_warn:
            omega_excess = (abs_omega - self.omega_warn) / (self.simulator.max_speed - self.omega_warn)
            r_omega = -self.w_omega * float(omega_excess ** 2)
        else:
            r_omega = 0.0

        # 5. Saturation Penalties
        r_sat = 0.0
        if is_torque_sat:
            r_sat -= self.w_torque_sat
        if is_wheel_sat:
            r_sat -= self.w_wheel_sat

        # 6. Precision Pointing Bonus: Smooth Gaussian with wheel suppression
        gaussian_pointing = np.exp(
            -0.5 * ((theta / self.tolerance_theta) ** 2 + (theta_dot / self.tolerance_theta_dot) ** 2)
        )
        wheel_ratio = abs_omega / self.simulator.max_speed
        wheel_suppression = max(0.0, 1.0 - float(wheel_ratio ** 2))

        # Bonus is completely zeroed if wheel is saturated or at/above max speed
        if is_wheel_sat or wheel_ratio >= 1.0:
            r_bonus = 0.0
        else:
            r_bonus = float(self.w_bonus * gaussian_pointing * wheel_suppression)

        # 7. Integral Error Penalty
        if self.include_integral:
            norm_z = self._integral_error / self.z_scale
            r_z = -self.w_z * float(norm_z ** 2)
        else:
            r_z = 0.0

        total_reward = float(r_theta + r_theta_dot + r_effort + r_omega + r_sat + r_bonus + r_z)

        components = {
            "r_theta": r_theta,
            "r_theta_dot": r_theta_dot,
            "r_effort": r_effort,
            "r_omega": r_omega,
            "r_saturation": r_sat,
            "r_bonus": r_bonus,
            "total_reward": total_reward,
        }
        if self.include_integral:
            components["r_z"] = r_z

        return total_reward, components

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        seed : Optional[int]
            Random number generator seed for reproducibility.
        options : Optional[Dict[str, Any]]
            Optional reset options:
            - 'scenario': 'pos10deg', 'neg10deg', 'large30deg', 'constant', 'sinusoidal', 'pulse',
                          'heldout_h1', 'heldout_h2', 'heldout_h3', 'heldout_h4', 'heldout_h5'
            - 'initial_state': [theta_0, theta_dot_0, Omega_0]
            - 'initial_z': float (initial integral state)
            - 'randomize': bool (override randomization)
            - 'mode': 'near_saturation'
            - 'disturbance': BaseDisturbance instance

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            (initial_observation, info_dictionary)
        """
        super().reset(seed=seed)

        opts = options or {}
        scenario_id = opts.get("scenario", None)
        randomize = opts.get("randomize", self.randomize_init)
        mode = opts.get("mode", "")
        is_near_sat = mode == "near_saturation" or bool(opts.get("near_saturation", False))

        # Determine disturbance model and initial state
        if scenario_id is not None:
            self._current_scenario_id = str(scenario_id)
            init_state, dist_model = self._configure_scenario(scenario_id)
        elif "disturbance" in opts:
            dist_model = opts["disturbance"]
            self._current_scenario_id = "custom_disturbance"
            init_state = opts.get("initial_state", self.default_initial_state)
        elif "initial_state" in opts:
            init_state = np.array(opts["initial_state"], dtype=np.float64)
            dist_model = ZeroDisturbance()
            self._current_scenario_id = "custom_initial_state"
        elif is_near_sat:
            # Near-saturation stress initialization (3800 to 4600 RPM)
            th_0 = float(self.np_random.uniform(self.theta_range_rad[0], self.theta_range_rad[1]))
            th_dot_0 = float(self.np_random.uniform(self.theta_dot_range_rad[0], self.theta_dot_range_rad[1]))
            om_mag = float(self.np_random.uniform(self.near_sat_omega_range_rad[0], self.near_sat_omega_range_rad[1]))
            om_sign = 1.0 if self.np_random.uniform(0.0, 1.0) >= 0.5 else -1.0
            init_state = np.array([th_0, th_dot_0, om_mag * om_sign], dtype=np.float64)
            dist_model = ZeroDisturbance()
            self._current_scenario_id = "near_saturation_stress"
        elif randomize:
            # Random initial state for nominal training exploration (Omega in [-500, 500] RPM)
            th_0 = float(self.np_random.uniform(self.theta_range_rad[0], self.theta_range_rad[1]))
            th_dot_0 = float(self.np_random.uniform(self.theta_dot_range_rad[0], self.theta_dot_range_rad[1]))
            om_0 = float(self.np_random.uniform(self.omega_range_rad[0], self.omega_range_rad[1]))
            init_state = np.array([th_0, th_dot_0, om_0], dtype=np.float64)
            dist_model = ZeroDisturbance()
            self._current_scenario_id = "random_training"
        else:
            init_state = self.default_initial_state
            dist_model = ZeroDisturbance()
            self._current_scenario_id = "default"

        self._current_disturbance = dist_model
        self.simulator.reset(initial_state=init_state)
        self._step_count = 0
        self._integral_error = float(opts.get("initial_z", 0.0))

        obs = self._get_obs()
        info = self._get_info(u_cmd=0.0, u_applied=0.0, is_torque_sat=False, is_wheel_sat=False)
        return obs, info

    def _configure_scenario(self, scenario_id: str) -> Tuple[np.ndarray, BaseDisturbance]:
        """Configure canonical benchmark scenarios plus Phase 4D held-out test scenarios."""
        sid = scenario_id.lower()
        if "pos10" in sid or sid == "scenario_1":
            return np.array([np.radians(10.0), 0.0, 0.0]), ZeroDisturbance()
        elif "neg10" in sid or sid == "scenario_2":
            return np.array([np.radians(-10.0), 0.0, 0.0]), ZeroDisturbance()
        elif "pos30" in sid or "large30" in sid or sid == "scenario_3":
            return np.array([np.radians(30.0), 0.0, 0.0]), ZeroDisturbance()
        elif "neg30" in sid:
            return np.array([np.radians(-30.0), 0.0, 0.0]), ZeroDisturbance()
        elif "constant_neg" in sid:
            return np.array([np.radians(-10.0), 0.0, 0.0]), ConstantDisturbance(torque_value=-0.02)
        elif "constant" in sid or "scenario_4" in sid:
            return np.array([np.radians(10.0), 0.0, 0.0]), ConstantDisturbance(torque_value=0.02)
        elif "sinusoidal" in sid or "scenario_5" in sid:
            return np.array([np.radians(10.0), 0.0, 0.0]), SinusoidalDisturbance(amplitude=0.02, frequency=0.1)
        elif "pulse_neg" in sid:
            return np.array([np.radians(-10.0), 0.0, 0.0]), PulseDisturbance(magnitude=-0.05, t_start=10.0, t_end=12.0)
        elif "pulse" in sid or "scenario_6" in sid:
            return np.array([np.radians(10.0), 0.0, 0.0]), PulseDisturbance(magnitude=0.05, t_start=10.0, t_end=12.0)
        # Phase 4D Held-Out Scenarios (H1 to H5)
        elif "heldout_h1" in sid or "h1" in sid:
            # H1: Unseen positive step (+17.5 deg)
            return np.array([np.radians(17.5), 0.0, 0.0]), ZeroDisturbance()
        elif "heldout_h2" in sid or "h2" in sid:
            # H2: Unseen negative step (-22.5 deg)
            return np.array([np.radians(-22.5), 0.0, 0.0]), ZeroDisturbance()
        elif "heldout_h3" in sid or "h3" in sid:
            # H3: Mixed state (+15.0 deg, -1.5 deg/s, +500 RPM wheel)
            return np.array([np.radians(15.0), np.radians(-1.5), 500.0 * 2.0 * np.pi / 60.0]), ZeroDisturbance()
        elif "heldout_h4" in sid or "h4" in sid:
            # H4: High-frequency sinusoidal disturbance (0.25 Hz, 0.03 N*m, +10 deg)
            return np.array([np.radians(10.0), 0.0, 0.0]), SinusoidalDisturbance(amplitude=0.03, frequency=0.25)
        elif "heldout_h5" in sid or "h5" in sid:
            # H5: Strong constant disturbance (-0.035 N*m, -10 deg)
            return np.array([np.radians(-10.0), 0.0, 0.0]), ConstantDisturbance(torque_value=-0.035)
        else:
            return self.default_initial_state, ZeroDisturbance()

    def step(
        self, action: Union[np.ndarray, float, list]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one control step in the environment.

        Parameters
        ----------
        action : Union[np.ndarray, float, list]
            Continuous control action in [-1.0, 1.0].

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            (observation, reward, terminated, truncated, info)
        """
        # 1. Process action
        act_arr = np.asarray(action, dtype=np.float32).flatten()
        if len(act_arr) != 1:
            raise ValueError(f"Action must be 1D array of shape (1,), got {act_arr.shape}")

        act_clipped = float(np.clip(act_arr[0], -1.0, 1.0))

        # 2. Map normalized action to requested motor torque command
        u_cmd = act_clipped * self.max_torque

        # Pre-step state for anti-windup integration
        st_pre = self.simulator.state
        theta_pre = st_pre[0]
        Omega_pre = st_pre[2]

        # 3. Advance physics simulation via RK4
        # Simulator strictly enforces motor torque & rotor speed saturation
        _, sim_info = self.simulator.step(u_cmd=u_cmd, disturbance=self._current_disturbance)

        self._step_count += 1
        st = self.simulator.state
        theta, theta_dot, Omega = st[0], st[1], st[2]
        u_applied = sim_info["u_applied"]
        is_torque_sat = sim_info["is_torque_saturated"]
        is_wheel_sat = sim_info["is_wheel_saturated"]

        # 4. Anti-Windup Conditional Integral State Update
        if self.include_integral:
            # Check conditional anti-windup freezing:
            # Freeze integration if motor torque is at/exceeds capacity and error pushes further into saturation,
            # or if corrective actuator command is blocked by wheel speed saturation.
            is_at_torque_limit = abs(u_cmd) >= (self.max_torque - 1e-5)
            outward_torque_push = (theta_pre * u_cmd > 0.0) and is_at_torque_limit
            wheel_blocked_push = bool(is_wheel_sat)
            freeze_integral = self.anti_windup and (outward_torque_push or wheel_blocked_push)

            if not freeze_integral:
                # Trapezoidal integration: (theta_pre + theta) / 2 * dt
                self._integral_error += 0.5 * (theta_pre + theta) * self.dt
                # Clamping anti-windup
                self._integral_error = float(np.clip(self._integral_error, -self.z_max, self.z_max))

        # 5. Compute reward
        reward, reward_components = self._compute_reward(
            theta=theta,
            theta_dot=theta_dot,
            Omega=Omega,
            u_cmd=u_cmd,
            u_applied=u_applied,
            is_torque_sat=is_torque_sat,
            is_wheel_sat=is_wheel_sat,
        )

        # 6. Check Termination Conditions
        # Physical tumble or rate runaway constitutes terminal mission failure
        is_tumbled = abs(theta) > self.failure_theta
        is_rate_runaway = abs(theta_dot) > self.failure_theta_dot

        # Wheel saturation alone does NOT terminate episode (wheel can be despun)
        wheel_sat_term = self.terminate_on_wheel_sat and is_wheel_sat

        terminated = bool(is_tumbled or is_rate_runaway or wheel_sat_term)

        # Truncation: reached max episode time limit
        truncated = bool(self._step_count >= self.max_episode_steps)

        # 7. Build observation and info dictionary
        obs = self._get_obs()
        info = self._get_info(
            u_cmd=u_cmd,
            u_applied=u_applied,
            is_torque_sat=is_torque_sat,
            is_wheel_sat=is_wheel_sat,
            reward_components=reward_components,
        )

        return obs, float(reward), terminated, truncated, info

    def _get_info(
        self,
        u_cmd: float,
        u_applied: float,
        is_torque_sat: bool,
        is_wheel_sat: bool,
        reward_components: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compile comprehensive telemetry and diagnostic dictionary."""
        st = self.simulator.state
        theta, theta_dot, Omega = st[0], st[1], st[2]
        t = self.simulator.time

        # Disturbance evaluation
        dist_torque = self._current_disturbance.torque(t)

        is_success = bool(
            (abs(theta) <= self.tolerance_theta) and (abs(theta_dot) <= self.tolerance_theta_dot)
        )

        info = {
            "time": float(t),
            "step": self._step_count,
            "scenario_id": self._current_scenario_id,
            "theta_rad": float(theta),
            "theta_deg": float(np.degrees(theta)),
            "theta_dot_rad_s": float(theta_dot),
            "theta_dot_deg_s": float(np.degrees(theta_dot)),
            "wheel_speed_rad_s": float(Omega),
            "wheel_speed_rpm": float(Omega * 60.0 / (2.0 * np.pi)),
            "u_cmd": float(u_cmd),
            "u_applied": float(u_applied),
            "disturbance_torque": float(dist_torque),
            "total_angular_momentum": float(self.simulator.total_angular_momentum()),
            "is_torque_saturated": bool(is_torque_sat),
            "is_wheel_saturated": bool(is_wheel_sat),
            "is_success": is_success,
        }

        if self.include_integral:
            info["integral_error_rad_s"] = float(self._integral_error)

        if reward_components is not None:
            info["reward_components"] = reward_components

        return info

    def close(self) -> None:
        """Clean up environment resources."""
        pass
