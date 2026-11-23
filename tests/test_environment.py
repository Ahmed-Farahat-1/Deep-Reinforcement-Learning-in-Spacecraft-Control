"""
Comprehensive Validation Tests for Spacecraft Attitude Gymnasium RL Environment (Phase 3).

Validates:
1. Environment initialization and spaces configuration.
2. Reset behavior and initial state consistency.
3. Observation space shape, dtype, bounds, and normalization scaling.
4. Action space bounds and clipping.
5. Normalized action-to-torque mapping.
6. Correct use and advancement of the validated physics simulator.
7. Deterministic reset reproducibility with identical random seeds.
8. Disturbance scenario selection (constant, sinusoidal, pulse, canonical).
9. Reward calculation, individual components, and finite values.
10. Episode termination (tumble failure) and truncation (time limits).
11. Actuator saturation handling (torque clamping and wheel despin without termination).
12. Diagnostic telemetry consistency.
13. Full compliance with Stable-Baselines3 environment checker (`check_env`).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import gymnasium as gym

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.simulation.disturbances import ConstantDisturbance, SinusoidalDisturbance, PulseDisturbance


class TestEnvironmentInitialization:
    """Test Suite 1: Environment initialization, observation, and action spaces."""

    def test_env_initialization(self):
        """Verify environment instantiates cleanly with correct space definitions."""
        env = SpacecraftAttitudeEnv()

        # Observation Space
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (3,)
        assert env.observation_space.dtype == np.float32
        assert np.all(env.observation_space.low == -1.0)
        assert np.all(env.observation_space.high == 1.0)

        # Action Space
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        assert env.action_space.dtype == np.float32
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

        # Underlying physics simulator
        assert env.simulator is not None
        assert env.simulator.J == pytest.approx(10.05)
        assert env.max_torque == pytest.approx(1.0)
        env.close()

    def test_gym_make_registration(self):
        """Verify environment is registered and instantiable via gymnasium.make."""
        import src.envs  # registers SpacecraftAttitude-v0
        env = gym.make("SpacecraftAttitude-v0")
        assert env is not None
        obs, info = env.reset()
        assert obs.shape == (3,)
        env.close()


class TestObservationAndActionSpaces:
    """Test Suite 2: Observation normalization and action-torque mapping."""

    def test_observation_normalization_and_bounds(self):
        """Verify exact mathematical normalization of state to [-1, 1] with theta_dot_scale = 2.10."""
        env = SpacecraftAttitudeEnv()

        # Test zero state
        obs, _ = env.reset(options={"initial_state": [0.0, 0.0, 0.0]})
        assert np.allclose(obs, [0.0, 0.0, 0.0], atol=1e-6)

        # Test state at positive limits [pi, 2.10, 523.6]
        obs, _ = env.reset(options={"initial_state": [np.pi, 2.10, env.simulator.max_speed]})
        assert obs[0] == pytest.approx(1.0, abs=1e-5)
        assert obs[1] == pytest.approx(1.0, abs=1e-5)
        assert obs[2] == pytest.approx(1.0, abs=1e-5)

        # Test state at negative limits [-pi, -2.10, -523.6]
        obs, _ = env.reset(options={"initial_state": [-np.pi, -2.10, -env.simulator.max_speed]})
        assert obs[0] == pytest.approx(-1.0, abs=1e-5)
        assert obs[1] == pytest.approx(-1.0, abs=1e-5)
        assert obs[2] == pytest.approx(-1.0, abs=1e-5)

        # Test physical failure threshold (120 deg/s = 2.094395 rad/s) is preserved without premature clipping
        obs_fail, _ = env.reset(options={"initial_state": [0.0, np.radians(120.0), 0.0]})
        assert obs_fail[1] < 1.0
        assert obs_fail[1] == pytest.approx(np.radians(120.0) / 2.10, rel=1e-4)

        # Test clipping when state exceeds scale
        obs_clip, _ = env.reset(options={"initial_state": [2.0 * np.pi, 5.0, 1000.0]})
        assert np.all(obs_clip <= 1.0)
        assert np.all(obs_clip >= -1.0)
        env.close()

    def test_action_to_torque_mapping(self):
        """Verify normalized action in [-1, 1] maps directly to u_cmd in [-u_max, u_max]."""
        env = SpacecraftAttitudeEnv()
        env.reset(options={"initial_state": [0.0, 0.0, 0.0]})

        # Action = +1.0 => u_cmd = +1.0 N*m
        _, _, _, _, info1 = env.step(np.array([1.0], dtype=np.float32))
        assert info1["u_cmd"] == pytest.approx(1.0)

        # Action = -1.0 => u_cmd = -1.0 N*m
        _, _, _, _, info2 = env.step(np.array([-1.0], dtype=np.float32))
        assert info2["u_cmd"] == pytest.approx(-1.0)

        # Action = +0.5 => u_cmd = +0.5 N*m
        _, _, _, _, info3 = env.step(np.array([0.5], dtype=np.float32))
        assert info3["u_cmd"] == pytest.approx(0.5)

        # Action exceeding bounds is clipped
        _, _, _, _, info4 = env.step(np.array([2.5], dtype=np.float32))
        assert info4["u_cmd"] == pytest.approx(1.0)
        env.close()


class TestPhysicsCouplingAndSeeding:
    """Test Suite 3: Physics coupling, deterministic reset, and disturbance scenarios."""

    def test_correct_physics_simulator_coupling(self):
        """Ensure stepping the environment advances the underlying Spacecraft1D dynamics."""
        env = SpacecraftAttitudeEnv()
        env.reset(options={"initial_state": [0.0, 0.0, 0.0]})

        # Positive torque command u = 1.0 N*m
        # J * theta_ddot = -u => theta_ddot < 0, Omega_dot = u / I > 0
        _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))

        assert info["theta_dot_rad_s"] < 0.0, "Positive torque must induce negative angular velocity"
        assert info["wheel_speed_rad_s"] > 0.0, "Positive torque must accelerate wheel positively"
        assert env.simulator.time == pytest.approx(0.01)
        env.close()

    def test_deterministic_seeding_reproducibility(self):
        """Verify that identical random seeds produce identical initial conditions."""
        env = SpacecraftAttitudeEnv()

        obs1, _ = env.reset(seed=12345, options={"randomize": True})
        obs2, _ = env.reset(seed=12345, options={"randomize": True})
        assert np.allclose(obs1, obs2), "Identical seeds must yield identical initial states"

        obs3, _ = env.reset(seed=99999, options={"randomize": True})
        assert not np.allclose(obs1, obs3), "Different seeds should yield different initial states"
        env.close()

    def test_disturbance_scenario_configuration(self):
        """Verify canonical disturbance scenarios can be activated via reset options."""
        env = SpacecraftAttitudeEnv()

        # Constant disturbance scenario
        _, info_c = env.reset(options={"scenario": "constant"})
        assert info_c["scenario_id"] == "constant"
        assert info_c["disturbance_torque"] == pytest.approx(0.02)
        assert info_c["theta_deg"] == pytest.approx(10.0)

        # Sinusoidal disturbance scenario
        _, info_s = env.reset(options={"scenario": "sinusoidal"})
        assert info_s["scenario_id"] == "sinusoidal"
        # At t=0, sin(0) = 0
        assert info_s["disturbance_torque"] == pytest.approx(0.0)

        # Step into sinusoidal wave
        _, _, _, _, info_step = env.step(np.array([0.0], dtype=np.float32))
        assert info_step["disturbance_torque"] > 0.0

        # Pulse disturbance scenario
        _, info_p = env.reset(options={"scenario": "pulse"})
        assert info_p["scenario_id"] == "pulse"
        assert info_p["disturbance_torque"] == pytest.approx(0.0)  # t=0, pulse is at t in [10, 12]
        env.close()


class TestRewardFunctionAndTermination:
    """Test Suite 4: Reward function engineering, termination, and truncation."""

    def test_reward_components_and_finiteness(self):
        """Verify reward components exist, are finite, and match expected penalties."""
        env = SpacecraftAttitudeEnv()
        env.reset(options={"initial_state": [np.radians(10.0), 0.0, 0.0]})

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.isfinite(reward), "Reward must be finite"
            assert "reward_components" in info

            rc = info["reward_components"]
            assert "r_theta" in rc
            assert "r_theta_dot" in rc
            assert "r_effort" in rc
            assert "r_omega" in rc
            assert "r_saturation" in rc
            assert "r_bonus" in rc
            assert np.isfinite(rc["total_reward"])
            assert reward == pytest.approx(rc["total_reward"])

        env.close()

    def test_precision_pointing_bonus(self):
        """Verify precision pointing bonus calculates smooth Gaussian with wheel suppression."""
        env = SpacecraftAttitudeEnv()
        # Initial state exactly at origin
        env.reset(options={"initial_state": [0.0, 0.0, 0.0]})

        _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["is_success"] is True
        # At zero error and zero wheel speed, Gaussian=1.0, suppression=1.0 => bonus = w_bonus = 1.0
        assert info["reward_components"]["r_bonus"] == pytest.approx(env.w_bonus)
        env.close()

    def test_smooth_fine_pointing_reward_continuity(self):
        """Verify smooth Gaussian formulation decreases monotonically without step discontinuity."""
        env = SpacecraftAttitudeEnv()

        # Test attitude error sweep across tolerance boundary
        sig_th = env.tolerance_theta  # 0.10 deg
        angles = np.linspace(0.0, 3.0 * sig_th, 31)
        bonuses = []
        for th in angles:
            _, rc = env._compute_reward(
                theta=th,
                theta_dot=0.0,
                Omega=0.0,
                u_cmd=0.0,
                u_applied=0.0,
                is_torque_sat=False,
                is_wheel_sat=False,
            )
            bonuses.append(rc["r_bonus"])

        bonuses = np.array(bonuses)
        # Check strictly non-increasing
        assert np.all(np.diff(bonuses) <= 0.0), "Gaussian bonus must be monotonically non-increasing with error"
        # Check continuous gradient (no huge step jump at boundary)
        diffs = np.abs(np.diff(bonuses))
        assert np.max(diffs) < 0.15, "Bonus gradient must be smooth with no discrete cliff"
        env.close()

    def test_wheel_saturation_net_negative_reward(self):
        """Verify total step reward is strictly net negative when attempting outward torque at saturation."""
        env = SpacecraftAttitudeEnv()
        max_speed = env.simulator.max_speed
        # Spacecraft at perfect pointing, but wheel at maximum speed
        env.reset(options={"initial_state": [0.0, 0.0, max_speed]})

        # Command accelerating torque (+1.0 N*m): outward torque blocked
        _, reward, _, _, info = env.step(np.array([1.0], dtype=np.float32))

        assert info["is_wheel_saturated"] is True
        assert info["reward_components"]["r_bonus"] == pytest.approx(0.0), "Bonus must be 0 when wheel is saturated"
        assert info["reward_components"]["r_saturation"] <= -1.0, "Wheel sat penalty (-1.0) must be applied"
        assert reward < -0.9, f"Total reward must be strictly net negative under blocked torque, got {reward}"
        env.close()

    def test_distinction_blocked_vs_valid_braking(self):
        """Verify w_wheel_sat is applied to blocked outward torque but NOT to valid despinning torque."""
        env = SpacecraftAttitudeEnv()
        max_speed = env.simulator.max_speed

        # Case A: Blocked accelerating torque at +max_speed
        env.reset(options={"initial_state": [0.0, 0.0, max_speed]})
        _, _, _, _, info_blocked = env.step(np.array([1.0], dtype=np.float32))
        assert info_blocked["is_wheel_saturated"] is True
        assert info_blocked["reward_components"]["r_saturation"] == pytest.approx(-1.0)

        # Case B: Valid braking/despinning torque at +max_speed
        env.reset(options={"initial_state": [0.0, 0.0, max_speed]})
        _, _, _, _, info_braking = env.step(np.array([-1.0], dtype=np.float32))
        assert info_braking["is_wheel_saturated"] is False
        assert info_braking["reward_components"]["r_saturation"] == pytest.approx(0.0)
        assert info_braking["u_applied"] == pytest.approx(-1.0)
        env.close()

    def test_discount_factor_configuration(self):
        """Verify discount factor gamma is set to 0.998 for 5.0 s planning horizon."""
        env = SpacecraftAttitudeEnv()
        assert env.gamma == pytest.approx(0.998)
        # Verify effective horizon tau = -1 / ln(gamma) * dt
        tau = -env.dt / np.log(env.gamma)
        assert tau == pytest.approx(5.0, rel=0.01)
        env.close()

    def test_near_saturation_reset(self):
        """Verify options={'mode': 'near_saturation'} initializes wheel speed in [3800, 4600] RPM."""
        env = SpacecraftAttitudeEnv()
        omega_min_rad = 3800.0 * 2.0 * np.pi / 60.0  # ~397.94 rad/s
        omega_max_rad = 4600.0 * 2.0 * np.pi / 60.0  # ~481.71 rad/s

        for seed in range(20):
            obs, info = env.reset(seed=seed, options={"mode": "near_saturation"})
            assert info["scenario_id"] == "near_saturation_stress"
            abs_omega = abs(info["wheel_speed_rad_s"])
            assert omega_min_rad <= abs_omega <= omega_max_rad + 1e-4
            assert abs(info["wheel_speed_rpm"]) >= 3800.0 - 0.1
            assert abs(info["wheel_speed_rpm"]) <= 4600.0 + 0.1

        env.close()

    def test_nominal_angular_momentum_bound(self):
        """Verify nominal random initialization strictly respects |H_0| <= 3.5 N*m*s."""
        env = SpacecraftAttitudeEnv()
        for seed in range(50):
            obs, info = env.reset(seed=seed, options={"randomize": True})
            H0 = info["total_angular_momentum"]
            # Theoretical max is J * 3 deg/s + I * 500 RPM = 0.5236 + 2.618 = 3.1416 N*m*s
            assert abs(H0) <= 3.5, f"Nominal angular momentum |H0| exceeded bound: {abs(H0)} N*m*s"
            assert abs(info["wheel_speed_rpm"]) <= 500.0 + 1e-3

        env.close()

    def test_episode_truncation_at_step_limit(self):
        """Verify episode truncates when reaching max_episode_steps."""
        env = SpacecraftAttitudeEnv()
        env.max_episode_steps = 10  # short limit for testing
        env.reset()

        for step in range(9):
            _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
            assert truncated is False

        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        assert truncated is True
        env.close()

    def test_tumble_failure_termination(self):
        """Verify episode terminates if attitude angle exceeds failure limit (|theta| > 90 deg)."""
        env = SpacecraftAttitudeEnv()
        # Start at 91 degrees (tumble)
        env.reset(options={"initial_state": [np.radians(91.0), 0.0, 0.0]})

        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        assert terminated is True
        env.close()


class TestActuatorSaturationAndDiagnostics:
    """Test Suite 5: Actuator saturation handling and diagnostic consistency."""

    def test_wheel_saturation_permits_despin_without_termination(self):
        """
        Wheel speed at Omega_max:
        - Positive torque (u > 0) is blocked (u_applied = 0), is_wheel_saturated = True.
        - Episode does NOT terminate.
        - Negative torque (u < 0) is permitted and successfully despins the wheel.
        """
        env = SpacecraftAttitudeEnv()
        max_speed = env.simulator.max_speed
        env.reset(options={"initial_state": [0.0, 0.0, max_speed]})

        # 1. Action = +1.0 (accelerating torque command): blocked
        _, _, term1, _, info1 = env.step(np.array([1.0], dtype=np.float32))
        assert info1["u_cmd"] == pytest.approx(1.0)
        assert info1["u_applied"] == pytest.approx(0.0)
        assert info1["is_wheel_saturated"] is True
        assert term1 is False, "Wheel saturation alone must not terminate the episode"

        # 2. Action = -1.0 (despinning/braking torque command): allowed
        _, _, term2, _, info2 = env.step(np.array([-1.0], dtype=np.float32))
        assert info2["u_cmd"] == pytest.approx(-1.0)
        assert info2["u_applied"] == pytest.approx(-1.0)
        assert info2["wheel_speed_rad_s"] < max_speed
        assert term2 is False
        env.close()

    def test_diagnostic_telemetry_consistency(self):
        """Verify unit conversions and physical telemetry consistency in info dict."""
        env = SpacecraftAttitudeEnv()
        env.reset(options={"initial_state": [np.radians(15.0), 0.1, 100.0]})

        _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))

        assert info["theta_deg"] == pytest.approx(np.degrees(info["theta_rad"]))
        assert info["theta_dot_deg_s"] == pytest.approx(np.degrees(info["theta_dot_rad_s"]))
        assert info["wheel_speed_rpm"] == pytest.approx(info["wheel_speed_rad_s"] * 60.0 / (2.0 * np.pi))

        expected_H = env.simulator.J * info["theta_dot_rad_s"] + env.simulator.I * info["wheel_speed_rad_s"]
        assert info["total_angular_momentum"] == pytest.approx(expected_H)
        env.close()


class TestStableBaselines3Compatibility:
    """Test Suite 6: Stable-Baselines3 environment checker."""

    def test_sb3_check_env(self):
        """
        Execute Stable-Baselines3's check_env to verify 100% adherence to Gymnasium standards.
        """
        from stable_baselines3.common.env_checker import check_env

        env = SpacecraftAttitudeEnv()
        # check_env raises exceptions if environment violates Gymnasium/SB3 specifications
        check_env(env, warn=True)
        env.close()
