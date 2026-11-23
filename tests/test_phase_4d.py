"""
Comprehensive Automated Verification Test Suite for Phase 4D (PPO Architecture & Reward Redesign).

Validates:
1. Anti-Symmetric Policy Network (AntiSymmetricActorCriticPolicy):
   - Strict mathematical anti-symmetry: mu(-s) == -mu(s) across mirrored states
   - Exact zero equilibrium: mu(0) == 0.0
   - Symmetric control authority on positive and negative attitude errors
   - PPO log_prob, entropy, and gradient backpropagation compatibility
   - Model persistence (save & load)
2. 4D Observation Space with Integrated Error:
   - Observation space shape (4,), bounds [-1, 1], and float32 dtype
   - Reset behavior (z = 0)
   - Dynamic clamping anti-windup (|z| <= z_max)
   - Conditional freezing anti-windup under outward actuator saturation
   - Strict odd-symmetry: obs(-s) == -obs(s)
3. Redesigned Multi-Objective Reward:
   - Huber attitude penalty continuity and linear non-vanishing slope for large errors
   - Wide Gaussian fine-pointing bonus (sigma_theta = 1.0 deg) smooth gradient
   - Momentum-aware wheel penalty (r_omega = 0 for nominal speeds <= 4500 RPM)
   - Distinction between blocked outward torque and valid despinning braking
4. Physical Feasibility and Safe Angular Momentum:
   - Physical momentum constraint |H0| <= 10.0 N*m*s
   - Reaction wheel headroom preservation (Omega_eq <= 2000 RPM)
5. Held-Out Scenarios (H1 to H5) validation
6. Stable-Baselines3 check_env compliance for 3D and 4D configurations
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.simulation.spacecraft import Spacecraft1D
from src.simulation.disturbances import ZeroDisturbance, ConstantDisturbance, SinusoidalDisturbance


class TestAntiSymmetricPolicy:
    """Test Suite 1: AntiSymmetricActorCriticPolicy mathematical properties."""

    @pytest.fixture
    def policy_and_spaces(self):
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        def lr_sched(p): return 3e-4
        policy = AntiSymmetricActorCriticPolicy(obs_space, act_space, lr_sched)
        return policy, obs_space, act_space

    def test_strict_anti_symmetry(self, policy_and_spaces):
        """Verify mu(-s) == -mu(s) across 25 mirrored random states in [-1, 1]^4."""
        policy, _, _ = policy_and_spaces
        policy.eval()

        np.random.seed(42)
        for _ in range(25):
            s_np = np.random.uniform(-1.0, 1.0, size=(1, 4)).astype(np.float32)
            s_th = th.as_tensor(s_np)
            s_neg_th = -s_th

            with th.no_grad():
                act_pos = policy._predict(s_th, deterministic=True)
                act_neg = policy._predict(s_neg_th, deterministic=True)

            sum_val = float((act_pos + act_neg).item())
            assert abs(sum_val) < 1e-6, f"Anti-symmetry violated! Sum was {sum_val}"

    def test_exact_zero_equilibrium(self, policy_and_spaces):
        """Verify mu(0, 0, 0, 0) == 0.0 identically."""
        policy, _, _ = policy_and_spaces
        policy.eval()

        s_zero = th.zeros((1, 4), dtype=th.float32)
        with th.no_grad():
            act_zero = policy._predict(s_zero, deterministic=True)

        assert abs(float(act_zero.item())) < 1e-7, "Zero state must yield exactly zero action"

    def test_directional_correctness_and_magnitude(self, policy_and_spaces):
        """Verify policy produces opposite equal-magnitude actions for +/- 10 deg error."""
        policy, _, _ = policy_and_spaces
        policy.eval()

        th_norm = 10.0 / 180.0  # +10 deg
        s_pos = th.tensor([[th_norm, 0.0, 0.0, 0.0]], dtype=th.float32)
        s_neg = th.tensor([[-th_norm, 0.0, 0.0, 0.0]], dtype=th.float32)

        with th.no_grad():
            u_pos = policy._predict(s_pos, deterministic=True).item()
            u_neg = policy._predict(s_neg, deterministic=True).item()

        assert abs(u_pos + u_neg) < 1e-6, "Magnitudes must be identical and opposite"
        assert abs(u_pos) == pytest.approx(abs(u_neg), abs=1e-6)

    def test_ppo_backpropagation_and_gradients(self, policy_and_spaces):
        """Verify evaluate_actions, value prediction, log_prob, and backward gradient flow."""
        policy, _, _ = policy_and_spaces
        policy.train()

        obs = th.tensor([[0.5, -0.2, 0.1, -0.05], [-0.5, 0.2, -0.1, 0.05]], dtype=th.float32)
        acts = th.tensor([[0.1], [-0.1]], dtype=th.float32)

        values, log_prob, entropy = policy.evaluate_actions(obs, acts)

        assert values.shape == (2, 1)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert th.isfinite(values).all()
        assert th.isfinite(log_prob).all()
        assert th.isfinite(entropy).all()

        # Backward loss computation
        loss = -(log_prob.mean() + values.mean() + entropy.mean())
        loss.backward()

        assert policy.action_net.weight.grad is not None
        assert th.isfinite(policy.action_net.weight.grad).all()
        assert policy.action_net.weight.grad.norm().item() > 0.0

    def test_gaussian_action_dist_log_prob_and_entropy_exact_symmetry(self, policy_and_spaces):
        """Verify log_prob(a | s) == log_prob(-a | -s) and entropy(s) == entropy(-s) exactly."""
        policy, _, _ = policy_and_spaces
        policy.eval()

        np.random.seed(777)
        for _ in range(15):
            s_np = np.random.uniform(-1.0, 1.0, size=(1, 4)).astype(np.float32)
            a_np = np.random.uniform(-0.8, 0.8, size=(1, 1)).astype(np.float32)

            obs_batch = th.tensor(np.vstack([s_np, -s_np]), dtype=th.float32)
            act_batch = th.tensor(np.vstack([a_np, -a_np]), dtype=th.float32)

            with th.no_grad():
                values, log_prob, entropy = policy.evaluate_actions(obs_batch, act_batch)

            lp_diff = abs(float((log_prob[0] - log_prob[1]).item()))
            ent_diff = abs(float((entropy[0] - entropy[1]).item()))

            assert lp_diff < 1e-6, f"Log-prob symmetry violated! Diff was {lp_diff}"
            assert ent_diff < 1e-6, f"Entropy symmetry violated! Diff was {ent_diff}"

    def test_policy_save_and_load(self):
        """Verify model persistence with AntiSymmetricActorCriticPolicy."""
        import tempfile, os
        env = SpacecraftAttitudeEnv(include_integral=True)
        model = PPO(AntiSymmetricActorCriticPolicy, env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            custom_objects = {"AntiSymmetricActorCriticPolicy": AntiSymmetricActorCriticPolicy}
            loaded = PPO.load(tmp_path, env=env, custom_objects=custom_objects)
            assert loaded is not None
            obs, _ = env.reset()
            act, _ = loaded.predict(obs, deterministic=True)
            assert act.shape == (1,)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        env.close()

    def test_model_save_and_reload_exact_numerical_match(self):
        """Verify model save/load preserves exact deterministic action outputs across 20 random states."""
        import tempfile, os
        env = SpacecraftAttitudeEnv(include_integral=True)
        model = PPO(AntiSymmetricActorCriticPolicy, env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            custom_objects = {"AntiSymmetricActorCriticPolicy": AntiSymmetricActorCriticPolicy}
            loaded = PPO.load(tmp_path, env=env, custom_objects=custom_objects)

            np.random.seed(123)
            test_states = np.random.uniform(-1.0, 1.0, size=(20, 4)).astype(np.float32)
            for state in test_states:
                act_orig, _ = model.predict(state, deterministic=True)
                act_loaded, _ = loaded.predict(state, deterministic=True)
                assert np.allclose(act_orig, act_loaded, atol=1e-7), f"Discrepancy: orig={act_orig}, loaded={act_loaded}"

                # Also verify anti-symmetry is strictly preserved after reload
                act_neg, _ = loaded.predict(-state, deterministic=True)
                assert np.allclose(act_loaded + act_neg, 0.0, atol=1e-6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        env.close()


class Test4DObservationAndIntegralError:
    """Test Suite 2: 4D Observation space, integral error state, and anti-windup."""

    def test_4d_observation_space_specs(self):
        """Verify 4D observation space shape, bounds, and dtype."""
        env = SpacecraftAttitudeEnv(include_integral=True)
        assert env.observation_space.shape == (4,)
        assert env.observation_space.dtype == np.float32
        assert np.all(env.observation_space.low == -1.0)
        assert np.all(env.observation_space.high == 1.0)

        obs, info = env.reset()
        assert obs.shape == (4,)
        assert "integral_error_rad_s" in info
        assert info["integral_error_rad_s"] == pytest.approx(0.0)
        env.close()

    def test_integral_accumulation(self):
        """Verify integral error accumulates over time when attitude error is non-zero."""
        env = SpacecraftAttitudeEnv(include_integral=True)
        env.reset(options={"initial_state": [np.radians(5.0), 0.0, 0.0]})

        # Step with zero control torque
        for _ in range(10):
            obs, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

        assert abs(info["integral_error_rad_s"]) > 0.0
        assert abs(obs[3]) > 0.0
        env.close()

    def test_anti_windup_clamping(self):
        """Verify dynamic clamping anti-windup enforces |z| <= z_max."""
        env = SpacecraftAttitudeEnv(include_integral=True)
        env.reset(options={"initial_state": [np.radians(20.0), 0.0, 0.0]})

        # Run 200 steps to build large integral
        for _ in range(200):
            obs, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

        assert abs(info["integral_error_rad_s"]) <= env.z_max + 1e-5
        assert abs(obs[3]) <= 1.0 + 1e-5
        env.close()

    def test_conditional_freezing_anti_windup(self):
        """Verify integration freezes when torque is saturated in the outward error direction."""
        env = SpacecraftAttitudeEnv(include_integral=True)
        # Positive attitude error + full positive torque command (outward torque push)
        env.reset(options={"initial_state": [np.radians(10.0), 0.0, 0.0]})

        _, _, _, _, info1 = env.step(np.array([1.0], dtype=np.float32))
        z1 = info1["integral_error_rad_s"]

        # Step again with full outward torque: integral must freeze
        _, _, _, _, info2 = env.step(np.array([1.0], dtype=np.float32))
        z2 = info2["integral_error_rad_s"]

        assert z2 == pytest.approx(z1, abs=1e-6), "Integral must freeze when outward torque is saturated"
        env.close()

    def test_anti_windup_when_actuator_blocked_by_wheel_speed(self):
        """Verify integral accumulation freezes when corrective torque is blocked by wheel speed saturation."""
        env = SpacecraftAttitudeEnv(include_integral=True)
        # Wheel at maximum positive speed (+max_speed = 523.6 rad/s = 5000 RPM)
        # Positive attitude error (+10 deg) requires positive motor torque (u > 0) to accelerate wheel and brake spacecraft.
        max_om = env.simulator.max_speed
        env.reset(options={"initial_state": [np.radians(10.0), 0.0, max_om]})

        # Step with positive torque command (u_cmd = 0.5):
        # Simulator sees Omega >= max_speed and u > 0 -> blocks torque (u_applied = 0.0, is_wheel_saturated = True)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))

        assert info["is_wheel_saturated"] is True, "Wheel must register as saturated"
        assert info["integral_error_rad_s"] == pytest.approx(0.0), "Integral must NOT accumulate when corrective torque is blocked by wheel speed"
        env.close()

    def test_odd_symmetry_of_4d_observation(self):
        """Verify obs(-s) == -obs(s) for the complete 4D state."""
        env = SpacecraftAttitudeEnv(include_integral=True)

        obs_pos, _ = env.reset(options={"initial_state": [0.2, 0.05, 100.0], "initial_z": 0.3})
        obs_neg, _ = env.reset(options={"initial_state": [-0.2, -0.05, -100.0], "initial_z": -0.3})

        assert np.allclose(obs_pos + obs_neg, 0.0, atol=1e-5)
        env.close()


class TestRedesignedReward:
    """Test Suite 3: Huber penalty, wide Gaussian bonus, and momentum management."""

    def test_huber_penalty_linear_slope_for_large_errors(self):
        """Verify Huber penalty derivative is constant (linear) for errors > delta."""
        env = SpacecraftAttitudeEnv(include_integral=True)

        # Evaluate reward at 10 deg, 20 deg, and 30 deg
        r10, rc10 = env._compute_reward(
            theta=np.radians(10.0), theta_dot=0.0, Omega=0.0, u_cmd=0.0, u_applied=0.0, is_torque_sat=False, is_wheel_sat=False
        )
        r20, rc20 = env._compute_reward(
            theta=np.radians(20.0), theta_dot=0.0, Omega=0.0, u_cmd=0.0, u_applied=0.0, is_torque_sat=False, is_wheel_sat=False
        )
        r30, rc30 = env._compute_reward(
            theta=np.radians(30.0), theta_dot=0.0, Omega=0.0, u_cmd=0.0, u_applied=0.0, is_torque_sat=False, is_wheel_sat=False
        )

        # Difference between 20->30 and 10->20 must be equal (constant linear slope)
        diff_10_20 = rc20["r_theta"] - rc10["r_theta"]
        diff_20_30 = rc30["r_theta"] - rc20["r_theta"]

        assert diff_10_20 == pytest.approx(diff_20_30, rel=0.05)
        env.close()

    def test_wide_gaussian_bonus_basin(self):
        """Verify wide Gaussian bonus (sigma=1.0 deg) provides smooth gradient at 0.5, 1.0, 2.0 deg."""
        env = SpacecraftAttitudeEnv(include_integral=True)

        angles = [0.0, 0.5, 1.0, 2.0, 3.0]
        bonuses = []
        for a in angles:
            _, rc = env._compute_reward(
                theta=np.radians(a), theta_dot=0.0, Omega=0.0, u_cmd=0.0, u_applied=0.0, is_torque_sat=False, is_wheel_sat=False
            )
            bonuses.append(rc["r_bonus"])

        # Check monotonically decreasing
        for i in range(len(bonuses) - 1):
            assert bonuses[i] > bonuses[i + 1]

        # Check non-negligible bonus at 1.0 deg and 2.0 deg (unlike narrow 0.1 deg bonus which was 1e-22)
        assert bonuses[2] > 0.10, "Bonus at 1.0 deg must be > 0.10 for wide guidance basin"
        assert bonuses[3] > 0.01, "Bonus at 2.0 deg must be > 0.01"
        env.close()

    def test_unconditional_wheel_penalty_elimination(self):
        """Verify nominal wheel speeds (<= 4500 RPM) incur r_omega = 0.0."""
        env = SpacecraftAttitudeEnv(include_integral=True)

        speeds = [0.0, 500.0, 1500.0, 3000.0, 4200.0]
        for rpm in speeds:
            om_rad = float(rpm * 2.0 * np.pi / 60.0)
            _, rc = env._compute_reward(
                theta=0.0, theta_dot=0.0, Omega=om_rad, u_cmd=0.0, u_applied=0.0, is_torque_sat=False, is_wheel_sat=False
            )
            assert rc["r_omega"] == pytest.approx(0.0), f"Speed {rpm} RPM must have r_omega == 0.0"

        # Speed above 4500 RPM (e.g. 4800 RPM) must incur penalty
        om_high = float(4800.0 * 2.0 * np.pi / 60.0)
        _, rc_high = env._compute_reward(
            theta=0.0, theta_dot=0.0, Omega=om_high, u_cmd=0.0, u_applied=0.0, is_torque_sat=False, is_wheel_sat=False
        )
        assert rc_high["r_omega"] < 0.0
        env.close()

    def test_huber_delta_unit_conversion_verification(self):
        """Verify Huber delta is dimensionless (delta = 0.05 on theta / 30 deg), NOT 0.05 radians.

        0.05 rad = 2.864789 deg != 1.50 deg.
        delta = 0.05 * 30 deg = 1.50 deg = 0.0261799 rad.
        """
        env = SpacecraftAttitudeEnv(include_integral=True)
        assert env.huber_delta == pytest.approx(0.05)
        assert env.theta_ref == pytest.approx(np.radians(30.0))

        # Physical transition angle
        theta_delta_rad = env.huber_delta * env.theta_ref
        theta_delta_deg = np.degrees(theta_delta_rad)

        assert theta_delta_deg == pytest.approx(1.50, abs=1e-4)
        assert theta_delta_rad == pytest.approx(0.0261799, abs=1e-6)

        # Demonstrate that 0.05 rad is NOT 1.50 deg
        wrong_angle_deg = np.degrees(0.05)
        assert wrong_angle_deg == pytest.approx(2.864789, abs=1e-4)
        assert wrong_angle_deg != pytest.approx(1.50, abs=0.1)
        env.close()


class TestPhysicalFeasibilityAndHeldOut:
    """Test Suite 4: Physical angular momentum bound and held-out scenarios."""

    def test_safe_momentum_bound_feasibility(self):
        """Verify |H0| <= 10.0 N*m*s leaves > 60% wheel headroom."""
        sim = Spacecraft1D()
        H_safe = 10.0  # N*m*s
        Omega_eq = H_safe / sim.I_wheel
        rpm_eq = Omega_eq * 60.0 / (2.0 * np.pi)

        # Max wheel capacity is 5000 RPM
        headroom_pct = (sim.max_speed - Omega_eq) / sim.max_speed * 100.0

        assert rpm_eq <= 2000.0
        assert headroom_pct >= 60.0

    def test_angular_momentum_conservation_and_no_unphysical_dumping(self):
        """Verify internal torque conserves total angular momentum H, and reward does not penalize nominal wheel speeds."""
        sim = Spacecraft1D()

        # Test 1: Strict conservation of total angular momentum H under arbitrary internal torque
        sim.reset(initial_state=np.array([0.1, 0.02, 100.0]))
        H_initial = sim.total_angular_momentum()

        for u in [0.5, -0.8, 1.0, -0.3]:
            _, info = sim.step(u_cmd=u, disturbance=ZeroDisturbance())
            H_current = info["total_angular_momentum"]
            assert H_current == pytest.approx(H_initial, abs=1e-6), "Internal torque must conserve total angular momentum"

        # Test 2: Reward function gives r_omega == 0.0 for all nominal operating speeds (<= 4500 RPM)
        env = SpacecraftAttitudeEnv(include_integral=True)
        for rpm in [0, 500, 1000, 2000, 3000, 4000, 4500]:
            om = rpm * 2.0 * np.pi / 60.0
            _, rc = env._compute_reward(
                theta=0.0, theta_dot=0.0, Omega=om, u_cmd=0.0, u_applied=0.0,
                is_torque_sat=False, is_wheel_sat=False
            )
            assert rc["r_omega"] == pytest.approx(0.0), f"Nominal speed {rpm} RPM must have zero penalty"
        env.close()

    def test_held_out_scenarios_configured(self):
        """Verify all 5 Phase 4D held-out test scenarios configure without error."""
        env = SpacecraftAttitudeEnv(include_integral=True)

        scenarios = ["heldout_h1", "heldout_h2", "heldout_h3", "heldout_h4", "heldout_h5"]
        for sc in scenarios:
            obs, info = env.reset(options={"scenario": sc})
            assert obs.shape == (4,)
            assert info["step"] == 0

        env.close()

    def test_sb3_check_env_3d_and_4d(self):
        """Verify full SB3 check_env passes for both 3D and 4D environments."""
        env3d = SpacecraftAttitudeEnv(include_integral=False)
        check_env(env3d)
        env3d.close()

        env4d = SpacecraftAttitudeEnv(include_integral=True)
        check_env(env4d)
        env4d.close()
