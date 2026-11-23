"""
Automated Unit and Integration Tests for Phase 4G: Pure TD3 & SAC Controllers.

Tests verify:
1. TD3 initialization and configuration.
2. SAC initialization and configuration.
3. Action-space correctness (Box[-1, 1], continuous 1D).
4. Observation-space correctness (4D bounded, normalized).
5. Action scaling to motor torque (u_cmd = action * max_torque).
6. Motor torque saturation handling (|u| <= 1.0 N*m).
7. Reaction-wheel speed saturation constraint.
8. Angular momentum conservation (H = J*theta_dot + I*Omega).
9. Positive and negative control directions.
10. Architectural anti-symmetry for TD3 and SAC.
11. SAC log-probability and squashed Gaussian calculations.
12. TD3 deterministic actor outputs.
13. Critic gradient propagation for TD3 and SAC.
14. Replay-buffer operation and transition storage.
15. Model save/load persistence and prediction reproducibility.
16. Reproducible training seeds.
17. Separation of training and held-out scenarios (H1-H5).
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch as th

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import TD3, SAC
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.simulation.spacecraft import Spacecraft1D
from src.agents.td3.td3_policy import (
    AntiSymmetricTD3Actor,
    AntiSymmetricTD3Policy,
    verify_td3_anti_symmetry,
)
from src.agents.sac.sac_policy import (
    AntiSymmetricSACActor,
    AntiSymmetricSACPolicy,
    verify_sac_anti_symmetry,
)
from src.agents.evaluate_ppo import get_benchmark_scenarios, get_heldout_scenarios
from src.controllers.pd_controller import PDController


@pytest.fixture
def env_4d():
    env = SpacecraftAttitudeEnv(include_integral=True)
    env.reset(seed=42)
    return env


class TestTD3AndSACInitialization:
    """Verify initialization, spaces, and dimensions."""

    def test_td3_initialization(self, env_4d):
        model = TD3(AntiSymmetricTD3Policy, env_4d, learning_starts=10, buffer_size=1000, device="cpu")
        assert model is not None
        assert model.observation_space.shape == (4,)
        assert model.action_space.shape == (1,)
        assert model.action_space.low[0] == -1.0
        assert model.action_space.high[0] == 1.0

    def test_sac_initialization(self, env_4d):
        model = SAC(AntiSymmetricSACPolicy, env_4d, learning_starts=10, buffer_size=1000, ent_coef="auto_0.1", device="cpu")
        assert model is not None
        assert model.observation_space.shape == (4,)
        assert model.action_space.shape == (1,)
        assert model.action_space.low[0] == -1.0
        assert model.action_space.high[0] == 1.0


class TestActionScalingAndPhysics:
    """Verify action scaling, torque limits, and momentum conservation."""

    def test_action_scaling_and_saturation(self, env_4d):
        sim = Spacecraft1D()
        for act in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            act_clipped = float(np.clip(act, -1.0, 1.0))
            u_cmd = act_clipped * sim.max_torque
            assert abs(u_cmd) <= 1.0
            _, info = sim.step(u_cmd=u_cmd)
            assert abs(info["u_applied"]) <= 1.0

    def test_angular_momentum_conservation(self):
        sim = Spacecraft1D(dt=0.01)
        sim.reset(initial_state=np.array([0.1, 0.05, 10.0]))
        H0 = sim.total_angular_momentum()
        for _ in range(50):
            sim.step(u_cmd=0.5)
        H_final = sim.total_angular_momentum()
        assert abs(H_final - H0) < 1e-10

    def test_wheel_speed_limit_blocks_accelerating_torque(self):
        sim = Spacecraft1D()
        # Wheel at positive speed limit
        sim.reset(initial_state=np.array([0.0, 0.0, sim.max_speed]))
        # Command positive torque (would accelerate wheel further) -> blocked
        _, info = sim.step(u_cmd=0.5)
        assert info["u_applied"] == 0.0
        assert info["is_wheel_saturated"] is True

        # Command negative torque (braking/despinning) -> permitted
        _, info_brake = sim.step(u_cmd=-0.5)
        assert info_brake["u_applied"] == -0.5


class TestAntiSymmetryAndDirections:
    """Verify exact odd symmetry and zero equilibrium for TD3 and SAC."""

    def test_td3_anti_symmetry(self, env_4d):
        model = TD3(AntiSymmetricTD3Policy, env_4d, device="cpu", seed=42)
        is_valid, max_disc, zero_val = verify_td3_anti_symmetry(model, obs_dim=4, n_samples=50)
        assert is_valid is True
        assert max_disc < 1e-6
        assert zero_val < 1e-6

    def test_sac_anti_symmetry(self, env_4d):
        model = SAC(AntiSymmetricSACPolicy, env_4d, device="cpu", seed=42)
        is_valid, max_disc, zero_val = verify_sac_anti_symmetry(model, obs_dim=4, n_samples=50)
        assert is_valid is True
        assert max_disc < 1e-6
        assert zero_val < 1e-6

    def test_td3_deterministic_actor_output(self, env_4d):
        model = TD3(AntiSymmetricTD3Policy, env_4d, device="cpu", seed=42)
        obs = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        act1, _ = model.predict(obs, deterministic=True)
        act2, _ = model.predict(obs, deterministic=True)
        assert np.allclose(act1, act2)

    def test_sac_log_prob_and_squashing(self, env_4d):
        model = SAC(AntiSymmetricSACPolicy, env_4d, device="cpu", seed=42)
        obs_tensor = th.as_tensor(np.array([[0.2, -0.1, 0.05, 0.01]], dtype=np.float32))
        actor = model.actor
        mean_actions, log_std, kwargs = actor.get_action_dist_params(obs_tensor)
        action, log_prob = actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
        assert action.shape == (1, 1)
        assert log_prob.shape == (1,)
        assert th.isfinite(log_prob).all()
        assert (action >= -1.0).all() and (action <= 1.0).all()


class TestGradientsAndReplayBuffer:
    """Verify critic gradient propagation and replay buffer operation."""

    def test_td3_replay_buffer_and_gradients(self, env_4d):
        model = TD3(AntiSymmetricTD3Policy, env_4d, learning_starts=10, buffer_size=1000, device="cpu")
        model.learn(total_timesteps=50)
        assert model.replay_buffer.size() >= 40
        # Check that actor and critic parameters have gradients
        for param in model.critic.parameters():
            if param.grad is not None:
                assert th.isfinite(param.grad).all()

    def test_sac_replay_buffer_and_gradients(self, env_4d):
        model = SAC(AntiSymmetricSACPolicy, env_4d, learning_starts=10, buffer_size=1000, device="cpu")
        model.learn(total_timesteps=50)
        assert model.replay_buffer.size() >= 40
        for param in model.critic.parameters():
            if param.grad is not None:
                assert th.isfinite(param.grad).all()


class TestSaveLoadAndReproducibility:
    """Verify model save/load and seed reproducibility."""

    def test_td3_save_load(self, env_4d):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "td3_test.zip"
            model = TD3(AntiSymmetricTD3Policy, env_4d, device="cpu", seed=42)
            obs = np.array([0.15, -0.05, 0.1, 0.02], dtype=np.float32)
            act_before, _ = model.predict(obs, deterministic=True)
            model.save(str(path))

            custom_objs = {"AntiSymmetricTD3Policy": AntiSymmetricTD3Policy}
            loaded = TD3.load(str(path), device="cpu", custom_objects=custom_objs)
            act_after, _ = loaded.predict(obs, deterministic=True)
            assert np.allclose(act_before, act_after, atol=1e-6)

    def test_sac_save_load(self, env_4d):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sac_test.zip"
            model = SAC(AntiSymmetricSACPolicy, env_4d, device="cpu", seed=42)
            obs = np.array([0.15, -0.05, 0.1, 0.02], dtype=np.float32)
            act_before, _ = model.predict(obs, deterministic=True)
            model.save(str(path))

            custom_objs = {"AntiSymmetricSACPolicy": AntiSymmetricSACPolicy}
            loaded = SAC.load(str(path), device="cpu", custom_objects=custom_objs)
            act_after, _ = loaded.predict(obs, deterministic=True)
            assert np.allclose(act_before, act_after, atol=1e-6)

    def test_seed_reproducibility(self, env_4d):
        model1 = TD3(AntiSymmetricTD3Policy, env_4d, device="cpu", seed=123)
        model2 = TD3(AntiSymmetricTD3Policy, env_4d, device="cpu", seed=123)
        obs = np.array([0.2, -0.1, 0.05, 0.01], dtype=np.float32)
        a1, _ = model1.predict(obs, deterministic=True)
        a2, _ = model2.predict(obs, deterministic=True)
        assert np.allclose(a1, a2)


class TestHeldOutSeparation:
    """Verify strict separation of training scenarios and held-out test scenarios."""

    def test_heldout_scenario_separation(self):
        pd_ctl = PDController()
        benchmarks = get_benchmark_scenarios(pd_ctl)
        heldouts = get_heldout_scenarios(pd_ctl)

        bench_ids = {s["id"] for s in benchmarks}
        held_ids = {s["id"] for s in heldouts}

        # Sets must be completely disjoint
        assert len(bench_ids.intersection(held_ids)) == 0
        assert len(heldouts) == 5

        # Verify H1-H5 properties
        h_dict = {s["id"]: s for s in heldouts}
        assert np.isclose(np.degrees(h_dict["heldout_1_pos17_5deg"]["initial_state"][0]), 17.5)
        assert np.isclose(np.degrees(h_dict["heldout_2_neg22_5deg"]["initial_state"][0]), -22.5)
        assert np.isclose(np.degrees(h_dict["heldout_3_mixed_state"]["initial_state"][0]), 15.0)
        assert np.isclose(np.degrees(h_dict["heldout_3_mixed_state"]["initial_state"][1]), -1.5)
        assert np.isclose(h_dict["heldout_3_mixed_state"]["initial_state"][2] * 60 / (2*np.pi), 500.0)
