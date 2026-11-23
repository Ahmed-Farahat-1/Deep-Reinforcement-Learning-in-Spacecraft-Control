"""
Custom Anti-Symmetric Actor & Policy for TD3 (Twin Delayed DDPG).

Enforces mathematical odd-symmetry:
    mu(-s) = -mu(s)
and exact zero-equilibrium:
    mu(0) == 0.0
by architectural construction:
    mu(s) = 0.5 * [f(s) - f(-s)]
using shared neural-network weights.
"""

from typing import Optional, Union, Dict, Any, Tuple
import torch as th
import numpy as np
import gymnasium as gym

from stable_baselines3.td3.policies import TD3Policy, Actor as TD3Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class AntiSymmetricTD3Actor(TD3Actor):
    """
    Deterministic Actor for TD3 enforcing strict anti-symmetry mu(-s) == -mu(s).
    """

    def forward(self, obs: th.Tensor) -> th.Tensor:
        feat_pos = self.extract_features(obs, self.features_extractor)
        feat_neg = self.extract_features(-obs, self.features_extractor)
        out_pos = self.mu(feat_pos)
        out_neg = self.mu(feat_neg)
        return 0.5 * (out_pos - out_neg)


class AntiSymmetricTD3Policy(TD3Policy):
    """
    TD3 Policy with Twin Q-Critics and Anti-Symmetric Deterministic Actor.
    """

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> TD3Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return AntiSymmetricTD3Actor(**actor_kwargs).to(self.device)


def verify_td3_anti_symmetry(
    actor_or_model: Any,
    obs_dim: int = 4,
    n_samples: int = 100,
    seed: int = 42,
    tol: float = 1e-6,
) -> Tuple[bool, float, float]:
    """
    Verify odd symmetry and zero equilibrium of TD3 policy across sampled states.

    Returns:
    (is_valid, max_symmetry_discrepancy, zero_equilibrium_value)
    """
    rng = np.random.default_rng(seed)
    sampled_obs = rng.uniform(-1.0, 1.0, size=(n_samples, obs_dim)).astype(np.float32)

    max_disc = 0.0
    for obs in sampled_obs:
        if hasattr(actor_or_model, "predict"):
            act_pos, _ = actor_or_model.predict(obs, deterministic=True)
            act_neg, _ = actor_or_model.predict(-obs, deterministic=True)
            disc = float(np.max(np.abs(act_pos + act_neg)))
        else:
            with th.no_grad():
                t_pos = th.as_tensor(obs).unsqueeze(0)
                t_neg = th.as_tensor(-obs).unsqueeze(0)
                a_pos = actor_or_model(t_pos).cpu().numpy()
                a_neg = actor_or_model(t_neg).cpu().numpy()
                disc = float(np.max(np.abs(a_pos + a_neg)))
        if disc > max_disc:
            max_disc = disc

    # Check zero state
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    if hasattr(actor_or_model, "predict"):
        act_zero, _ = actor_or_model.predict(zero_obs, deterministic=True)
        zero_val = float(np.max(np.abs(act_zero)))
    else:
        with th.no_grad():
            t_zero = th.as_tensor(zero_obs).unsqueeze(0)
            act_zero = actor_or_model(t_zero).cpu().numpy()
            zero_val = float(np.max(np.abs(act_zero)))

    is_valid = (max_disc < tol) and (zero_val < tol)
    return is_valid, max_disc, zero_val
