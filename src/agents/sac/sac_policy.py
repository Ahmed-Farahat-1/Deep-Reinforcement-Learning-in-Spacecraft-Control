"""
Custom Anti-Symmetric Actor & Policy for SAC (Soft Actor-Critic).

Enforces mathematical odd-symmetry for policy mean:
    mu(-s) = -mu(s)
and even-symmetry for exploration log-std:
    log_std(-s) = log_std(s)
yielding an odd-symmetric deterministic action:
    tanh(mu(-s)) = -tanh(mu(s))
and exact zero equilibrium:
    tanh(mu(0)) == 0.0
by architectural construction using shared neural-network weights.
"""

from typing import Optional, Union, Dict, Any, Tuple
import torch as th
import numpy as np
import gymnasium as gym

from stable_baselines3.sac.policies import SACPolicy, Actor as SACActor, LOG_STD_MAX, LOG_STD_MIN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class AntiSymmetricSACActor(SACActor):
    """
    Gaussian Actor for SAC enforcing odd-symmetric mean and even-symmetric log-std.
    """

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        feat_pos = self.extract_features(obs, self.features_extractor)
        feat_neg = self.extract_features(-obs, self.features_extractor)

        latent_pos = self.latent_pi(feat_pos)
        latent_neg = self.latent_pi(feat_neg)

        # Odd mean: mu(-s) == -mu(s)
        mean_actions = 0.5 * (self.mu(latent_pos) - self.mu(latent_neg))

        # Even log_std: log_std(-s) == log_std(s)
        log_std = 0.5 * (self.log_std(latent_pos) + self.log_std(latent_neg))
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean_actions, log_std, {}


class AntiSymmetricSACPolicy(SACPolicy):
    """
    SAC Policy with Twin Q-Critics, Entropy Temperature Tuning, and Anti-Symmetric Gaussian Actor.
    """

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SACActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return AntiSymmetricSACActor(**actor_kwargs).to(self.device)


def verify_sac_anti_symmetry(
    actor_or_model: Any,
    obs_dim: int = 4,
    n_samples: int = 100,
    seed: int = 42,
    tol: float = 1e-6,
) -> Tuple[bool, float, float]:
    """
    Verify odd symmetry and zero equilibrium of SAC deterministic policy across sampled states.

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
                a_pos = actor_or_model(t_pos, deterministic=True).cpu().numpy()
                a_neg = actor_or_model(t_neg, deterministic=True).cpu().numpy()
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
            act_zero = actor_or_model(t_zero, deterministic=True).cpu().numpy()
            zero_val = float(np.max(np.abs(act_zero)))

    is_valid = (max_disc < tol) and (zero_val < tol)
    return is_valid, max_disc, zero_val
