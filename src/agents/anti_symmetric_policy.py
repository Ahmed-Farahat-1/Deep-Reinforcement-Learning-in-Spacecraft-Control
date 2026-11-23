"""
Anti-Symmetric Policy Network for Spacecraft Attitude Stabilization.

This module implements an anti-symmetric actor-critic policy (AntiSymmetricActorCriticPolicy)
for Stable-Baselines3 PPO. It enforces the fundamental physical symmetry:
    u(-theta, -theta_dot, -Omega, -z) = -u(theta, theta_dot, Omega, z)
    u(0, 0, 0, 0) = 0.0

By computing the mean action as:
    mu(s) = 0.5 * [f(s) - f(-s)]
where f(s) is the neural network actor forward pass with shared weights, the policy
guarantees odd symmetry by construction without requiring data augmentation or
penalty losses. The standard value function V(s) is maintained for the Critic,
and a state-independent symmetric Gaussian exploration sigma is used.
"""

from typing import Optional, Tuple, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class AntiSymmetricActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy enforcing anti-symmetric mean action:
        mu(s) = 0.5 * (f(s) - f(-s))

    Guarantees:
        1. mu(-s) = -mu(s) for all states s in R^4.
        2. mu(0) = 0 identically (exact zero-error equilibrium torque).
        3. Identical control authority and symmetric response in both directions (+ / -).
        4. Standard Critic V(s) and standard PPO loss compatibility with exact gradients.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _forward_actor(self, obs: th.Tensor) -> th.Tensor:
        """
        Evaluate raw forward pass through actor feature extractor and MLP.

        :param obs: Observation tensor on policy device.
        :return: Raw mean action tensor f(obs).
        """
        if self.share_features_extractor:
            features = super().extract_features(obs, self.features_extractor)
        else:
            features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self.action_net(latent_pi)

    def _get_anti_symmetric_mean(self, obs: th.Tensor) -> th.Tensor:
        """
        Compute anti-symmetric mean action:
            mu(s) = 0.5 * [f(s) - f(-s)]

        :param obs: Observation tensor on policy device.
        :return: Anti-symmetric mean action tensor.
        """
        mean_pos = self._forward_actor(obs)
        mean_neg = self._forward_actor(-obs)
        return 0.5 * (mean_pos - mean_neg)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all networks (actor and critic).

        :param obs: Observation tensor.
        :param deterministic: Whether to sample or use deterministic mean actions.
        :return: action, value, and log probability of the action.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)

        mean_actions = self._get_anti_symmetric_mean(obs)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy given observations.

        :param obs: Observation tensor.
        :param actions: Actions tensor.
        :return: estimated value, log likelihood of taking actions, and entropy.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)

        mean_actions = self._get_anti_symmetric_mean(obs)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given observations.

        :param obs: Observation tensor.
        :return: The action distribution.
        """
        mean_actions = self._get_anti_symmetric_mean(obs)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to policy for a given observation.

        :param observation: Observation tensor.
        :param deterministic: Whether to use deterministic or sampled actions.
        :return: Action tensor.
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)
